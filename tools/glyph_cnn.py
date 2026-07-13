"""dev-only learned identity matcher: a small metric/embedding cnn on the extracted glyph.

phase-2 of the matcher work (classical ncc is exhausted at ~58% real top1, see matching-method-plan).
trains a SHARED (siamese) encoder so f(extracted glyph) sits near f(clean sprite) for the same icon,
then production does nearest-cosine over a cached bank of sprite embeddings. metric not classifier so
the self-updating wiki library just needs new sprites embedded, no retrain.

torch is DEV-ONLY (a cuda build, not shipped). the runtime is cv2.dnn on the exported onnx (opencv
4.6.0 is old, so the export stays basic-ops and the l2-norm happens in numpy after the forward, not
in-graph). the encoder keeps a 6x6 spatial grid instead of global-average-pooling because gap
discarded layout and only tied ncc before (matching-method-plan frozen-cnn lesson).

subcommands:
train: train + select on held-out synth + export best onnx; metrics persist to
<out>_metrics.json next to the model, flushed every eval
plot: render that json to <out>_metrics.png (loss + synth-val top1 curves)
"""

import os
# torch's libomp and mkl's libiomp5md both load an openmp runtime and abort on the duplicate; this
# is the standard workaround, safe for our workload (dev-only). must be set before numpy/torch
# import their omp runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # tools/ for synth_glyphs + eval_matchers

INPUT_RES = 96          # matches synth_glyphs.INPUT_RES (the encoder input side)
EMB_DIM = 128           # l2-normalized embedding width
ONNX_PATH = ROOT / "data" / "models" / "glyph_encoder.onnx"

import torch                      # dev-only, heavy import kept below the light constants
import torch.nn as nn
import torch.nn.functional as F
import onnx


class GlyphEncoder(nn.Module):
    """small from-scratch cnn, (B,3,96,96) bgr in [0,1] -> (B,128) RAW embedding (pre l2-norm).

    conv stack halves the map four times to a 6x6 grid then a 1x1 neck trims channels before one
    linear projection, so the embedding still encodes WHERE features are (near-dups like medkit vs
    toolbox differ in local detail a gap'd embedding averages away). ~545k params. the l2-norm is
    done by the caller (torch for the loss, numpy after cv2.dnn at inference) so the exported graph
    stays basic ops only for opencv 4.6.0.
    """

    def __init__(self, emb_dim=EMB_DIM, p_drop=0.1):
        super().__init__()

        def block(cin, cout):
            # conv -> bn -> relu -> maxpool2, standard downsampling unit (all opencv 4.6.0 parses)
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),           # 96 -> 48
            block(32, 64),          # 48 -> 24
            block(64, 128),         # 24 -> 12
            block(128, 128),        # 12 -> 6
            nn.Conv2d(128, 64, 1, bias=False),      # 1x1 neck, cut fc width, keep the 6x6 grid
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(p_drop)              # fight synth overfit, off in eval/export
        self.head = nn.Linear(64 * 6 * 6, emb_dim)  # flatten the 6x6 grid -> emb, exports as Gemm

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(self.drop(x))              # raw embedding, l2-norm applied outside


def _strip_identity_nodes(model):
    """delete Identity nodes and rewire their consumers to the source tensor. constant folding fuses
    bn into the convs but then aliases duplicate fused constants (e.g. all-zero biases at random init)
    through Identity, and opencv 4.6.0's dnn importer asserts on an Identity fed by an initializer.
    a general fix (any Identity, any weights) so the export is deterministic, not dependent on the
    trained bn stats happening to avoid the dedup."""
    g = model.graph
    alias = {n.output[0]: n.input[0] for n in g.node if n.op_type == "Identity"}

    def resolve(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    keep = [n for n in g.node if n.op_type != "Identity"]
    for n in keep:
        for i, inp in enumerate(n.input):
            n.input[i] = resolve(inp)                # point past any aliasing Identity
    del g.node[:]
    g.node.extend(keep)
    return model


def export_onnx(model, path=ONNX_PATH, res=INPUT_RES):
    """export the encoder to onnx for the cv2.dnn runtime. fixed batch=1 + no dynamic axes because
    the old opencv importer wants fully static shapes (production forwards one glyph at a time anyway,
    bank build is cached). eval mode folds bn into constants and drops dropout. opset 12, basic ops,
    l2-norm intentionally left OUT of the graph (done in numpy)."""
    model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, res, res, device=next(model.parameters()).device)   # match the model's device
    # dynamo=False forces the legacy TorchScript exporter. torch 2.9+ defaults to the new
    # torch.export/onnxscript path, which needs onnxscript (not installed) and can emit ops opencv
    # 4.6.0 chokes on; the legacy exporter emits the plain conv/bn/relu/pool/gemm graph cv2.dnn parses.
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["glyph"], output_names=["emb"],
        opset_version=12, do_constant_folding=True, dynamo=False,
    )
    m = _strip_identity_nodes(onnx.load(str(path)))     # opencv 4.6.0 can't parse Identity-on-initializer
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return path


# ------------------------------------------------------------------------------- training data

import synth_glyphs as SG          # dev-only, reuses the real extraction path so train==test glyphs


def _resize_u8(img, res=INPUT_RES):
    """glyph/anchor (bgr) -> (res,res,3) uint8. same INTER_AREA resize as to_input, kept uint8 in
    the pool to quarter the ram (converted to float on the gpu per batch)."""
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)


def build_pool(rows, k_variants, seed, res=INPUT_RES):
    """pre-generate the training pool once, in ram. A = one clean color anchor per class; Q = k
    augmented extracted-glyph variants per class, each a fresh render_node -> real isolate ->
    normalize_glyph -> aug. shapes A (n,res,res,3) uint8, Q (n,k,res,res,3) uint8.

    a fixed pool (not on-the-fly) keeps the training loop pure-gpu and deterministic; k heavy-aug
    views per class is plenty for a 545k-param net, and model selection on FRESH synth catches
    memorization. rarities cycle across the k variants so each class is seen on several disks."""
    n = len(rows)
    rng = np.random.default_rng(seed)
    A = np.zeros((n, res, res, 3), np.uint8)
    Q = np.zeros((n, k_variants, res, res, 3), np.uint8)
    for i, row in enumerate(tqdm(rows, desc=f"pool k={k_variants}")):
        A[i] = _resize_u8(SG.gallery_glyph(row["file"], color=True), res)
        for k in range(k_variants):
            g = None
            for _ in range(4):                       # normalize_glyph rarely returns None, retry
                g = SG.make_synth_glyph(
                    row["file"], SG.rarity_for_row(row, k, rng), rng, category=row.get("category")
                )
                if g is not None:
                    break
            Q[i, k] = _resize_u8(g, res) if g is not None else A[i]   # fall back to the anchor
    return A, Q


def val_synth(rows, seed, res=INPUT_RES):
    """held-out synth val queries: one FRESH synth glyph per class (different seed than the pool),
    for model selection. (n,res,res,3) uint8. selecting on synth keeps the real labels a clean test
    the model never saw, the anti-overfit discipline from the plan."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    V = np.zeros((n, res, res, 3), np.uint8)
    for i, row in enumerate(tqdm(rows, desc="val synth")):
        g = None
        for _ in range(4):
            g = SG.make_synth_glyph(
                row["file"], SG.rarity_for_row(row, i, rng), rng, category=row.get("category")
            )
            if g is not None:
                break
        V[i] = _resize_u8(g, res) if g is not None else _resize_u8(SG.gallery_glyph(row["file"]), res)
    return V


def _confusion_neighbors(Tz, k=8):
    """top-k nearest classes per class by ncc-template cosine (Tz is the z-normed sprite bank from
    detect). builds hard-negative family batches so near-dup glyphs (medkits, toolboxes, reagents)
    land in the same batch and become each other's negatives.
    fixed pixel-space families: only the warmup source now, see _embedding_neighbors."""
    S = Tz @ Tz.T                            # (n,n) cosine, cheap at n~1471
    np.fill_diagonal(S, -1.0)
    return np.argsort(-S, axis=1)[:, :k]     # (n,k) neighbor class indices


@torch.no_grad()
def _embedding_neighbors(model, A_u8, dev, k=8, skip=1, bs=512):
    """top-k confusable classes per class in the MODEL's own embedding space, (n,k) like
    _confusion_neighbors so it drops straight into _sample_classes.

    the point: _confusion_neighbors mines pixel-space ncc, i.e. the confusions of the matcher we
    REPLACED (55% real top1 vs the cnn's 83%), and it's computed once and frozen for the whole run.
    so half the batches harden against the wrong adversary while the pairs the model actually gets
    wrong may never share a batch. mining the live embedding instead closes the loop: negatives track
    what the model confuses NOW, and move on as it fixes them. that's aimed squarely at the symptom
    -- real top5 98.6% vs top1 83.1% means the answer is in the shortlist and only the ranking among
    near neighbours is weak, which is exactly what hard negatives train (and what trips the ocr gate).

    skip drops the nearest few (semi-hard): always feeding the single hardest negative destabilises
    infonce. costs one forward pass over the sprite bank, ~ms, so refresh cadence is free.
    """
    model.eval()
    embs = [F.normalize(model(_to_gpu(A_u8[i:i + bs], dev)), dim=1)
            for i in range(0, len(A_u8), bs)]
    S = torch.cat(embs) @ torch.cat(embs).t()          # (n,n) cosine
    S.fill_diagonal_(-1.0)
    nn = S.topk(k + skip, dim=1).indices[:, skip:]     # drop the `skip` nearest, keep the next k
    model.train()
    return nn.cpu().numpy()


# ------------------------------------------------------------------------------- training

def _sample_classes(n, N, nn, p_family, rng):
    """N distinct class indices for one batch. with prob p_family, grow the batch from a few seed
    classes plus their confusion neighbors (hard negatives), else a plain random distinct set."""
    if nn is not None and rng.random() < p_family:
        fam_k = nn.shape[1]
        picked, seen = [], set()
        for s in rng.choice(n, size=max(1, N // (fam_k + 1)), replace=False):
            for c in (int(s), *nn[s].tolist()):
                if c not in seen:
                    seen.add(c)
                    picked.append(c)
        while len(picked) < N:                   # top up to N distinct with randoms
            c = int(rng.integers(n))
            if c not in seen:
                seen.add(c)
                picked.append(c)
        return np.array(picked[:N])
    return rng.choice(n, size=N, replace=False)


def _to_gpu(x_u8, dev):
    """(N,res,res,3) uint8 hwc bgr -> (N,3,res,res) float [0,1] on dev, matching to_input's scaling."""
    return torch.from_numpy(x_u8).to(dev).float().div_(255.0).permute(0, 3, 1, 2).contiguous()


def _light_aug(x, bright, noise):
    """cheap on-gpu aug to multiply pool variety for free: per-sample brightness scale + additive
    noise. no flips (glyphs are not mirror-invariant: key teeth, directional tools). kept light so
    the anchor branch stays close to the clean bank it embeds from at inference."""
    if bright:
        b = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * bright
        x = x * b
    if noise:
        x = x + torch.randn_like(x) * noise
    return x.clamp_(0.0, 1.0)


@torch.no_grad()
def _embed(model, x_u8, dev, bs=512):
    """embed a stack of uint8 glyphs to l2-normed rows (M,128) via the torch model in eval mode
    (bn running stats, dropout off) so val matches the exported inference path."""
    model.eval()
    outs = []
    for i in range(0, len(x_u8), bs):
        e = model(_to_gpu(x_u8[i:i + bs], dev))
        outs.append(F.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(outs)


def _synth_top1(model, A_u8, V_u8, dev):
    """held-out synth top-1: embed the clean anchor bank + fresh val queries with current weights,
    nearest-cosine, fraction correct. the model-selection signal (real stays untouched)."""
    bank = _embed(model, A_u8, dev)
    q = _embed(model, V_u8, dev)
    pred = (q @ bank.T).argmax(1)
    return float((pred == np.arange(len(A_u8))).mean())


def _metrics_path(out):
    """where a run's metrics json lives, next to the model it belongs to."""
    out = Path(out)
    return out.with_name(out.stem + "_metrics.json")


def train(args):
    """train the shared encoder with cross-modal InfoNCE (extracted glyph <-> clean sprite), select
    on held-out synth, export the best to onnx. cudnn disabled (its sublibs mismatch this cuda build);
    the net is tiny so the native cuda conv is fine.
    metrics (per-step loss/lr, per-eval synth-val top1) persist to <out>_metrics.json, flushed every
    eval so a killed run still leaves a readable file; `plot` turns it into curves."""
    import copy
    torch.backends.cudnn.enabled = False
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows, Tz = SG._matchable()
    n = len(rows)
    print(f"classes: {n}  device: {dev}  batch: {args.batch}  steps: {args.steps}")
    A_u8, Q_u8 = build_pool(rows, args.k, args.seed)
    V_u8 = val_synth(rows, args.seed + 10_000)
    nn_idx = _confusion_neighbors(Tz, k=args.fam_k) if args.p_family > 0 else None

    # run metadata + histories for the metrics file, one per --out, overwritten per run (rename it
    # to keep a run's curves around).
    t0 = time.time()
    metrics = {
        "meta": {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "classes": n, "device": dev,
            "args": {k: str(v) for k, v in vars(args).items() if k != "func"},
        },
        "steps": [],   # {step, loss, lr} every step, the raw curve
        "evals": [],   # {step, loss, scale, val_top1, best, t} every eval_every steps
    }
    mpath = _metrics_path(args.out)

    def _flush_metrics():
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(metrics))

    model = GlyphEncoder(p_drop=args.dropout).to(dev)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=dev))
    params = list(model.parameters()) + [logit_scale]
    aux = nn.Linear(EMB_DIM, n).to(dev) if args.aux_weight > 0 else None
    if aux is not None:
        params += list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    warmup = max(1, int(0.04 * args.steps))
    def lr_at(step):                              # linear warmup then cosine decay to ~1% of lr
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * t))

    best_top1, best_state, best_step = -1.0, None, -1
    bar = tqdm(range(args.steps), desc="train")
    mining = args.p_family > 0 and not args.no_mine
    for step in bar:
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(step)
        # once the embedding is worth mining, re-source the hard negatives from it; the fixed ncc
        # families only stand in for the warmup (a random-init embedding gives nonsense neighbours).
        if mining and step >= args.mine_after and step % args.mine_every == 0:
            nn_idx = _embedding_neighbors(model, A_u8, dev, args.fam_k, args.mine_skip)
        cls = _sample_classes(n, args.batch, nn_idx, args.p_family, rng)
        var = rng.integers(0, args.k, size=len(cls))
        q = _light_aug(_to_gpu(Q_u8[cls, var], dev), args.aug_bright, args.aug_noise)
        a = _light_aug(_to_gpu(A_u8[cls], dev), args.aug_bright * 0.5, args.aug_noise * 0.5)

        model.train()
        zq = F.normalize(model(q), dim=1)
        za = F.normalize(model(a), dim=1)
        scale = logit_scale.clamp(max=np.log(100.0)).exp()
        logits = scale * zq @ za.t()             # (N,N) query-vs-anchor similarities
        tgt = torch.arange(len(cls), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
        if aux is not None:                      # aux class head (discarded at export), sharpens near-dups
            cls_t = torch.from_numpy(np.asarray(cls, dtype=np.int64)).to(dev)   # ce targets must be long
            loss = loss + args.aux_weight * 0.5 * (
                F.cross_entropy(aux(zq), cls_t) + F.cross_entropy(aux(za), cls_t))

        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = loss.item()
        metrics["steps"].append({"step": step, "loss": round(lv, 4), "lr": args.lr * lr_at(step)})
        bar.set_postfix(loss=f"{lv:.3f}", best=f"{100 * max(best_top1, 0):.1f}%")

        if step % args.eval_every == 0 or step == args.steps - 1:
            top1 = _synth_top1(model, A_u8, V_u8, dev)
            tag = ""
            if top1 > best_top1:
                best_top1, best_state, best_step = top1, copy.deepcopy(model.state_dict()), step
                tag = " *best"
            metrics["evals"].append({
                "step": step, "loss": round(lv, 4), "scale": round(float(scale.detach()), 2),
                "val_top1": round(top1, 4), "best": tag != "", "t": round(time.time() - t0, 1),
            })
            _flush_metrics()
            # bar.write keeps the milestone log above the live bar instead of clobbering it
            bar.write(f"step {step:5d}  loss {lv:.3f}  scale {float(scale.detach()):5.1f}  "
                      f"synth-val top1 {100 * top1:.1f}%{tag}")
            bar.set_postfix(loss=f"{lv:.3f}", val=f"{100 * top1:.1f}%", best=f"{100 * best_top1:.1f}%")

    print(f"best synth-val top1 {100 * best_top1:.1f}% @ step {best_step}")
    metrics["meta"]["best_top1"] = round(best_top1, 4)
    metrics["meta"]["best_step"] = best_step
    metrics["meta"]["wall_s"] = round(time.time() - t0, 1)
    _flush_metrics()
    print(f"metrics -> {mpath} (plot with: python tools/glyph_cnn.py plot)")
    model.load_state_dict(best_state)
    torch.save(best_state, str(Path(args.out).with_suffix(".pt")))
    path = export_onnx(model, args.out)
    print(f"exported best -> {path}")
    return 0


def cmd_plot(args):
    """turn a run's metrics json into a two-panel png: loss curve (raw + smoothed) and synth-val top1
    with the selected-best step marked. saved next to the json, savefig only so it works over ssh /
    while a run is live (the file is flushed every eval)."""
    import matplotlib
    matplotlib.use("Agg")             # save-only, no gui backend needed
    import matplotlib.pyplot as plt

    mpath = _metrics_path(args.out)
    if not mpath.exists():
        print(f"no metrics at {mpath}, train first")
        return 1
    m = json.loads(mpath.read_text())
    steps = m["steps"]
    evals = m["evals"]
    if not steps:
        print(f"{mpath} has no step records")
        return 1

    xs = np.array([r["step"] for r in steps])
    ls = np.array([r["loss"] for r in steps])
    # ema smoothing so the trend reads through the per-batch noise, raw kept faint behind it
    alpha = 2.0 / (max(20, len(ls) // 50) + 1)
    ema = np.empty_like(ls)
    acc = ls[0]
    for i, v in enumerate(ls):
        acc = alpha * v + (1 - alpha) * acc
        ema[i] = acc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, ls, color="#4269d0", alpha=0.2, linewidth=0.8)
    ax1.plot(xs, ema, color="#4269d0", linewidth=2)
    ax1.set_title("infonce loss")
    ax1.set_xlabel("step")

    if evals:
        ex = [r["step"] for r in evals]
        ev = [100 * r["val_top1"] for r in evals]
        ax2.plot(ex, ev, color="#3ca951", linewidth=2, marker="o", markersize=4)
        bi = int(np.argmax([r["val_top1"] for r in evals]))
        ax2.annotate(
            f"best {ev[bi]:.1f}% @ {ex[bi]}", (ex[bi], ev[bi]),
            textcoords="offset points", xytext=(6, -12), fontsize=8, color="#555555",
        )
        ax2.plot([ex[bi]], [ev[bi]], marker="o", markersize=8,
                 markerfacecolor="none", markeredgecolor="#3ca951")
    ax2.set_title("synth-val top1 (%)")
    ax2.set_xlabel("step")

    started = m["meta"].get("started", "?")
    for ax in (ax1, ax2):                          # recessive grid, data carries the plot
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"glyph_cnn run {started}", fontsize=10)
    fig.tight_layout()
    ppath = mpath.with_suffix(".png")
    fig.savefig(str(ppath), dpi=150)
    print(f"plot -> {ppath}")
    return 0


def cmd_smoke(args):
    """the de-risk gate: random-weight encoder -> onnx -> cv2.dnn load + forward -> parity vs torch.
    proves opencv 4.6.0 can parse and run the graph before any training effort goes in."""
    torch.manual_seed(0)
    model = GlyphEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encoder params: {n_params:,}")
    model.eval()

    path = export_onnx(model, args.out)
    print(f"exported onnx -> {path}  ({path.stat().st_size/1024:.1f} KiB)")

    net = cv2.dnn.readNetFromONNX(str(path))       # the hard gate, throws if 4.6.0 can't parse it
    print("cv2.dnn.readNetFromONNX: ok")

    rng = np.random.default_rng(0)
    x = rng.random((INPUT_RES, INPUT_RES, 3), dtype=np.float32)   # hwc bgr [0,1], like to_input
    blob = x.transpose(2, 0, 1)[None]                             # -> (1,3,96,96) nchw, bgr kept
    net.setInput(blob)
    onnx_emb = net.forward().reshape(-1)                          # (128,) raw

    with torch.no_grad():
        torch_emb = model(torch.from_numpy(blob)).numpy().reshape(-1)

    max_abs = float(np.max(np.abs(onnx_emb - torch_emb)))
    # cosine of the l2-normed vectors is what matters (inference normalizes then matches)
    a = onnx_emb / (np.linalg.norm(onnx_emb) + 1e-9)
    b = torch_emb / (np.linalg.norm(torch_emb) + 1e-9)
    cos = float(a @ b)
    print(f"parity: max|onnx-torch|={max_abs:.2e}  cos(normed)={cos:.6f}")
    ok = max_abs < 1e-3 and cos > 0.9999
    print("SMOKE PASS" if ok else "SMOKE FAIL (investigate before training)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="learned glyph identity matcher (phase-2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_smoke = sub.add_parser("smoke", help="export a random net + verify cv2.dnn can load/forward it")
    s_smoke.add_argument("--out", type=Path, default=ONNX_PATH)
    s_smoke.set_defaults(func=cmd_smoke)

    s_train = sub.add_parser("train", help="train the encoder (InfoNCE) and export the best onnx")
    s_train.add_argument("--steps", type=int, default=5000)
    s_train.add_argument("--batch", type=int, default=256, help="distinct classes per batch (InfoNCE negatives)")
    s_train.add_argument("--k", type=int, default=24, help="synth query variants per class in the pool")
    s_train.add_argument("--lr", type=float, default=1e-3)
    s_train.add_argument("--wd", type=float, default=1e-4)
    s_train.add_argument("--dropout", type=float, default=0.1)
    s_train.add_argument("--p-family", dest="p_family", type=float, default=0.5,
                         help="fraction of batches drawn from confusion-neighbor families (hard negatives)")
    s_train.add_argument("--fam-k", dest="fam_k", type=int, default=8, help="neighbors per seed in a family batch")
    s_train.add_argument("--no-mine", dest="no_mine", action="store_true",
                         help="keep the fixed ncc families; don't re-mine negatives from the embedding")
    s_train.add_argument("--mine-after", dest="mine_after", type=int, default=1000,
                         help="step to start mining hard negatives from the embedding (warmup before)")
    s_train.add_argument("--mine-every", dest="mine_every", type=int, default=250,
                         help="steps between hard-negative refreshes")
    s_train.add_argument("--mine-skip", dest="mine_skip", type=int, default=1,
                         help="drop the N nearest per class (semi-hard; 0 = use the hardest)")
    s_train.add_argument("--aux-weight", dest="aux_weight", type=float, default=0.3,
                         help="weight of the aux class head (0 disables it); discarded at export")
    s_train.add_argument("--aug-bright", dest="aug_bright", type=float, default=0.15)
    s_train.add_argument("--aug-noise", dest="aug_noise", type=float, default=0.03)
    s_train.add_argument("--eval-every", dest="eval_every", type=int, default=250)
    s_train.add_argument("--seed", type=int, default=0)
    s_train.add_argument("--cpu", action="store_true", help="force cpu (debug only, slow)")
    s_train.add_argument("--out", type=Path, default=ONNX_PATH)
    s_train.set_defaults(func=train)

    s_plot = sub.add_parser("plot", help="plot a run's metrics json (loss + synth-val curves) to png")
    s_plot.add_argument("--out", type=Path, default=ONNX_PATH,
                        help="the model path the run wrote to; the metrics json sits next to it")
    s_plot.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    raise SystemExit(args.func(args))
