"""dev-only embedding diagnostics for the cnn matcher.

answers "why did X match Y": a wrong match is either anchor similarity (the two sprites genuinely
embed close, a ranking problem) or query drift (the real extracted glyph lands far from its own
anchor, an extraction problem). self-score vs anchor-sim(want,got) is the split: hits self-score
~0.89 median, drifted queries ~0.2-0.4.

subcommands (default = landscape+real+hygiene):
  landscape   anchor-anchor cosine over the matchable library, nn-sim distribution + closest pairs
  real        every real label re-embedded: top1, self-score, true-rank, classified miss census
  hygiene     rows detect.is_matchable rejects, grouped by reason
  pair A B    one suspect pair, anchor sim + mutual rank, e.g. `pair focusLens saboteur`
  plots       post-retrain dashboard png (training curves + gate landscape) + rescue sweep table

run: conda run -n dbdbp python -m tools.model_diagnose [subcommand]
"""

import sys
import time
import argparse
import contextlib
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import detect as D
from src.node import CNN_CONF_MIN

import tools.eval_matchers as EM

# anchor-sim above this = genuine near-dup (ranking problem); below it with a collapsed
# self-score = the query itself drifted (extraction problem)
NEARDUP_SIM = 0.60
DRIFT_SELF = 0.50


def load_bank():
    """(rows, net, B): matchable rows, the runtime onnx, and their l2-normed anchor bank.
    reuses eval_matchers so numbers here always agree with cnneval."""
    if not EM.CNN_ONNX.is_file():
        sys.exit("no model at data/models/glyph_encoder.onnx, train first (tools.glyph_cnn train)")
    lib = EM.load_matchable()
    net = cv2.dnn.readNetFromONNX(str(EM.CNN_ONNX))
    return lib["rows"], net, EM.cnn_bank(lib["rows"], net)


def rname(rows, i):
    r = rows[i]
    return f"{r['key']} [{r.get('category', '?')}/{r.get('rarity', '?')}]"


def cmd_landscape(args):
    rows, _net, B = load_bank()
    S = B @ B.T
    np.fill_diagonal(S, -2.0)
    nn = S.max(axis=1)
    print(f"\n=== anchor-anchor landscape (n={len(rows)}) ===")
    print(f"nearest-anchor sim: median {np.median(nn):.3f}  p90 {np.percentile(nn, 90):.3f}  "
          f"p99 {np.percentile(nn, 99):.3f}  max {nn.max():.3f}")
    for t in (0.9, 0.8, 0.7, CNN_CONF_MIN, 0.6):
        print(f"  anchors with a neighbor > {t:.2f}: {int((nn > t).sum())}")
    iu = np.triu_indices(len(rows), 1)
    order = np.argsort(-S[iu])[:args.top]
    print(f"\ntop {args.top} most-similar anchor pairs (the intrinsically confusable set):")
    for o in order:
        i, j = iu[0][o], iu[1][o]
        print(f"  {S[i, j]:.3f}  {rname(rows, i):45s} <-> {rname(rows, j)}")


def cmd_pair(args):
    rows, _net, B = load_bank()
    ki = {r["key"]: i for i, r in enumerate(rows)}
    for k in (args.a, args.b):
        if k not in ki:
            sys.exit(f"key not in the matchable library: {k}")
    S = B @ B.T
    np.fill_diagonal(S, -2.0)
    a, b = ki[args.a], ki[args.b]
    print(f"\nanchor sim {args.a} <-> {args.b} = {S[a, b]:.3f}")
    for i, k in ((a, args.a), (b, args.b)):
        rank = int((S[i] > S[i, a if i == b else b]).sum()) + 1
        print(f"  rank as seen from {k}: #{rank} of {len(rows) - 1}")
        for r, j in enumerate(np.argsort(-S[i])[:args.top], 1):
            print(f"    #{r:<3d} {S[i, j]:.3f}  {rname(rows, j)}")


def embed_real(rows, net, B, sources=None):
    """re-embed every usable real label through the production extraction (same path as
    eval_matchers.eval_real). yields (rec, scores, order, self_score, true_rank)."""
    ki = {r["key"]: i for i, r in enumerate(rows)}
    for rec in tqdm(EM.load_real_labels(sources=sources), desc="real queries"):
        want = rec["key"]
        crop = cv2.imread(str(ROOT / rec["crop_path"]))
        if want not in ki or crop is None:
            continue
        rarity = rec.get("rarity")
        r = int(rec.get("r") or round(min(crop.shape[:2]) / (2 * EM.BOX_K)))
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        iso = D.isolate_node_contents(crop, cx, cy, r, rarity)
        if iso is None:
            continue
        glyph = D.normalize_glyph(iso[4], iso[3], rarity)
        if glyph is None:
            continue
        s = B @ EM._cnn_embed(net, glyph)
        order = np.argsort(-s)
        yield rec, s, order, float(s[ki[want]]), int(np.where(order == ki[want])[0][0]) + 1


def cmd_real(args):
    rows, net, B = load_bank()
    ki = {r["key"]: i for i, r in enumerate(rows)}
    S = B @ B.T
    detail = set((args.detail or "").split(",")) - {""}
    sources = EM.SOURCES_INDEP if args.independent else None
    hits_self, miss_self, misses, scored, hits = [], [], [], 0, 0
    for rec, s, order, self_s, true_rank in embed_real(rows, net, B, sources=sources):
        want, top = rec["key"], order[0]
        scored += 1
        if rows[top]["key"] == want:
            hits += 1
            hits_self.append(self_s)
        else:
            miss_self.append(self_s)
            misses.append((rec, s, order, self_s, true_rank))
        if want in detail:
            print(f"\n  {rec['crop_path']} want={want}: true-rank #{true_rank} self={self_s:.3f}")
            for r, j in enumerate(order[:8], 1):
                print(f"    #{r:<2d} {s[j]:.3f}  {rname(rows, j)}"
                      + (" <== TRUE" if rows[j]["key"] == want else ""))

    print(f"\n=== real queries ({'independent' if args.independent else 'all sources'}) ===")
    print(f"scored {scored}  top1 {100 * hits / max(scored, 1):.1f}%")
    print(f"self-score (q . own anchor): hits median "
          f"{np.median(hits_self) if hits_self else float('nan'):.3f} | "
          f"misses median {np.median(miss_self) if miss_self else float('nan'):.3f}")

    print(f"\nmiss census ({len(misses)}), most confident first; >= {CNN_CONF_MIN} would be "
          f"TRUSTED live (wrong buy risk), below it routes to ocr (hover cost only):")
    for rec, s, order, self_s, true_rank in sorted(misses, key=lambda m: -m[1][m[2][0]]):
        want, got = rec["key"], rows[order[0]]["key"]
        win, pair = float(s[order[0]]), float(S[ki[want], ki[got]])
        family = ("near-dup" if pair >= NEARDUP_SIM
                  else "query-drift" if self_s < DRIFT_SELF else "ranking")
        gate = "ABOVE-GATE" if win >= CNN_CONF_MIN else "ocr"
        print(f"  {want:22s} -> {got:22s} win={win:.3f} self={self_s:.3f} rank#{true_rank:<5d} "
              f"anchor-sim={pair:.3f}  {family:11s} {gate}  rar={rec.get('rarity')}")


def cmd_hygiene(_args):
    rows, _ = D.load_index()
    dropped = [r for r in rows if not D.is_matchable(r)]
    power = [r for r in dropped if r.get("category") == "power"]
    retired = [r for r in dropped if r.get("category") != "power"
               and r.get("obtainable") == "unavailable" and r.get("rarity") is not None]
    rarityless = [r for r in dropped if r not in power and r not in retired]
    print(f"\n=== library hygiene ===")
    print(f"index {len(rows)} rows, {len(rows) - len(dropped)} matchable, {len(dropped)} dropped:")
    print(f"  {len(power)} powers, {len(retired)} wiki-retired/dead-art, {len(rarityless)} "
          f"rarity-less non-perk (raw-upload twins / retired art / in-match pickups)")
    for r in sorted(rarityless, key=lambda r: (r["category"], r["key"])):
        print(f"    {r['key']:36s} {str(r.get('name')):40s} {r['category']}")
    # desc-less matchable rows are article-less too, mostly dead perk art under renamed/relicensed
    # articles. not auto-pruned (perks have no real labels to verify against yet), just listed.
    suspects = [r for r in rows if D.is_matchable(r) and not r.get("desc")]
    print(f"\n  {len(suspects)} matchable rows with no desc (article-less, likely dead art; "
          f"kept, verify before pruning):")
    for r in sorted(suspects, key=lambda r: (r["category"], r["key"])):
        print(f"    {r['key']:36s} {str(r.get('name')):40s} {r['category']}/{r.get('rarity')}")


# dashboard colors, light-surface steps from the validated reference palette (dataviz skill):
# series blue/green for measures, status good/critical strictly for hit/miss, recessive chrome
_C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "blue": "#2a78d6", "green": "#008300",
    "hit": "#0ca30c", "miss": "#d03b3b",
}


def _style(ax, title):
    ax.set_title(title, fontsize=10, color=_C["ink"], loc="left")
    ax.grid(color=_C["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors=_C["ink2"])


def cmd_plots(args):
    """render the post-retrain dashboard png and print the rescue-gate sweep table.
    one look answers the recurring questions: did training converge, where do real hits/misses
    sit against the 0.65 trust gate, and what would the rescue knobs (node.CNN_RESCUE_*) buy
    (ocr hovers saved) or break (wrong buys) at each setting."""
    import json
    import matplotlib
    matplotlib.use("Agg")             # save-only, no gui backend needed
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from src.node import CNN_RESCUE_MIN, CNN_RESCUE_MARGIN, CNN_MARGIN_CAP

    rows, net, B = load_bank()
    recs = []                          # one dict per scored real label, all sources
    for rec, s, order, self_s, _rank in embed_real(rows, net, B):
        recs.append({
            "hit": rows[order[0]]["key"] == rec["key"], "want": rec["key"],
            "got": rows[order[0]]["key"], "win": float(s[order[0]]),
            "margin": float(s[order[0]] - s[order[1]]),
            "rarity": rec.get("rarity"), "source": rec.get("source"),
        })
    wins = np.array([r["win"] for r in recs])
    margins = np.array([r["margin"] for r in recs])
    hits = np.array([r["hit"] for r in recs])

    # the sweep: what each (min score, margin floor) rescue setting would promote off the ocr
    # path. only sub-gate nodes are in play (>= gate is trusted regardless), so cells read
    # saves+/wrongs- ; a setting is shippable only with 0 wrongs.
    sub = wins < CNN_CONF_MIN
    mins = (0.40, 0.45, 0.50, 0.55, 0.60)
    margs = (0.05, 0.10, 0.15, 0.20, 0.25)
    print(f"\n=== rescue-gate sweep (real labels n={len(recs)}, sub-gate n={int(sub.sum())}, "
          f"cells = hovers-saved+ / wrong-buys-) ===")
    print(f"current knobs: node.py {CNN_RESCUE_MIN}/{CNN_RESCUE_MARGIN}  trust gate {CNN_CONF_MIN}")
    print("min\\margin " + "".join(f"{m:>9.2f}" for m in margs))
    for mn in mins:
        cells = []
        for mg in margs:
            box = sub & (wins >= mn) & (margins >= mg) & (margins <= CNN_MARGIN_CAP)
            cells.append(f"{int((box & hits).sum()):>5d}+/{int((box & ~hits).sum())}-")
        print(f"{mn:>9.2f} " + "".join(cells))

    fig, ((ax_loss, ax_val, ax_rar), (ax_dist, ax_gate, ax_resc)) = plt.subplots(
        2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(_C["surface"])
    for ax in fig.axes:
        ax.set_facecolor(_C["surface"])

    # A+B training curves off the run's metrics json (written by glyph_cnn train)
    mpath = EM.CNN_ONNX.with_name(EM.CNN_ONNX.stem + "_metrics.json")
    if mpath.is_file():
        m = json.loads(mpath.read_text())
        xs = np.array([r["step"] for r in m["steps"]])
        ls = np.array([r["loss"] for r in m["steps"]])
        alpha = 2.0 / (max(20, len(ls) // 50) + 1)     # ema so the trend reads through the noise
        ema, acc = np.empty_like(ls), ls[0]
        for i, v in enumerate(ls):
            acc = alpha * v + (1 - alpha) * acc
            ema[i] = acc
        ax_loss.plot(xs, ls, color=_C["blue"], alpha=0.15, linewidth=0.6)
        ax_loss.plot(xs, ema, color=_C["blue"], linewidth=1.8)
        ax_loss.set_xlabel("step", fontsize=8, color=_C["ink2"])
        ev = m["evals"]
        ax_val.plot([r["step"] for r in ev], [100 * r["val_top1"] for r in ev],
                    color=_C["green"], linewidth=1.8)
        bi = int(np.argmax([r["val_top1"] for r in ev]))
        bx, by = ev[bi]["step"], 100 * ev[bi]["val_top1"]
        ax_val.plot([bx], [by], "o", markersize=7, markerfacecolor="none",
                    markeredgecolor=_C["green"])
        ax_val.annotate(f"best {by:.1f}% @ {bx}", (bx, by), textcoords="offset points",
                        xytext=(-6, -14), fontsize=8, color=_C["ink2"], ha="right")
        ax_val.set_xlabel("step", fontsize=8, color=_C["ink2"])
        ax_val.set_ylim(top=100)
    else:
        for ax in (ax_loss, ax_val):
            ax.text(0.5, 0.5, "no metrics json", ha="center", color=_C["muted"], fontsize=9)
    _style(ax_loss, "train infonce loss (raw + ema)")
    _style(ax_val, "synth-val top1 %")

    # C real top1 by rarity, independent labels only (the honest bar)
    indep = [r for r in recs if r["source"] in EM.SOURCES_INDEP]
    cats = [c for c in EM.RARITY_CYCLE + [None] if any(r["rarity"] == c for r in indep)]
    accs, ns = [], []
    for c in cats:
        rr = [r for r in indep if r["rarity"] == c]
        accs.append(100 * sum(r["hit"] for r in rr) / len(rr))
        ns.append(len(rr))
    xpos = np.arange(len(cats))
    ax_rar.bar(xpos, accs, width=0.62, color=_C["blue"])
    for x, a, n in zip(xpos, accs, ns):
        ax_rar.annotate(f"{a:.0f}%", (x, a), ha="center", xytext=(0, 3),
                        textcoords="offset points", fontsize=8, color=_C["ink"])
        ax_rar.annotate(f"n={n}", (x, 0), ha="center", xytext=(0, 4),
                        textcoords="offset points", fontsize=7, color="#ffffff")
    ax_rar.set_xticks(xpos, [str(c).replace(" ", "\n") for c in cats])
    ax_rar.set_ylim(0, 108)
    _style(ax_rar, f"real top1 by rarity (independent n={len(indep)})")

    # D where hits and misses sit vs the trust gate
    bins = np.linspace(min(0.0, wins.min()), 1.0, 26)
    ax_dist.hist(wins[hits], bins=bins, color=_C["hit"], alpha=0.6, label="hits")
    ax_dist.hist(wins[~hits], bins=bins, color=_C["miss"], alpha=0.75, label="misses")
    ax_dist.axvline(CNN_CONF_MIN, color=_C["ink2"], linewidth=1, linestyle="--")
    ax_dist.annotate(f"trust gate {CNN_CONF_MIN}", (CNN_CONF_MIN, ax_dist.get_ylim()[1]),
                     xytext=(-4, -2), textcoords="offset points", fontsize=8,
                     color=_C["ink2"], ha="right", va="top")
    ax_dist.legend(fontsize=8, frameon=False, labelcolor=_C["ink2"])
    ax_dist.set_xlabel("win score (cosine)", fontsize=8, color=_C["ink2"])
    _style(ax_dist, f"real win-score distribution (all sources n={len(recs)})")

    # E score-only gate sweep: precision of what's trusted vs how much skips ocr
    ts = np.linspace(0.30, 0.95, 66)
    cov = np.array([(wins >= t).mean() * 100 for t in ts])
    prec = np.array([100 * hits[wins >= t].mean() if (wins >= t).any() else np.nan for t in ts])
    ax_gate.plot(ts, prec, color=_C["green"], linewidth=1.8, label="precision of trusted")
    ax_gate.plot(ts, cov, color=_C["blue"], linewidth=1.8, label="coverage (skips ocr)")
    ax_gate.axvline(CNN_CONF_MIN, color=_C["ink2"], linewidth=1, linestyle="--")
    ax_gate.legend(fontsize=8, frameon=False, labelcolor=_C["ink2"], loc="lower left")
    ax_gate.set_xlabel("score threshold", fontsize=8, color=_C["ink2"])
    ax_gate.set_ylim(0, 103)
    _style(ax_gate, "trust-gate sweep %")

    # F the rescue plane: score x margin, current trusted + rescue regions shaded; any miss
    # inside a shaded region is a live wrong buy, annotated by name
    ax_resc.axvspan(CNN_CONF_MIN, 1.02, color=_C["hit"], alpha=0.06)
    ax_resc.add_patch(Rectangle(
        (CNN_RESCUE_MIN, CNN_RESCUE_MARGIN), CNN_CONF_MIN - CNN_RESCUE_MIN,
        CNN_MARGIN_CAP - CNN_RESCUE_MARGIN, color=_C["blue"], alpha=0.08, linewidth=0))
    ax_resc.scatter(wins[hits], margins[hits], s=12, color=_C["hit"], alpha=0.55,
                    linewidths=0, label="hits")
    ax_resc.scatter(wins[~hits], margins[~hits], s=34, color=_C["miss"], marker="x",
                    linewidths=1.4, label="misses")
    ymax = max(margins.max() + 0.06, 0.45)
    ax_resc.annotate("trusted", (1.0, ymax), fontsize=8, color=_C["muted"], ha="right", va="top",
                     xytext=(-2, -2), textcoords="offset points")
    ax_resc.annotate("rescue", (CNN_RESCUE_MIN, ymax), fontsize=8, color=_C["muted"], va="top",
                     xytext=(2, -2), textcoords="offset points")
    flip = 1
    for r in recs:                     # name the wrong buys, alternating offsets vs overlap
        rescued = (CNN_RESCUE_MIN <= r["win"] < CNN_CONF_MIN
                   and CNN_RESCUE_MARGIN <= r["margin"] <= CNN_MARGIN_CAP)
        if not r["hit"] and (r["win"] >= CNN_CONF_MIN or rescued):
            ax_resc.annotate(f"{r['want']}→{r['got']}", (r["win"], r["margin"]),
                             fontsize=7, color=_C["ink2"], ha="right",
                             xytext=(-5, flip * 7 - 3), textcoords="offset points")
            flip = 1 - flip
    ax_resc.legend(fontsize=8, frameon=False, labelcolor=_C["ink2"], loc="upper left")
    ax_resc.set_xlabel("win score", fontsize=8, color=_C["ink2"])
    ax_resc.set_ylabel("runner-up margin", fontsize=8, color=_C["ink2"])
    ax_resc.set_ylim(-0.02, ymax)
    _style(ax_resc, f"rescue plane (rescue {CNN_RESCUE_MIN}/{CNN_RESCUE_MARGIN})")

    fig.suptitle(
        f"glyph encoder dashboard  |  real top1 {100 * hits.mean():.1f}% (n={len(recs)}), "
        f"independent {100 * np.mean([r['hit'] for r in indep]):.1f}% (n={len(indep)})",
        fontsize=11, color=_C["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ppath = EM.CNN_ONNX.with_name("glyph_encoder_dashboard.png")
    fig.savefig(str(ppath), dpi=150, facecolor=_C["surface"])
    print(f"\ndashboard -> {ppath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("landscape", help="anchor-anchor similarity over the matchable library")
    p.add_argument("--top", type=int, default=30, help="how many top pairs to list")
    p = sub.add_parser("real", help="real-label query analysis + classified miss census")
    p.add_argument("--independent", action="store_true", help="only ocr/manual labels (honest bar)")
    p.add_argument("--detail", help="comma-separated keys to print full top-8 rankings for")
    p = sub.add_parser("pair", help="anchor sim + mutual neighbor rank for one suspect pair")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--top", type=int, default=10, help="neighbors to list per side")
    sub.add_parser("hygiene", help="rows is_matchable rejects, grouped by reason")
    sub.add_parser("plots", help="dashboard png (training + gate landscape) + rescue sweep table")
    args = ap.parse_args()
    # every report also lands next to the model so a later session can read it without re-running
    rpath = EM.CNN_ONNX.parent / "diagnose_report.txt"
    with open(rpath, "w", encoding="utf-8") as rf, \
            contextlib.redirect_stdout(EM._Tee(sys.stdout, rf)):
        print(time.strftime("model_diagnose %Y-%m-%d %H:%M") + f"  cmd={args.cmd or 'full'}")
        if args.cmd is None:                  # the full report
            cmd_hygiene(args)
            cmd_landscape(argparse.Namespace(top=30))
            cmd_real(argparse.Namespace(independent=False, detail=None))
        else:
            {"landscape": cmd_landscape, "real": cmd_real,
             "pair": cmd_pair, "hygiene": cmd_hygiene, "plots": cmd_plots}[args.cmd](args)
    print(f"\nreport -> {rpath}")


if __name__ == "__main__":
    main()
