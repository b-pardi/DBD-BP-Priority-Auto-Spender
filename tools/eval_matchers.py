"""dev-only matcher evaluation harness (not part of the shipped pipeline).

compares icon matchers by running the REAL src/detect pipeline (normalize_glyph + the id_icon*
matchers) so the numbers reflect production, not a reimplementation. three signals:

1. synthetic-node eval (broad, fully labeled): render every sampled library glyph as a bloodweb
    node (rarity disk + downscale to the in-game source res + blur/noise/jitter), extract + match,
    measure top-1 over all rarities. noise is calibrated so synthetic cosines land in the same band
    as real extractions, which makes the relative ranking of the EXTRACTION matchers trustworthy.
2. real gold acceptance (narrow, hand-labeled ground truth): the 6 known gold/event nodes in the
    fixtures. thin, but the only fully trustworthy real-degradation signal we have today (the
    annotator being built in a separate session will grow this into a real labeled set).
3. calibration (not just top-1): production only trusts a match confident enough to skip the OCR
    tooltip hover, so we also report precision over the most-confident half of nodes. a well
    calibrated matcher that knows when it is right can beat a higher-top-1 matcher that does not.

matcher families:
  glyph matchers (ncc, ncc_masked, phash) match the EXTRACTED glyph (normalize_glyph output) vs the
    bare-sprite library. these are the current/shipped path.
  crop matchers (crop_ncc, crop_resid) are the experimental "B" direction: match the whole node
    crop (no glyph extraction) vs a bank of SYNTHETIC RENDERED nodes (each sprite composited onto
    the observed rarity disk). crop_resid additionally removes the shared per-pixel common mode (the
    disk) so discrimination is not washed out by the identical background.
    CAVEAT: the crop matchers' SYNTH numbers are circular (their templates and the synthetic queries
    both come from render_node) so treat synth for crop_* as a plumbing smoke test only, and judge
    the crop matchers on the gold/real-labeled set.

pool note: socket shape is no longer used to prune the candidate set (it is an agreement check that
triggers OCR fallback in the spender), so every matcher here searches the full matchable library.
matchable = the index minus obtainable=='unavailable' (killer powers, retired content) which never
appear in the bloodweb.

run (needs the conda env):
  conda run -n dbdbp-env python tools/eval_matchers.py compare      # all matchers, synth + gold
  conda run -n dbdbp-env python tools/eval_matchers.py synth -n 300 --matcher crop_resid
  conda run -n dbdbp-env python tools/eval_matchers.py gold --matcher crop_resid
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import detect as D
from src import paths

# the fixture crop the gold coords were read against (auto-bbox is deferred, so this stays fixed).
WEB_BBOX = {'x0': 300, 'y0': 200, 'xf': 1500, 'yf': 1300}

# 6 hand-labeled gold/event nodes: {fixture: {(x, y) in the cropped frame: expected key}}.
GOLD = {
    "web-005122.png": {(373, 327): "banquetMedKit", (757, 632): "banquetToolbox",
                       (289, 638): "banquetFlashlight", (470, 639): "10thAnniversary"},
    "web-005135.png": {(610, 396): "banquetFlashlight", (471, 474): "10thAnniversary"},
}

# rarity disk colors (bgr) derived from detect's empirical hsv anchors, for the synthetic render.
DISK_BGR = {
    rar: cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    for rar, hsv in D.EMPIRICAL_SEED.items()
}
RARITY_CYCLE = ["common", "uncommon", "rare", "very rare", "ultra rare", "event"]

NCC_RES = D.NCC_RES         # match the glyph matchers' vector resolution
BOX_K = 1.3                 # crop/render half-width in node radii (render_node uses side=2.6r)

MATCHER_NAMES = ("ncc", "ncc_masked", "phash", "crop_ncc", "crop_resid")


# ----------------------------------------------------------------------------- matchable library

def load_matchable():
    """load the index and drop obtainable=='unavailable' rows (killer powers, retired content) that
    never appear in the bloodweb, so no matcher can ever propose them. returns a dict with every
    template representation aligned to the SAME filtered row order.

    the ncc/phash matrices are built from the FULL index first (so detect's shared on-disk cache
    stays full-sized and uncorrupted) then masked down, rather than rebuilding a filtered cache."""
    rows_full, hashes_full = D.load_index()
    keep = np.array([r.get("obtainable") != "unavailable" for r in rows_full])
    rows = [r for r, k in zip(rows_full, keep) if k]
    T_full, T2_full = D.load_ncc_templates(rows_full)          # (n, res*res) built/cached from full
    Tz_full = D.ncc_plain_templates((T_full, T2_full))
    print(f"matchable library: {len(rows)}/{len(rows_full)} rows "
          f"({int((~keep).sum())} unavailable dropped)")
    return {
        "rows": rows,
        "hashes": hashes_full[keep],
        "ncc_T": (T_full[keep], T2_full[keep]),
        "Tz": Tz_full[keep],
    }


# --------------------------------------------------------------------------- synthetic node render

def render_node(file, rarity, r=40, noise=25, blur=1.2, tint=0.4, jitter=4, rng=None, degrade=True):
    """compose one synthetic bloodweb node and return (crop_bgr, contour). the contour is the disk
    circle, fed to normalize_glyph so the extraction step runs exactly as in production.

    steps: colored rarity disk, whitish line-art glyph composited via its alpha (slightly tinted
    toward the disk to mimic the in-game low-contrast fill), then optional degradation.
    degrade=True (the default, for synthetic QUERIES) adds downscale to the in-game source res,
    gaussian blur, noise, and a small affine jitter, calibrated to real-extraction cosines.
    degrade=False (for the crop-matcher TEMPLATE bank) skips all of that and returns the clean
    composite, so a template is the ideal node and the query is the degraded one."""
    rng = rng or np.random.default_rng(0)
    side = int(2.6 * r)
    cx = cy = side // 2
    crop = np.full((side, side, 3), 28, np.uint8)              # dark web-ish background
    disk = np.array(DISK_BGR[rarity], np.float32)
    cv2.circle(crop, (cx, cy), r, [int(c) for c in disk], -1)

    g = Image.open(ROOT / "data" / file).convert("RGBA")
    bb = g.getbbox()
    g = g.crop(bb) if bb else g
    gs = int(1.6 * r)
    g = g.resize((gs, gs), Image.LANCZOS)
    ga = np.array(g).astype(np.float32)
    grgb = (1 - tint) * ga[..., :3][..., ::-1] + tint * disk   # rgb->bgr, tinted toward disk
    alpha = ga[..., 3:] / 255.0
    y0, x0 = cy - gs // 2, cx - gs // 2
    roi = crop[y0:y0 + gs, x0:x0 + gs].astype(np.float32)
    crop[y0:y0 + gs, x0:x0 + gs] = (alpha * grgb + (1 - alpha) * roi).astype(np.uint8)

    if degrade:
        small = cv2.resize(crop, (side // 2, side // 2), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(small, (side, side), interpolation=cv2.INTER_LINEAR)
        if blur:
            crop = cv2.GaussianBlur(crop, (0, 0), blur)
        crop = np.clip(crop.astype(np.float32) + rng.normal(0, noise, crop.shape),
                       0, 255).astype(np.uint8)
        if jitter:
            ang = rng.uniform(-4, 4)
            tx, ty = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            crop = cv2.warpAffine(crop, M, (side, side), borderValue=(28, 28, 28))

    th = np.linspace(0, 2 * np.pi, 40)
    pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1).astype(np.int32)
    return crop, pts.reshape(-1, 1, 2)


# ------------------------------------------------------------------------------- glyph matchers

def glyph_matchers(lib):
    """the extraction matchers, each wrapped to a uniform glyph->(row, score, margin). no socket
    pool (shape is agreement-only now) so every match searches the full matchable library."""
    rows = lib["rows"]
    hashes, (T, T2), Tz = lib["hashes"], lib["ncc_T"], lib["Tz"]
    return {
        "ncc":        lambda g: D.id_icon_ncc(g, rows, Tz)[:3],
        "ncc_masked": lambda g: D.id_icon_ncc_masked(g, rows, (T, T2))[:3],
        "phash":      lambda g: D.id_icon_hamming(g, rows, hashes)[:3],
    }


# -------------------------------------------------------------------------- crop matchers ("B")

_BANK = {}      # {rarity: (n, res*res) z-normed clean-render vectors}
_RESID = {}     # {rarity: (Br, mu)} common-mode-removed bank + the removed mean


def _crop_to_vec(crop_bgr, res=NCC_RES):
    """whole node crop -> z-normed (mean-removed, unit-norm) grayscale vector at res*res.
    the crop is assumed already boxed to ~BOX_K*r half-width so scale is normalized by the known
    node radius before it gets here (same box the templates are rendered into)."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    v = cv2.resize(g, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def _clean_render_vec(file, rarity, res=NCC_RES):
    """one clean (undegraded) rendered-node template vector. a missing/broken sprite becomes a zero
    vector so a single bad file does not abort a whole bank build."""
    try:
        crop, _ = render_node(file, rarity, degrade=False)
        return _crop_to_vec(crop, res)
    except Exception:
        return np.zeros(res * res, np.float32)


def render_bank(rows, rarity, res=NCC_RES):
    """z-normed clean-render template matrix (n, res*res) for every matchable row composited on
    `rarity`'s disk. lazy and cached in-memory + on disk, keyed by rarity/res/row-count and guarded
    by the index mtime. building a bank renders ~1.5k nodes so the disk cache matters."""
    if rarity in _BANK:
        return _BANK[rarity]
    idx_mtime = Path(D.DEFAULT_INDEX).stat().st_mtime
    cache = paths.cache_dir() / f"renderbank-{rarity.replace(' ', '_')}-{res}-{len(rows)}.npy"
    if cache.is_file() and cache.stat().st_mtime >= idx_mtime:
        B = np.load(cache)
    else:
        print(f"  building render bank for '{rarity}' ({len(rows)} icons)...")
        B = np.stack([_clean_render_vec(r["file"], rarity, res) for r in rows]).astype(np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, B)
        except OSError:
            pass
    _BANK[rarity] = B
    return B


def resid_bank(rows, rarity, res=NCC_RES):
    """the common-mode-removed bank: subtract the per-pixel mean over the rarity's templates (which
    is dominated by the shared disk) from each template, then renorm. returns (Br, mu). removing the
    shared background analytically is the crop_resid experiment (avoids the disk washing out the
    small glyph region, the failure mode that sank the frozen-cnn embedding)."""
    if rarity in _RESID:
        return _RESID[rarity]
    B = render_bank(rows, rarity, res)
    mu = B.mean(axis=0)
    Br = B - mu
    Br /= (np.linalg.norm(Br, axis=1, keepdims=True) + 1e-6)
    _RESID[rarity] = (Br, mu)
    return Br, mu


def crop_matchers(lib):
    """the experimental whole-crop matchers, each wrapped to (crop, rarity)->(row, score, margin).
    the crop must already be boxed to ~BOX_K*r half-width; rarity picks the render bank's disk."""
    rows = lib["rows"]

    def crop_ncc(crop, rarity):
        B = render_bank(rows, rarity)
        q = _crop_to_vec(crop)
        s = B @ q                                      # (n,) cosine, both sides unit z-normed
        o = np.argsort(-s)
        return rows[o[0]], float(s[o[0]]), float(s[o[0]] - s[o[1]])

    def crop_resid(crop, rarity):
        Br, mu = resid_bank(rows, rarity)
        q = _crop_to_vec(crop) - mu
        q /= (np.linalg.norm(q) + 1e-6)
        s = Br @ q
        o = np.argsort(-s)
        return rows[o[0]], float(s[o[0]]), float(s[o[0]] - s[o[1]])

    return {"crop_ncc": crop_ncc, "crop_resid": crop_resid}


def build_registry(lib):
    """{name: (kind, fn)} for every matcher, where kind is 'glyph' (fn takes the extracted glyph)
    or 'crop' (fn takes the boxed node crop + observed rarity). lets the eval loops stay generic."""
    reg = {name: ("glyph", fn) for name, fn in glyph_matchers(lib).items()}
    reg.update({name: ("crop", fn) for name, fn in crop_matchers(lib).items()})
    return reg


# ------------------------------------------------------------------------------------- helpers

def crop_box(frame, cx, cy, r, box_k=BOX_K):
    """box crop centered on a node at box_k*r half-width, clamped to the frame, for the crop
    matchers so the query is framed like a rendered template (scale normalized by the known r)."""
    half = int(round(box_k * r))
    h, w = frame.shape[:2]
    return frame[max(cy - half, 0):min(h, cy + half), max(cx - half, 0):min(w, cx + half)]


def summarize(records, higher, frac=0.5):
    """records = [(correct_bool, score), ...]; higher = True if a bigger score is a better match.
    returns (top1_pct, confident_precision_pct, n). confident precision = accuracy over the most
    confident `frac` of nodes, a threshold-free calibration read comparable across matchers."""
    if not records:
        return 0.0, 0.0, 0
    correct = np.array([c for c, _ in records], dtype=bool)
    score = np.array([s for _, s in records], dtype=np.float32)
    order = np.argsort(-score if higher else score)          # most confident first
    k = max(1, int(frac * len(records)))
    return 100.0 * correct.mean(), 100.0 * correct[order[:k]].mean(), len(records)


# ------------------------------------------------------------------------------------- evals

def eval_synth(lib, name, fn, kind, n=300, seed=0, noise=25):
    """render n synthetic nodes (one per sampled glyph, rarities cycled), extract or box per the
    matcher kind, match, and return (records, higher). for crop matchers this is circular (template
    and query share render_node) so it is a smoke test only, see the module docstring."""
    rng = np.random.default_rng(seed)
    rows = lib["rows"]
    keys = [r["key"] for r in rows]
    sample = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    higher = name != "phash"
    records = []
    for c, i in enumerate(sample):
        rarity = RARITY_CYCLE[c % 6]
        crop, contour = render_node(rows[i]["file"], rarity, noise=noise, rng=rng)
        if kind == "glyph":
            glyph = D.normalize_glyph(crop, contour)
            if glyph is None:
                continue
            row, score, _ = fn(glyph)
        else:
            row, score, _ = fn(crop, rarity)             # render_node already boxes to 2.6r
        records.append((row["key"] == keys[i], score))
    return records, higher


def eval_gold(lib, name, fn, kind, verbose=False):
    """run the 6 real gold nodes through the real localize step and match per the matcher kind
    (glyph: normalize_glyph output; crop: box crop of the raw frame). returns (n_correct, total).
    also reports the socket-shape agreement (matched category vs the read socket shape), the signal
    that would trigger an OCR fallback in production."""
    rows = lib["rows"]
    correct = total = agree = 0
    for fx, exp in GOLD.items():
        frame = cv2.imread(str(ROOT / "tests" / "fixtures" / fx))
        frame = frame[WEB_BBOX['y0']:WEB_BBOX['yf'], WEB_BBOX['x0']:WEB_BBOX['xf']]
        nodes = D.find_nodes_in_frame(frame)
        for (x, y), want in exp.items():
            total += 1
            nd = min(nodes, key=lambda n: (n[0] - x) ** 2 + (n[1] - y) ** 2)
            iso = D.isolate_node_contents(frame, nd[0], nd[1], nd[2], nd[3])
            if iso is None:
                continue
            cx, cy, r_ref, contour, crop = iso
            if kind == "glyph":
                glyph = D.normalize_glyph(crop, contour)
                if glyph is None:
                    continue
                row, score, margin = fn(glyph)
            else:
                row, score, margin = fn(crop_box(frame, cx, cy, r_ref), nd[3])
            hit = row["key"] == want
            shp = D.classify_socket(contour)
            shape_ok = row.get("category") in D.NODE_SHAPE_DICT.get(shp, [])
            correct += hit
            agree += shape_ok
            if verbose:
                print(f"   {fx[:7]} ({x},{y}) want={want:17s} -> {row['key']:18s} "
                      f"score={score:.3f} m={margin:.3f} shape={shp}"
                      f"{'' if shape_ok else '!='} {'OK' if hit else 'xx'}")
    return correct, total, agree


# ------------------------------------------------------------------------------------- commands

def cmd_compare(args):
    lib = load_matchable()
    reg = build_registry(lib)
    print(f"\nmatcher comparison  (synth N={args.n} seed={args.seed} noise={args.noise}, no pool "
          f"| gold 6 real)\n")
    print(f"{'matcher':12} {'kind':6} {'synth top1':>11} {'synth conf50':>13} {'gold':>6}")
    for name, (kind, fn) in reg.items():
        recs, higher = eval_synth(lib, name, fn, kind, n=args.n, seed=args.seed, noise=args.noise)
        top1, conf, _ = summarize(recs, higher)
        gok, gtot, _ = eval_gold(lib, name, fn, kind)
        star = "*" if kind == "crop" else " "
        print(f"{name:12} {kind:6} {top1:10.1f}% {conf:12.1f}% {gok:4}/{gtot}{star}")
    print("\n* crop_* synth is circular (templates and queries both from render_node); it is a "
          "plumbing smoke test only.\n  judge crop_* on the gold column (and the real labeled set "
          "once the annotator lands).")


def cmd_synth(args):
    lib = load_matchable()
    kind, fn = build_registry(lib)[args.matcher]
    recs, higher = eval_synth(lib, args.matcher, fn, kind, n=args.n, seed=args.seed, noise=args.noise)
    top1, conf, scored = summarize(recs, higher)
    tag = "  (CIRCULAR smoke test)" if kind == "crop" else ""
    print(f"{args.matcher}: synth top1={top1:.1f}%  conf50={conf:.1f}%  "
          f"(N={scored}, noise={args.noise}, seed={args.seed}){tag}")


def cmd_gold(args):
    lib = load_matchable()
    kind, fn = build_registry(lib)[args.matcher]
    print(f"{args.matcher} on the 6 real gold nodes:")
    gok, gtot, agree = eval_gold(lib, args.matcher, fn, kind, verbose=True)
    print(f"   => {gok}/{gtot} correct, {agree}/{gtot} socket-shape agree")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="dev matcher eval (synthetic-node + real gold)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_cmp = sub.add_parser("compare", help="all matchers on synth + gold side by side")
    s_cmp.add_argument("-n", type=int, default=300, help="synthetic nodes to render")
    s_cmp.add_argument("--seed", type=int, default=0)
    s_cmp.add_argument("--noise", type=float, default=25.0)
    s_cmp.set_defaults(func=cmd_compare)

    s_syn = sub.add_parser("synth", help="one matcher on the synthetic-node eval")
    s_syn.add_argument("--matcher", choices=MATCHER_NAMES, default="ncc")
    s_syn.add_argument("-n", type=int, default=300)
    s_syn.add_argument("--seed", type=int, default=0)
    s_syn.add_argument("--noise", type=float, default=25.0)
    s_syn.set_defaults(func=cmd_synth)

    s_gld = sub.add_parser("gold", help="one matcher on the 6 real gold nodes")
    s_gld.add_argument("--matcher", choices=MATCHER_NAMES, default="ncc")
    s_gld.set_defaults(func=cmd_gold)

    args = ap.parse_args()
    args.func(args)
