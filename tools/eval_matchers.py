"""dev-only matcher evaluation harness (not part of the shipped pipeline).

two complementary signals for comparing icon matchers, both running the REAL src/detect pipeline
(normalize_glyph + the id_icon* matchers) so the numbers reflect production, not a reimplementation:

1. synthetic-node eval (broad, fully labeled): render every sampled library glyph as a bloodweb
    node (rarity disk + downscale to the in-game source res + blur/noise/jitter), run it through
    normalize_glyph, match, and measure top-1 over all rarities. the noise is calibrated so the
    synthetic cosines land in the same band as real extractions (cos ~0.2-0.35 for the masked
    matcher), which makes the relative ranking of matchers trustworthy even though it is a proxy.
2. real gold acceptance (narrow, hand-labeled ground truth): the 6 known gold/event nodes in the
    fixtures, the only fully trustworthy real-degradation signal we have.

these started as scratchpad probes; this is their stable home. the headline result they produced:
plain 'ncc' beats 'ncc_masked' and 'phash' on the broad eval and on confidence, while 'ncc_masked'
only wins the narrow gold subset it was tuned on, so 'ncc' is now the detect default.

run (needs the conda env):
  conda run -n dbdbp-env python tools/eval_matchers.py compare      # all matchers, synth + gold
  conda run -n dbdbp-env python tools/eval_matchers.py synth -n 300 --matcher ncc
  conda run -n dbdbp-env python tools/eval_matchers.py gold --matcher ncc
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


def build_matchers(rows):
    """prepare every matcher's templates once and return {name: id_fn(glyph_bgr, pool)} so the eval
    loops stay matcher-agnostic. mirrors detect()'s own setup (load_ncc_templates etc.)."""
    _, hashes = D.load_index()
    ncc_T = D.load_ncc_templates(rows)
    ncc_plain_T = D.ncc_plain_templates(ncc_T)
    return {
        "ncc":        lambda g, pool: D.id_icon_ncc_plain(g, rows, ncc_plain_T, pool=pool),
        "ncc_masked": lambda g, pool: D.id_icon_ncc(g, rows, ncc_T, pool=pool),
        "phash":      lambda g, pool: D.id_icon(g, rows, hashes, pool=pool),
    }


def render_node(file, rarity, r=40, noise=25, blur=1.2, tint=0.4, jitter=4, rng=None):
    """compose one synthetic bloodweb node and return (crop_bgr, contour). the contour is the disk
    circle, fed to normalize_glyph so the extraction step runs exactly as in production. steps:
    colored rarity disk, whitish line-art glyph composited via its alpha (slightly tinted toward the
    disk to mimic the in-game low-contrast fill), downscale->up to the in-game source res, gaussian
    blur + noise, small affine jitter. noise defaults are calibrated to real-extraction cosines."""
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

    small = cv2.resize(crop, (side // 2, side // 2), interpolation=cv2.INTER_AREA)
    crop = cv2.resize(small, (side, side), interpolation=cv2.INTER_LINEAR)
    if blur:
        crop = cv2.GaussianBlur(crop, (0, 0), blur)
    crop = np.clip(crop.astype(np.float32) + rng.normal(0, noise, crop.shape), 0, 255).astype(np.uint8)
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


def eval_synth(rows, id_fn, n=300, seed=0, noise=25, use_pool=False):
    """render n synthetic nodes (one per sampled glyph, rarities cycled), extract + match, return
    (top1, cos_mean, n_scored). pooling is off by default: a circular contour misreads socket shape,
    so the broad eval searches the full library (a stricter, shape-agnostic number)."""
    rng = np.random.default_rng(seed)
    cats = np.array([r['category'] for r in rows])
    keys = [r['key'] for r in rows]
    sample = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    ok = scored = 0
    coss = []
    for c, i in enumerate(sample):
        crop, contour = render_node(rows[i]["file"], RARITY_CYCLE[c % 6], noise=noise, rng=rng)
        glyph = D.normalize_glyph(crop, contour)
        if glyph is None:
            continue
        pool = (np.isin(cats, D.NODE_SHAPE_DICT[D.classify_socket(contour)]) if use_pool else None)
        row, cos, _ = id_fn(glyph, pool)
        ok += row['key'] == keys[i]
        scored += 1
        coss.append(cos)
    return 100.0 * ok / scored, float(np.mean(coss)), scored


def eval_gold(rows, id_fn, verbose=False):
    """run the 6 real gold nodes through the real localize+extract pipeline (socket-pooled, as in
    production) and return (n_correct, total). prints per-node detail when verbose."""
    cats = np.array([r['category'] for r in rows])
    correct = total = 0
    for fx, exp in GOLD.items():
        frame = cv2.imread(str(ROOT / "tests" / "fixtures" / fx))
        frame = frame[WEB_BBOX['y0']:WEB_BBOX['yf'], WEB_BBOX['x0']:WEB_BBOX['xf']]
        nodes = D.find_nodes_in_frame(frame)
        for (x, y), want in exp.items():
            n = min(nodes, key=lambda nd: (nd[0] - x) ** 2 + (nd[1] - y) ** 2)
            iso = D.isolate_node_contents(frame, n[0], n[1], n[2], n[3])
            if iso is None:
                total += 1
                continue
            _, _, _, contour, crop = iso
            glyph = D.normalize_glyph(crop, contour)
            pool = np.isin(cats, D.NODE_SHAPE_DICT[D.classify_socket(contour)])
            row, cos, margin = id_fn(glyph, pool)
            hit = row['key'] == want
            correct += hit
            total += 1
            if verbose:
                print(f"   {fx[:7]} ({x},{y}) want={want:17s} -> {row['key']:18s} "
                      f"score={cos:.3f} m={margin:.3f} {'OK' if hit else 'xx'}")
    return correct, total


def cmd_compare(args):
    rows, _ = D.load_index()
    matchers = build_matchers(rows)
    print(f"matcher comparison  (synth N={args.n} seed={args.seed} noise={args.noise}, no pool | gold 6 real, pooled)\n")
    # score is a cosine for ncc/ncc_masked (higher better) but a hamming distance for phash
    # (lower better), so the score column is only comparable within a matcher, not across.
    print(f"{'matcher':12} {'synth top1':>11} {'synth scoreμ':>13} {'gold':>6}")
    for name, id_fn in matchers.items():
        top1, scorem, _ = eval_synth(rows, id_fn, n=args.n, seed=args.seed, noise=args.noise)
        gok, gtot = eval_gold(rows, id_fn)
        print(f"{name:12} {top1:10.1f}% {scorem:13.3f} {gok:4}/{gtot}")


def cmd_synth(args):
    rows, _ = D.load_index()
    id_fn = build_matchers(rows)[args.matcher]
    top1, cosm, scored = eval_synth(rows, id_fn, n=args.n, seed=args.seed, noise=args.noise)
    print(f"{args.matcher}: synth top1={top1:.1f}%  cosμ={cosm:.3f}  (N={scored}, noise={args.noise}, seed={args.seed})")


def cmd_gold(args):
    rows, _ = D.load_index()
    id_fn = build_matchers(rows)[args.matcher]
    print(f"{args.matcher} on the 6 real gold nodes:")
    gok, gtot = eval_gold(rows, id_fn, verbose=True)
    print(f"   => {gok}/{gtot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="dev matcher eval (synthetic-node + real gold)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_cmp = sub.add_parser("compare", help="all matchers on synth + gold side by side")
    s_cmp.add_argument("-n", type=int, default=300, help="synthetic nodes to render")
    s_cmp.add_argument("--seed", type=int, default=0)
    s_cmp.add_argument("--noise", type=float, default=25.0)
    s_cmp.set_defaults(func=cmd_compare)

    s_syn = sub.add_parser("synth", help="one matcher on the synthetic-node eval")
    s_syn.add_argument("--matcher", choices=D.MATCHERS, default="ncc")
    s_syn.add_argument("-n", type=int, default=300)
    s_syn.add_argument("--seed", type=int, default=0)
    s_syn.add_argument("--noise", type=float, default=25.0)
    s_syn.set_defaults(func=cmd_synth)

    s_gld = sub.add_parser("gold", help="one matcher on the 6 real gold nodes")
    s_gld.add_argument("--matcher", choices=D.MATCHERS, default="ncc")
    s_gld.set_defaults(func=cmd_gold)

    args = ap.parse_args()
    args.func(args)
