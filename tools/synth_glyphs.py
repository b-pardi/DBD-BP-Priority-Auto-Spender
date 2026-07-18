"""dev-only synthetic extracted-glyph generator for the phase-2 learned matcher (not shipped).

the cnn trains on extracted glyphs, so this makes unlimited labeled ones by reusing the real query
path: render a node, degrade it, jitter the coarse crop, then run the exact isolate_node_contents +
normalize_glyph the live pipeline runs. keeps train-extraction == test-extraction and models the
off-center/mis-scaled crops the detector produces.

two glyphs per class: gallery is the clean bare sprite (the inference reference), query is a
heavily augmented extracted glyph the encoder must map near its gallery glyph.

torch-free on purpose, so the aug can be tuned against real crops before any training code exists.
run:
  vsreal  -> real crop | real extracted glyph | synth query | anchor, tune aug to match real
  thermo  -> ncc top1 on fresh synth, should sit in the real band (75.7-76.7% post-prune), much
             higher means aug is too weak and a cnn will overfit synth
  preview -> anchor vs synth-query pairs only
"""

import sys
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # tools/ so eval_matchers imports
from src import detect as D
import eval_matchers as EM        # reuse render_node + RARITY_CYCLE + BOX_K + load_real_labels

# encoder input side, matches NCC_RES so near-dup detail the cnn must split isn't lost
INPUT_RES = 96

# perks are null-rarity on tier-colored diamond sockets, never common/event, so cycle these tiers
# only. NOTE: pip appearance is unconfirmed from a real perk node, perk synth is unvalidated.
PERK_TIERS = ("uncommon", "rare", "very rare")


@dataclass
class AugCfg:
    # pre-extraction (fed to render_node), tuned so otsu output matches real extracted glyphs;
    # real has almost no color tint and almost no speckle, so these stay small
    r: int = 56                    # bigger radius closes the sharpness gap vs real (less baked-in blur)
    noise: float = 4.0              # real post-otsu glyphs are clean, heavy gaussian looked artificial
    blur: float = 0.0              # NOTE sigma < ~0.8 is a silent no-op (rounds to a 1x1 kernel)
    downscale: float = 1.0         # off, the game renders native-res; >1 was the main over-blur source
    disk_grad: float = 0.2        # radial shading so the event tier doesn't extract as a flat disk
    glyph_white_event: float = 0.45  # event only, lifts the gold glyph over the equally-bright gold disk
    event_speckle: float = 1.0     # event only, gold splatter + speckle so extraction leaks gold like
                                   # real event nodes (separates the banquet/masquerade reskins)
    tint: tuple = (0.2, 0.5)       # glyph tint toward disk color, the rarity cue the cnn needs on
                                   # near-dups; calibrated to real stroke saturation
    tint_event: tuple = (0.0, 0.15)  # event only, weaker: the leaked gold bg already tints the glyph
    # art faintness: in-game art draws semi-transparent over a bright plate, so at the low end otsu
    # keeps only stroke fragments -- the focusLens query-drift family the encoder must learn.
    # event keeps alpha high (its glyph_white lift is what survives the gold disk at all).
    art_alpha: tuple = (0.55, 1.0)
    art_alpha_event: tuple = (0.85, 1.0)
    plate_gain: tuple = (0.85, 1.15)  # plate brightness wobble, moves the otsu cut like live lighting
    # composite the UNCROPPED icon canvas at plate size like the game (per-icon padding survives, so
    # small-art icons come out small + pixelated like real nodes). the old bbox-crop-to-1.6r drew
    # every icon equally large and crisp -- the last visible gap vs real focusLens crops.
    art_crop: bool = False
    # art-vs-plate misregistration (max px / max frac): unlike crop_jitter below, this survives
    # isolate's recentering, since real art isn't always drawn dead-center on the plate.
    art_shift: float = 5.0
    art_zoom: float = 0.05
    jitter: int = 2
    texture: tuple = None          # (min,max) bright speck count, None = off (real glyphs have ~none)
    # real node layout: dark socket ring + rarity-colored textured plate (square/hexagon/diamond) +
    # add-on '+' marker, not a flat disk. real extraction leaks plate color into the glyph so queries
    # must train on it.
    plate: bool = True
    # menu-floor background patches (data/bw-bgs), since real crop corners sit at V~51-97, not
    # near-black. randomly cropped/zoomed/blurred/dimmed into the measured corner range.
    bg: bool = True
    bg_zoom: tuple = (1.0, 2.5)     # sample a (zoom*side)^2 region then shrink, for natural blur
    bg_blur: tuple = (0.8, 2.2)     # extra gaussian sigma, the caps are sharper than the live bg
    bg_dim: tuple = (50.0, 95.0)    # target mean gray, the measured real corner V range
    bg_contrast: tuple = (0.45, 0.8)  # local-contrast squeeze toward the patch mean, see _bg_patch
    # the two real node states: selectable = opaque + solid rim, otherwise a translucent fill with a
    # dim rim. alpha sits high (0.7-0.9) because PLATE_BGR was measured on mostly-translucent nodes,
    # so a lower alpha here would dim twice.
    selectable_p: float = 0.4
    node_alpha: tuple = (0.7, 0.9)
    # coarse-crop jitter fed to the real isolate step, so the query sees off-center/mis-scaled crops
    crop_jitter: bool = True
    center_jitter: float = 0.10     # max center offset as a fraction of r
    scale_jitter: float = 0.10      # max +/- radius error as a fraction of r
    # perk tier pips, bright marks that survive otsu and pollute a real perk glyph. off until the
    # real appearance is confirmed, then the cnn can learn to ignore them.
    pips: bool = False
    # post-extraction (on the 128 glyph), models residual extraction artifacts
    morph_p: float = 0.35           # random erode/dilate, stroke-thickness drift
    dropout_p: float = 0.08         # tiny random gaps, real otsu drops a stroke, never a big square
    reg_jitter: float = 2.5         # deg, residual bbox rotation
    reg_scale: float = 0.06
    reg_shift: float = 2.0

DEFAULT_AUG = AugCfg()


_BG_IMAGES = None   # lazily loaded menu-floor caps from data/bw-bgs (None = not scanned yet)


def _bg_images():
    """the user-curated floor screencaps under data/bw-bgs, loaded once. globbed not named, since the
    set grows as new game versions change the menu scene. [] when the dir is absent/empty."""
    global _BG_IMAGES
    if _BG_IMAGES is None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        _BG_IMAGES = [
            img for p in sorted((ROOT / "data" / "bw-bgs").glob("*"))
            if p.suffix.lower() in exts and (img := cv2.imread(str(p))) is not None
        ]
    return _BG_IMAGES


def _bg_patch(rng, side, cfg):
    """one processed floor patch: random image, random zoomed square crop, blur + dim into the
    measured real-corner band, so the synth node sits on the landscape the real web floats over.
    returns None when no caps are available (render falls back to the flat fill)."""
    imgs = _bg_images()
    if not imgs or not cfg.bg:
        return None
    img = imgs[int(rng.integers(len(imgs)))]
    h, w = img.shape[:2]
    s = int(side * rng.uniform(*cfg.bg_zoom))
    s = min(s, h, w)
    y0 = int(rng.integers(0, h - s + 1))
    x0 = int(rng.integers(0, w - s + 1))
    patch = cv2.resize(img[y0:y0 + s, x0:x0 + s], (side, side), interpolation=cv2.INTER_AREA)
    patch = cv2.GaussianBlur(patch, (0, 0), rng.uniform(*cfg.bg_blur))
    mean = max(1.0, float(patch.mean()))
    patch = patch.astype(np.float32) * (rng.uniform(*cfg.bg_dim) / mean)
    # squeeze local contrast toward the mean, the live floor reads flatter than a crisp screencap
    patch = patch.mean() + (patch - patch.mean()) * rng.uniform(*cfg.bg_contrast)
    return np.clip(patch, 0, 255).astype(np.uint8)


def rarity_for_row(row, i, rng):
    """rarity disk to render this row on. perks cycle their tier rarities (null-rarity, tier only
    picks the disk color); everything else cycles all six rarities."""
    if row.get("category") == "perk":
        return PERK_TIERS[i % len(PERK_TIERS)]
    return EM.RARITY_CYCLE[i % len(EM.RARITY_CYCLE)]


def _add_socket_texture(crop, rng, strength, radius):
    """sprinkle a few bright blobs inside the disk so otsu picks up realistic speckle, since
    plain gaussian noise is too uniform to reproduce the real texture leak."""
    w = crop.shape[1]
    cx = cy = w // 2                                       # render_node centers the disk at side//2
    out = crop.copy()
    for _ in range(int(rng.integers(strength[0], strength[1] + 1))):
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.9 * radius)
        px, py = int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang))
        rr = int(rng.uniform(1, max(2, radius * 0.08)))
        val = int(rng.uniform(120, 220))
        cv2.circle(out, (px, py), rr, (val, val, val), -1)
    return out


def _add_tier_pips(crop, rng, radius):
    """placeholder tier-pip marks (small bright dots) until the real appearance is confirmed;
    teaches the encoder to ignore bright marks that aren't part of the emblem."""
    w = crop.shape[1]
    cx = cy = w // 2
    out = crop.copy()
    n = int(rng.integers(1, 4))
    for k in range(n):
        px = int(cx + (k - (n - 1) / 2) * radius * 0.22)
        py = int(cy + radius * 0.72)
        cv2.circle(out, (px, py), max(1, int(radius * 0.06)), (230, 230, 230), -1)
    return out


def _aug_glyph(glyph, rng, cfg):
    """post-extraction aug on the 128 bgr glyph, modeling residual extraction artifacts."""
    g = glyph
    if rng.random() < cfg.morph_p:                        # stroke-thickness drift
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        g = cv2.erode(g, k) if rng.random() < 0.5 else cv2.dilate(g, k)
    if rng.random() < cfg.dropout_p:                      # otsu dropping dim regions
        g = g.copy()
        h, w = g.shape[:2]
        for _ in range(int(rng.integers(1, 3))):          # small stroke-sized gaps, not big squares
            pw, ph = int(rng.integers(w // 16, w // 8)), int(rng.integers(h // 16, h // 8))
            x, y = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
            g[y:y + ph, x:x + pw] = 0
    if cfg.reg_jitter:                                    # residual bbox jitter
        h, w = g.shape[:2]
        ang = rng.uniform(-cfg.reg_jitter, cfg.reg_jitter)
        s = rng.uniform(1 - cfg.reg_scale, 1 + cfg.reg_scale)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        M[0, 2] += rng.uniform(-cfg.reg_shift, cfg.reg_shift)
        M[1, 2] += rng.uniform(-cfg.reg_shift, cfg.reg_shift)
        g = cv2.warpAffine(g, M, (w, h))
    return g


def make_synth_glyph(file, rarity, rng, cfg=DEFAULT_AUG, category=None, return_crop=False):
    """one augmented extracted glyph (128 bgr) for `file` on `rarity`'s disk via the real query path
    (render -> texture/pips -> crop jitter -> isolate -> normalize_glyph -> post aug), or None if
    extraction produced nothing. return_crop=True also returns the rendered node crop, for vsreal."""
    gw = cfg.glyph_white_event if rarity == "event" else 0.0
    es = cfg.event_speckle if rarity == "event" else 0.0
    tr = cfg.tint_event if rarity == "event" else cfg.tint
    # plate shape by category (items/add-ons square, offerings hexagon, perks diamond)
    shape = {"offering": "hexagon", "perk": "rhombus"}.get(category, "square") if cfg.plate else None
    crop, contour = EM.render_node(
        file, rarity, r=cfg.r, noise=cfg.noise, blur=cfg.blur,
        tint=float(rng.uniform(*tr)), jitter=cfg.jitter, rng=rng,
        downscale=cfg.downscale, disk_grad=cfg.disk_grad, glyph_white=gw,
        event_speckle=es, plate_shape=shape, plus_marker=(category == "addon"),
        bg=_bg_patch(rng, int(2.6 * cfg.r), cfg) if shape else None,
        selectable=bool(rng.random() < cfg.selectable_p),
        node_alpha=float(rng.uniform(*cfg.node_alpha)),
        art_alpha=float(rng.uniform(*(cfg.art_alpha_event if rarity == "event" else cfg.art_alpha))),
        plate_gain=float(rng.uniform(*cfg.plate_gain)), art_crop=cfg.art_crop,
        art_shift=cfg.art_shift, art_zoom=cfg.art_zoom,
    )
    if cfg.texture:
        crop = _add_socket_texture(crop, rng, cfg.texture, cfg.r)
    if cfg.pips and category == "perk":
        crop = _add_tier_pips(crop, rng, cfg.r)

    render = crop
    if cfg.crop_jitter:
        # jitter center + radius, then let real isolate recenter/rescale, so the query sees
        # off-center/mis-scaled crops like production does
        side = crop.shape[0]
        cx = cy = side // 2
        jx = int(rng.uniform(-cfg.center_jitter, cfg.center_jitter) * cfg.r)
        jy = int(rng.uniform(-cfg.center_jitter, cfg.center_jitter) * cfg.r)
        rj = int(cfg.r * rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter))
        iso = D.isolate_node_contents(crop, cx + jx, cy + jy, rj, rarity)
        if iso is None:
            return (None, render) if return_crop else None
        _, _, _, contour, crop = iso                      # iso = (cx, cy, r, contour, crop)

    glyph = D.normalize_glyph(crop, contour, rarity)      # the exact production extraction
    if glyph is None:
        return (None, render) if return_crop else None
    glyph = _aug_glyph(glyph, rng, cfg)
    return (glyph, render) if return_crop else glyph


def gallery_glyph(file, color=True):
    """clean anchor for `file`, framed like the ncc templates. color=True keeps bgr (what the cnn
    uses, color=False is the gray version detect's ncc uses)."""
    p = Path(D.DEFAULT_INDEX).parent / file
    if not color:
        return D._sprite_glyph_gray(p)
    img = Image.open(p).convert("RGBA")
    bbox = img.getbbox()
    g = img.crop(bbox) if bbox else img
    gw, gh = g.size
    side = max(gw, gh)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    canvas.alpha_composite(g, ((side - gw) // 2, (side - gh) // 2))
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_input(img, res=INPUT_RES):
    """glyph -> (res,res,3) float32 bgr in [0,1], the encoder input. color is KEPT (the cnn uses it
    to split near-dups the grayscale ncc could not); a gray input is promoted to 3 channels."""
    x = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(x, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _matchable():
    """(rows, Tz) for matchable icons, built from the FULL index then masked so detect's shared ncc
    cache is not overwritten with a smaller matrix (same guard the harness uses)."""
    rows_full, _ = D.load_index()
    keep = np.array([D.is_matchable(r) for r in rows_full])
    T_full, T2_full = D.load_ncc_templates(rows_full)
    Tz = D.ncc_plain_templates((T_full, T2_full))[keep]
    return [r for r, k in zip(rows_full, keep) if k], Tz


def ncc_thermometer(n=300, seed=0, cfg=DEFAULT_AUG):
    """calibration read: ncc top1 on freshly rendered synth. tune AugCfg until this sits in the real
    band, 75.7% (honest, the 148 independent labels) to 76.7% (all 210), both re-measured 2026-07-17
    after normalize_glyph's peripheral prune lifted real ncc from the old 55.4-62.9 band;
    much higher means synth is too easy and a cnn trained on it will not transfer."""
    rng = np.random.default_rng(seed)
    rows, Tz = _matchable()
    sample = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    hits = tot = 0
    for c, i in enumerate(sample):
        row = rows[i]
        g = make_synth_glyph(row["file"], rarity_for_row(row, c, rng), rng, cfg, row.get("category"))
        if g is None:
            continue
        best, _, _, _ = D.id_icon_ncc(g, rows, Tz)
        hits += int(best["key"] == row["key"])
        tot += 1
    print(f"ncc thermometer: synth top1 {100 * hits / tot:.1f}% over {tot} nodes "
          f"(real band 75.7-76.7%). aim inside the band, well above it = aug too weak for a cnn.")


def preview(n=24, seed=0, cfg=DEFAULT_AUG):
    """gallery of clean anchor vs augmented synth-query pairs."""
    rng = np.random.default_rng(seed)
    rows, _ = _matchable()
    items = []
    for c, i in enumerate(rng.choice(len(rows), size=n, replace=False)):
        row = rows[i]
        g = make_synth_glyph(row["file"], rarity_for_row(row, c, rng), rng, cfg, row.get("category"))
        items.append((gallery_glyph(row["file"]), f"{row['key'][:12]} anchor"))
        items.append((g, f"{row.get('category', '?')[:5]} query"))
    D._show_gallery(items, title="synth-glyphs", cols=8, savefig=True)


def preview_vs_real(n=24, seed=0, cfg=DEFAULT_AUG):
    """the realism check the aug is tuned against: for real labeled nodes show
    real crop | real extracted glyph | synth query (same class) | clean anchor, side by side.
    if the synth query does not look like the real extracted glyph, the aug is off."""
    rng = np.random.default_rng(seed)
    rows, _ = _matchable()
    by_key = {r["key"]: r for r in rows}
    labels = [r for r in EM.load_real_labels() if r["key"] in by_key]
    labels = [labels[k] for k in rng.permutation(len(labels))]
    items = []
    for rec in labels[:n]:
        row = by_key[rec["key"]]
        rarity = rec.get("rarity")
        crop = cv2.imread(str(ROOT / rec["crop_path"]))
        real_glyph = None
        if crop is not None:
            r = int(rec.get("r") or round(min(crop.shape[:2]) / (2 * EM.BOX_K)))
            iso = D.isolate_node_contents(crop, crop.shape[1] // 2, crop.shape[0] // 2, r, rarity)
            if iso is not None:
                real_glyph = D.normalize_glyph(iso[4], iso[3], rarity)
        synth, synth_crop = make_synth_glyph(
            row["file"], rarity, rng, cfg, row.get("category"), return_crop=True)
        items += [
            (_zoom(crop), f"{rec['key'][:12]} real"), (real_glyph, "real glyph"),
            (_zoom(synth_crop), f"{(rarity or '?')[:5]} render"), (synth, "synth glyph"),
            (gallery_glyph(row["file"]), "anchor"),
        ]
    D._show_gallery(items, title="synth-vs-real", cols=10, savefig=True)


def _zoom(img, keep=0.86):
    """display-only center crop for the vsreal node tiles, so the node fills the tile but the rim
    stays visible. never used on the extraction path (isolate needs full context)."""
    if img is None:
        return None
    h, w = img.shape[:2]
    dy, dx = int(h * (1 - keep) / 2), int(w * (1 - keep) / 2)
    return img[dy:h - dy, dx:w - dx]


def _add_common(p, n_default):
    """shared subcommand args. --blur/--downscale override the two blurriness knobs from the cli so
    you can sweep them without editing AugCfg; everything else is tuned in the AugCfg defaults."""
    p.add_argument("-n", type=int, default=n_default)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blur", type=float, default=None, help="override AugCfg.blur (lower = sharper)")
    p.add_argument("--downscale", type=float, default=None,
                   help="override AugCfg.downscale (1 = off/sharpest, higher = blurrier)")
    p.add_argument("--art-alpha", dest="art_alpha", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="override AugCfg.art_alpha jitter range")
    p.add_argument("--plate-gain", dest="plate_gain", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="override AugCfg.plate_gain jitter range")
    p.add_argument("--art-crop", dest="art_crop", action="store_true",
                   help="legacy bbox-crop-to-1.6r art render (default = uncropped canvas at plate size)")
    p.add_argument("--art-shift", dest="art_shift", type=float, default=None,
                   help="override AugCfg.art_shift (max px art offset vs the plate)")
    p.add_argument("--art-zoom", dest="art_zoom", type=float, default=None,
                   help="override AugCfg.art_zoom (max frac art scale jitter vs the plate)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="synth extracted-glyph generator (phase-2 data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_common(sub.add_parser("thermo", help="ncc top1 on fresh synth, calibrate aug to the real bar"), 300)
    _add_common(sub.add_parser("preview", help="save an anchor-vs-query gallery to the debug dir"), 24)
    _add_common(sub.add_parser("vsreal", help="save a real-crop vs synth-query gallery (aug realism check)"), 24)
    args = ap.parse_args()

    over = {k: getattr(args, k) for k in ("blur", "downscale", "art_shift", "art_zoom")
            if getattr(args, k) is not None}
    for k in ("art_alpha", "plate_gain"):
        if getattr(args, k) is not None:
            over[k] = tuple(getattr(args, k))
    if args.art_crop:
        over["art_crop"] = True
    cfg = replace(DEFAULT_AUG, **over) if over else DEFAULT_AUG
    if args.cmd == "thermo":
        ncc_thermometer(n=args.n, seed=args.seed, cfg=cfg)
    elif args.cmd == "vsreal":
        preview_vs_real(n=args.n, seed=args.seed, cfg=cfg)
    else:
        preview(n=args.n, seed=args.seed, cfg=cfg)
