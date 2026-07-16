"""dev-only matcher evaluation harness (not part of the shipped pipeline).

compares icon matchers by running the REAL src/detect pipeline (normalize_glyph + the id_icon*
matchers) so the numbers reflect production, not a reimplementation. three signals:

1. synthetic-node eval (broad, fully labeled): render every sampled glyph as a bloodweb node,
    extract + match, measure top-1. noise is calibrated to real-extraction cosines so the relative
    ranking of the EXTRACTION matchers is trustworthy.
2. real gold acceptance (narrow, hand-labeled): the 6 known gold/event fixture nodes. thin, but the
    only fully trustworthy real-degradation signal today (the annotator will grow this).
3. calibration: production only trusts a match confident enough to skip the OCR hover, so we also
    report precision over the most-confident half. good calibration can beat a higher top-1.

matcher families:
  glyph matchers (ncc, ncc_masked, phash) match the EXTRACTED glyph vs the bare-sprite library, the
    shipped path.
  crop matchers (crop_ncc, crop_resid) are the experimental "B" direction: match the whole node crop
    (no extraction) vs a bank of SYNTHETIC RENDERED nodes. crop_resid also removes the shared
    per-pixel common mode (the disk) so it does not wash out discrimination.
    CAVEAT: crop_* SYNTH numbers are circular (templates and queries both from render_node), so treat
    them as a plumbing smoke test and judge crop_* on the gold/real set.

pool note: socket shape no longer prunes candidates (it is an agreement check that triggers OCR
fallback in the spender), so every matcher searches the full matchable library.
matchable = the index minus obtainable=='unavailable' (killer powers, retired content), which never
appear in the bloodweb.

run (needs the conda env):
  conda run -n dbdbp-env python tools/eval_matchers.py compare      # all matchers, synth + gold
  conda run -n dbdbp-env python tools/eval_matchers.py synth -n 300 --matcher crop_resid
  conda run -n dbdbp-env python tools/eval_matchers.py gold --matcher crop_resid
"""

import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import detect as D
from src import paths
from src.resolution import Resolution

# the fixture crop the gold coords were read against, single-sourced from Resolution (the baseline
# fixtures are 3440x1440 so this stays {x0:300, y0:200, xf:1500, yf:1300}).
WEB_BBOX = Resolution().web_bbox_fallback_px()

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

# real nodes do not paint the whole disk: the glyph sits on a rarity-colored textured PLATE (square
# for items/add-ons, hexagon for offerings, diamond for perks) inside a dark neutral socket ring.
# per-rarity (top, bottom) plate gradient bgr from medians on the 07-04 live scans, brighter toward
# the top like the tile art. event keeps its gold-splatter colors (that render already matched real).
PLATE_BGR = {
    # bottoms kept ~0.85x the measured median (not darker): on dark tiers a few dn of channel
    # separation is all that keeps hue inside isolate's +-8 band, common bottom widened for the same
    # reason (probe 2026-07-05)
    "common":     ((40, 52, 70), (25, 32, 44)),
    "uncommon":   ((32, 77, 30), (20, 48, 19)),
    "rare":       ((84, 61, 45), (53, 38, 28)),
    "very rare":  ((84, 42, 70), (53, 26, 44)),
    "ultra rare": ((51, 24, 111), (32, 15, 70)),
    "event":      ((55, 190, 235), (25, 100, 185)),
}
SOCKET_BGR = (44, 42, 46)      # dark neutral disk fill around the plate (measured ring color)
RIM_BGR = (84, 105, 111)       # beige rim of an immediately-selectable node (mean over bright-rim
                               # real crops, probe 2026-07-05); non-selectable nodes show only a dim
                               # thin outline over a translucent fill


def _plate_poly(shape, cx, cy, half):
    """plate outline points for cv2.fillPoly, sized to fill the socket like the real tile art.
    square for items/add-ons, pointy-top hexagon for offerings, diamond for perks."""
    if shape == "hexagon":
        th = np.pi / 2 + np.arange(6) * np.pi / 3
        pts = np.stack([cx + 1.1 * half * np.cos(th), cy - 1.1 * half * np.sin(th)], 1)
    elif shape == "rhombus":
        pts = np.array([[cx, cy - 1.15 * half], [cx + 1.15 * half, cy],
                        [cx, cy + 1.15 * half], [cx - 1.15 * half, cy]])
    else:                                                   # square (items / add-ons)
        pts = np.array([[cx - half, cy - half], [cx + half, cy - half],
                        [cx + half, cy + half], [cx - half, cy + half]])
    return pts.astype(np.int32).reshape(1, -1, 2)
RARITY_CYCLE = ["common", "uncommon", "rare", "very rare", "ultra rare", "event"]

NCC_RES = D.NCC_RES         # match the glyph matchers' vector resolution
BOX_K = 1.3                 # crop/render half-width in node radii (render_node uses side=2.6r)

MATCHER_NAMES = ("ncc", "ncc_masked", "phash", "crop_ncc", "crop_resid", "cnn")


# ----------------------------------------------------------------------------- matchable library

def load_matchable():
    """load the index and drop obtainable=='unavailable' rows (killer powers, retired content) so no
    matcher can propose them. returns a dict with every template rep aligned to the SAME filtered
    row order.

    the ncc/phash matrices are built from the FULL index then masked down, not rebuilt filtered, so
    detect's shared on-disk cache stays full-sized and uncorrupted."""
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

def render_node(file, rarity, r=40, noise=25, blur=1.2, tint=0.4, jitter=4, rng=None, degrade=True, downscale=2.0, disk_grad=0.0, glyph_white=0.0, event_speckle=0.0, plate_shape=None, plus_marker=False, bg=None, selectable=True, node_alpha=0.62):
    """compose one synthetic bloodweb node and return (crop_bgr, contour). the contour is the disk
    circle, fed to normalize_glyph so extraction runs exactly as in production.

    steps: colored rarity disk, whitish line-art glyph composited via its alpha (tinted toward the
    disk to mimic the low-contrast fill), then optional degradation. degrade=True (default, for
    synthetic QUERIES) adds downscale, blur, noise, and affine jitter calibrated to real cosines.
    degrade=False (the crop-matcher TEMPLATE bank) returns the clean ideal composite.

    plate_shape ('square' | 'hexagon' | 'rhombus') switches to the REAL layout: a dark socket ring
    with a rarity-colored textured plate under the glyph, not a full rarity-colored disk. real
    extraction leaks plate color/texture into the glyph (focus-lens/luckless-mouse miss, probe
    2026-07-05) so training queries must see it; None keeps the legacy flat-disk render.
    plus_marker adds the bright add-on '+' at the plate top-right, which survives otsu and pollutes
    the extracted glyph.
    bg is an optional side-sized menu-floor patch used as the base layer (real crop corners sit at
    V~51-97, not near-black).
    selectable picks the two real node states (plate mode only): True = opaque node + solid beige
    rim; False = fill blended over the floor at node_alpha (~0.62) with only a dim rim outline, a
    node that cannot be bought yet."""
    rng = rng or np.random.default_rng(0)
    side = int(2.6 * r)
    cx = cy = side // 2
    if bg is not None:
        crop = cv2.resize(bg, (side, side), interpolation=cv2.INTER_AREA) if bg.shape[0] != side else bg.copy()
    else:
        crop = np.full((side, side, 3), 28, np.uint8)          # dark web-ish background
    base = crop.copy()                                          # pre-node floor, for the translucent state
    disk = np.array(SOCKET_BGR if plate_shape else DISK_BGR[rarity], np.float32)
    if disk_grad > 0:
        # radial shading + faint texture so the disk is not flat; a flat fill lets the event-tier
        # CLAHE in normalize_glyph boost the whole disk and swallow the white glyph.
        yy, xx = np.ogrid[:side, :side]
        rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1)          # 0 center .. 1 rim
        shade = (1.0 - disk_grad * np.clip(rad, 0, 1)).astype(np.float32)   # dim toward the rim
        disk_img = disk[None, None, :] * shade[..., None]
        disk_img += rng.normal(0, disk_grad * 18, disk_img.shape)           # faint disk texture
        inside = rad <= 1.0
        crop[inside] = np.clip(disk_img[inside], 0, 255).astype(np.uint8)
    else:
        cv2.circle(crop, (cx, cy), r, [int(c) for c in disk], -1)

    if plate_shape:
        # rarity-colored plate: vertical gradient + grain + dark grunge speckle, clipped to the disk.
        # event keeps its denser gold speckle, other tiers get light grunge so the otsu leak has the
        # texture real plates have.
        half = int(0.75 * r)
        top_c, bot_c = (np.array(c, np.float32) for c in PLATE_BGR[rarity])
        # glyph's rarity cast comes from the plate, not the dark neutral socket, so tint pulls toward
        # the plate mid color
        disk = (top_c + bot_c) / 2
        yy, xx = np.ogrid[:side, :side]
        in_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        pmask = np.zeros((side, side), np.uint8)
        cv2.fillPoly(pmask, _plate_poly(plate_shape, cx, cy, half), 255)
        m = in_disk & (pmask > 0)
        tg = np.clip((np.arange(side) - (cy - half)) / max(1, 2 * half), 0, 1)  # 0 top .. 1 bottom
        grad = top_c[None, :] * (1 - tg[:, None]) + bot_c[None, :] * tg[:, None]
        patch = np.repeat(grad[:, None, :], side, axis=1)
        # grain must be LUMINANCE-correlated (scale all 3 channels together), not per-channel iid:
        # iid noise on these dark fills scrambles hue out of isolate's +-8 band, the mask fragments,
        # and isolate latches a glyph chunk -> sliced extractions (probe 2026-07-05: real common
        # plates 41% in-band vs 11% synth with iid grain)
        patch *= (1 + rng.normal(0, 0.12, patch.shape[:2]))[..., None]
        patch += rng.normal(0, 2, patch.shape)                 # faint residual color grain
        crop[m] = np.clip(patch[m], 0, 255).astype(np.uint8)
        density = 3.0 * event_speckle if rarity == "event" else 0.5
        for _ in range(int(density * r)):                      # dark grunge/speckle holes
            ang, rad_f = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 1))
            px = int(cx + rad_f * half * np.cos(ang))
            py = int(cy + rad_f * half * np.sin(ang))
            if 0 <= py < side and 0 <= px < side and m[py, px]:
                cv2.circle(crop, (px, py), int(rng.integers(1, max(2, r // 16))), (26, 24, 28), -1)
        if plus_marker:
            # bright add-on '+' straddling the plate top-right; real ones survive otsu and leak a
            # corner notch into the extracted glyph
            mx, my = cx + half, cy - half
            arm, th_px = max(2, int(0.18 * r)), max(2, int(0.07 * r))
            c = tuple(int(v) for v in rng.integers(170, 215, 1).repeat(3))
            cv2.rectangle(crop, (mx - arm, my - th_px), (mx + arm, my + th_px), c, -1)
            cv2.rectangle(crop, (mx - th_px, my - arm), (mx + th_px, my + arm), c, -1)
        if selectable:
            # immediately-buyable node: stays opaque, draw the solid beige rim with a darker arc so
            # it is not a perfect uniform annulus (no real rim is)
            rim_c = np.clip(np.array(RIM_BGR) + rng.normal(0, 8, 3), 0, 255).astype(int)
            cv2.circle(crop, (cx, cy), int(1.06 * r), tuple(int(v) for v in rim_c),
                       max(2, int(0.11 * r)))
            a0 = rng.uniform(0, 360)
            cv2.ellipse(crop, (cx, cy), (int(1.06 * r), int(1.06 * r)), 0, a0,
                        a0 + rng.uniform(60, 160), tuple(int(v * 0.7) for v in rim_c),
                        max(1, int(0.05 * r)))
        else:
            # not-yet-selectable: the node fill sits translucently over the menu floor and only a
            # dim thin rim outline shows
            yy2, xx2 = np.ogrid[:side, :side]
            in_disk2 = (xx2 - cx) ** 2 + (yy2 - cy) ** 2 <= int(1.12 * r) ** 2
            blend = node_alpha * crop.astype(np.float32) + (1 - node_alpha) * base.astype(np.float32)
            crop[in_disk2] = np.clip(blend[in_disk2], 0, 255).astype(np.uint8)
            cv2.circle(crop, (cx, cy), int(1.06 * r),
                       tuple(int(v * 0.62) for v in RIM_BGR), max(1, r // 18))
    elif event_speckle > 0:
        # real event sockets are not a flat gold disk: a gold splatter SQUARE sits under the glyph,
        # bright yellow fading to orange, riddled with dark speckle holes. its V spans ~27..255
        # continuously so otsu can't split fill from glyph and the real glyph keeps a gold bg (the
        # leak IS the signal separating the banquet/masquerade/anniversary reskins). without this
        # the synth event glyph extracts clean-on-black, queries that never occur live (probe 07-03).
        half = int(0.75 * r)                                    # splatter square ~1.5r wide
        yy, xx = np.ogrid[:side, :side]
        in_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        in_sq = (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
        m = in_disk & in_sq
        tg = np.clip((np.arange(side) - (cy - half)) / max(1, 2 * half), 0, 1)   # 0 top .. 1 bottom
        top_c = np.array([55, 190, 235], np.float32)            # bright gold-yellow
        bot_c = np.array([25, 100, 185], np.float32)            # darker orange
        grad = top_c[None, :] * (1 - tg[:, None]) + bot_c[None, :] * tg[:, None] # (side,3) per row
        patch = np.repeat(grad[:, None, :], side, axis=1)
        patch += rng.normal(0, 12, patch.shape)                 # gold grain
        crop[m] = np.clip(patch[m], 0, 255).astype(np.uint8)
        for _ in range(int(event_speckle * 3 * r)):             # dark speckle holes, denser look
            ang, rad_f = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 1))
            px = int(cx + rad_f * half * np.cos(ang))
            py = int(cy + rad_f * half * np.sin(ang))
            if 0 <= py < side and 0 <= px < side and m[py, px]:
                cv2.circle(crop, (px, py), int(rng.integers(1, max(2, r // 16))), (26, 24, 28), -1)

    g = Image.open(ROOT / "data" / file).convert("RGBA")
    bb = g.getbbox()
    g = g.crop(bb) if bb else g
    gs = int(1.6 * r)
    g = g.resize((gs, gs), Image.LANCZOS)
    ga = np.array(g).astype(np.float32)
    grgb = (1 - tint) * ga[..., :3][..., ::-1] + tint * disk   # rgb->bgr, tinted toward disk
    if glyph_white:
        # lift toward white so otsu keeps the glyph not the disk; needed for event where a gold
        # glyph on a gold disk is otherwise the same brightness (gold-blob fix)
        grgb = (1 - glyph_white) * grgb + glyph_white * 255.0
    alpha = ga[..., 3:] / 255.0
    y0, x0 = cy - gs // 2, cx - gs // 2
    roi = crop[y0:y0 + gs, x0:x0 + gs].astype(np.float32)
    crop[y0:y0 + gs, x0:x0 + gs] = (alpha * grgb + (1 - alpha) * roi).astype(np.uint8)

    if degrade:
        if downscale and downscale != 1:
            ds = max(1, int(round(side / downscale)))                       # model the in-game low res
            small = cv2.resize(crop, (ds, ds), interpolation=cv2.INTER_AREA)
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
            # with a floor patch behind the node a flat gray warp border would be a fake edge
            crop = cv2.warpAffine(
                crop, M, (side, side),
                borderMode=cv2.BORDER_REPLICATE if bg is not None else cv2.BORDER_CONSTANT,
                borderValue=(28, 28, 28))

    th = np.linspace(0, 2 * np.pi, 40)
    pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1).astype(np.int32)
    return crop, pts.reshape(-1, 1, 2)


# ------------------------------------------------------------------------------- glyph matchers

def glyph_matchers(lib):
    """the extraction matchers, each wrapped to a uniform glyph->(row, score, margin). no socket pool
    (shape is agreement-only now) so every match searches the full matchable library."""
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
    assumed already boxed to ~BOX_K*r half-width, the same box the templates render into, so scale
    is normalized by the known radius before it gets here."""
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
    """z-normed clean-render template matrix (n, res*res) for every matchable row on `rarity`'s disk.
    lazy, cached in-memory + on disk keyed by rarity/res/row-count and guarded by the index mtime;
    a bank renders ~1.5k nodes so the disk cache matters."""
    if rarity in _BANK:
        return _BANK[rarity]
    idx_mtime = Path(D.DEFAULT_INDEX).stat().st_mtime
    cache = paths.template_cache_dir() / f"renderbank-{rarity.replace(' ', '_')}-{res}-{len(rows)}.npy"
    if cache.is_file() and cache.stat().st_mtime >= idx_mtime:
        B = np.load(cache)
    else:
        B = np.stack([_clean_render_vec(r["file"], rarity, res)
                      for r in tqdm(rows, desc=f"render bank {rarity}")]).astype(np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, B)
        except OSError:
            pass
    _BANK[rarity] = B
    return B


def resid_bank(rows, rarity, res=NCC_RES):
    """the common-mode-removed bank: subtract the per-pixel mean over the rarity's templates (which
    is dominated by the shared disk) from each, then renorm. returns (Br, mu). removing the shared bg
    analytically is the crop_resid experiment, so the disk does not wash out the small glyph region
    (the failure mode that sank the frozen-cnn embedding)."""
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
    crop must already be boxed to ~BOX_K*r half-width; rarity picks the render bank's disk."""
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


# ------------------------------------------------------------------------- cnn matcher (phase 2)

CNN_ONNX = ROOT / "data" / "models" / "glyph_encoder.onnx"
CNN_RES = 96                # encoder input side, matches synth_glyphs.INPUT_RES + tools/glyph_cnn


def _cnn_anchor(file):
    """clean color sprite framed like the ncc templates, the bank reference for one library row.
    MUST match synth_glyphs.gallery_glyph(color=True) (the training anchor) so bank embeddings live
    in the same space the encoder maps extracted glyphs into."""
    p = Path(D.DEFAULT_INDEX).parent / file
    img = Image.open(p).convert("RGBA")
    bbox = img.getbbox()
    g = img.crop(bbox) if bbox else img
    gw, gh = g.size
    side = max(gw, gh)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    canvas.alpha_composite(g, ((side - gw) // 2, (side - gh) // 2))
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cnn_blob(bgr, res=CNN_RES):
    """glyph/anchor bgr -> (1,3,res,res) float32 nchw in [0,1], the cv2.dnn input. same INTER_AREA
    resize + /255 as synth_glyphs.to_input; bgr order kept on purpose (color splits near-dups the
    grayscale ncc could not)."""
    x = cv2.resize(bgr, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def _cnn_embed(net, bgr):
    """one glyph -> l2-normed 128 embedding via cv2.dnn (the deployable runtime, no torch). l2-norm
    is done here in numpy, not in the onnx graph, so the export stays basic-ops for the old opencv
    importer."""
    net.setInput(_cnn_blob(bgr))
    e = net.forward().reshape(-1).astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def cnn_bank(rows, net):
    """(n,128) l2-normed anchor embeddings aligned to rows, cached like the ncc templates. keyed by
    CONTENT like detect's bank (rows key+phash fingerprint + onnx bytes, see detect._library_fingerprint),
    and 'eval-' prefixed so it can never collide with the runtime bank in the shared cache dir."""
    fp = f"{D._library_fingerprint(rows)}-{D._onnx_fingerprint()}"
    cache = paths.template_cache_dir() / f"embed-eval-{CNN_ONNX.stem}-{CNN_RES}-{fp}.npy"
    if cache.is_file():
        B = np.load(cache)
        if B.shape[0] == len(rows):
            return B
    B = np.stack([_cnn_embed(net, _cnn_anchor(r["file"]))
                  for r in tqdm(rows, desc="cnn bank")]).astype(np.float32)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, B)
    except OSError:
        pass
    return B


def cnn_matcher(lib):
    """the learned matcher wrapped to glyph->(row, score, margin), or {} when no model is trained
    yet (so compare/registry still work pre-training). score = cosine over the embedding bank."""
    if not CNN_ONNX.is_file():
        return {}
    rows = lib["rows"]
    net = cv2.dnn.readNetFromONNX(str(CNN_ONNX))
    B = cnn_bank(rows, net)

    def cnn(glyph):
        q = _cnn_embed(net, glyph)
        s = B @ q                                      # (n,) cosine, both l2-normed
        o = np.argsort(-s)
        return rows[o[0]], float(s[o[0]]), float(s[o[0]] - s[o[1]])

    return {"cnn": cnn}


def build_registry(lib):
    """{name: (kind, fn)} for every matcher; kind is 'glyph' (fn takes the extracted glyph) or 'crop'
    (fn takes the boxed node crop + observed rarity), so the eval loops stay generic. the cnn matcher
    is glyph-kind and is skipped if untrained."""
    reg = {name: ("glyph", fn) for name, fn in glyph_matchers(lib).items()}
    reg.update({name: ("crop", fn) for name, fn in crop_matchers(lib).items()})
    reg.update({name: ("glyph", fn) for name, fn in cnn_matcher(lib).items()})
    return reg


# ------------------------------------------------------------------------------------- helpers

def crop_box(frame, cx, cy, r, box_k=BOX_K):
    """box crop centered on a node at box_k*r half-width, clamped to the frame, so the crop-matcher
    query is framed like a rendered template (scale normalized by the known r)."""
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

def eval_synth(lib, name, fn, kind, n=300, seed=0, noise=25, progress=False):
    """render n synthetic nodes (one per sampled glyph, rarities cycled), extract or box per the
    matcher kind, match, return (records, higher). for crop matchers this is circular (template and
    query share render_node) so it is a smoke test only, see the module docstring.
    progress shows a tqdm bar (off inside compare's loop to avoid stacked bars)."""
    rng = np.random.default_rng(seed)
    rows = lib["rows"]
    keys = [r["key"] for r in rows]
    sample = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    higher = name != "phash"
    records = []
    for c, i in enumerate(tqdm(sample, desc=f"synth {name}", disable=not progress)):
        rarity = RARITY_CYCLE[c % 6]
        crop, contour = render_node(rows[i]["file"], rarity, noise=noise, rng=rng)
        if kind == "glyph":
            glyph = D.normalize_glyph(crop, contour, rarity)
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
    also reports socket-shape agreement (matched category vs read shape), the signal that would
    trigger an OCR fallback in production."""
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
                glyph = D.normalize_glyph(crop, contour, nd[3])
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


SOURCES_INDEP = ("ocr", "manual")   # labels independent of any matcher (ocr tooltip or hand-typed);
                                    # 'matcher'-sourced labels are the old ncc guess a human only
                                    # eyeballed, so scoring ncc on them is partly circular


def load_real_labels(path=None, sources=None):
    """the annotator's real labeled nodes (data/labels/real_nodes.json), resolved keys only.
    sources filters by label origin: None keeps all, SOURCES_INDEP keeps only the honest
    (matcher-independent) ocr/manual labels. returns [] if the file is absent."""
    path = path or (ROOT / "data" / "labels" / "real_nodes.json")
    if not Path(path).is_file():
        return []
    recs = [r for r in json.loads(Path(path).read_text("utf-8")) if r.get("key")]
    if sources is not None:
        recs = [r for r in recs if r.get("source") in sources]
    return recs


def eval_real(lib, name, fn, kind, verbose=False, sources=None, progress=False):
    """score a matcher on the annotator's real labeled crops. crop matchers get the saved 1.3*r box
    directly (the annotator's BOX_K matches ours); glyph matchers re-extract via
    isolate_node_contents + normalize_glyph off the crop center. a label whose key is not in the
    matchable library (unavailable, or a stale/alias key) is skipped, not scored.
    returns (records, higher, correct, total, skipped); records = [(correct_bool, score)].
    progress shows a tqdm bar (off inside compare's loop to avoid stacked bars)."""
    rows = lib["rows"]
    keyset = {r["key"] for r in rows}
    higher = name != "phash"
    records = []
    correct = total = skipped = 0
    for rec in tqdm(load_real_labels(sources=sources), desc=f"real {name}", disable=not progress):
        want = rec["key"]
        crop = cv2.imread(str(ROOT / rec["crop_path"]))
        if want not in keyset or crop is None:
            skipped += 1
            continue
        rarity = rec.get("rarity")
        if kind == "glyph":
            r = int(rec.get("r") or round(min(crop.shape[:2]) / (2 * BOX_K)))
            cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
            iso = D.isolate_node_contents(crop, cx, cy, r, rarity)
            if iso is None:
                skipped += 1
                continue
            glyph = D.normalize_glyph(iso[4], iso[3], rarity)   # iso = (cx, cy, r, contour, crop)
            if glyph is None:
                skipped += 1
                continue
            row, score, _ = fn(glyph)
        else:
            if rarity not in DISK_BGR:                        # crop matcher needs a known disk
                skipped += 1
                continue
            row, score, _ = fn(crop, rarity)
        hit = row["key"] == want
        correct += hit
        total += 1
        records.append((hit, score))
        if verbose and not hit:
            stem = rec["crop_path"].replace("\\", "/").split("/")[-1]
            print(f"   {stem:22s} want={want:18s} -> {row['key']:18s} score={score:6.3f} rar={rarity}")
    return records, higher, correct, total, skipped


# ------------------------------------------------------------------------------------- commands

def cmd_compare(args):
    lib = load_matchable()
    reg = build_registry(lib)
    n_all = len(load_real_labels())
    n_ind = len(load_real_labels(sources=SOURCES_INDEP))
    print(f"\nmatcher comparison  (synth N={args.n} seed={args.seed} noise={args.noise} | "
          f"real all {n_all} / independent {n_ind} | gold 6, no pool)\n")
    print(f"{'matcher':12} {'kind':6} {'synth':>7} {'all t1':>8} {'all c50':>8} "
          f"{'ind t1':>8} {'ind c50':>8} {'gold':>6}")
    for name, (kind, fn) in reg.items():
        srecs, shigh = eval_synth(lib, name, fn, kind, n=args.n, seed=args.seed, noise=args.noise)
        stop1, _, _ = summarize(srecs, shigh)
        rrecs, rhigh, _, _, _ = eval_real(lib, name, fn, kind)
        rtop1, rconf, _ = summarize(rrecs, rhigh)
        irecs, ihigh, _, _, _ = eval_real(lib, name, fn, kind, sources=SOURCES_INDEP)
        itop1, iconf, _ = summarize(irecs, ihigh)
        gok, gtot, _ = eval_gold(lib, name, fn, kind)
        star = "*" if kind == "crop" else " "
        print(f"{name:12} {kind:6} {stop1:5.1f}%{star} {rtop1:7.1f}% {rconf:7.1f}% "
              f"{itop1:7.1f}% {iconf:7.1f}% {gok:4}/{gtot}")
    print("\n* crop_* synth is circular (templates and queries both from render_node), ignore it; "
          "judge crop_* on real/gold.\n  ind = ocr+manual labels only (independent of any matcher, "
          "the honest bar); all includes the 63 'matcher'-sourced labels that flatter ncc.\n  "
          "conf50 = accuracy over the most-confident half, the fewer-OCR-hovers metric.")


def cmd_real(args):
    lib = load_matchable()
    kind, fn = build_registry(lib)[args.matcher]
    print(f"{args.matcher} on the real labeled set (misses shown):")
    records, higher, correct, total, skipped = eval_real(lib, args.matcher, fn, kind, verbose=True, progress=True)
    top1, conf, _ = summarize(records, higher)
    irec, ihigh, icorr, itot, _ = eval_real(lib, args.matcher, fn, kind, sources=SOURCES_INDEP)
    itop1, iconf, _ = summarize(irec, ihigh)
    print(f"   => all:         {correct}/{total} correct ({top1:.1f}%), conf50={conf:.1f}%, {skipped} skipped")
    print(f"   => independent: {icorr}/{itot} correct ({itop1:.1f}%), conf50={iconf:.1f}%  "
          f"(ocr+manual only, the honest bar)")


def cmd_synth(args):
    lib = load_matchable()
    kind, fn = build_registry(lib)[args.matcher]
    recs, higher = eval_synth(lib, args.matcher, fn, kind, n=args.n, seed=args.seed, noise=args.noise, progress=True)
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


# --------------------------------------------------------------- cnn vs ncc ship-decision readout

def _paired_eval(lib, sources=None):
    """aligned per-node results for cnn and ncc on the real labels. both are glyph matchers, so the
    glyph is extracted ONCE per node and fed to both, guaranteeing a paired comparison (needed for
    McNemar). returns per-node dicts, skipping labels not in the matchable library or where
    extraction fails (the same skips eval_real makes)."""
    rows = lib["rows"]
    keyset = {r["key"] for r in rows}
    Tz = lib["Tz"]
    net = cv2.dnn.readNetFromONNX(str(CNN_ONNX))
    B = cnn_bank(rows, net)
    out = []
    for rec in tqdm(load_real_labels(sources=sources), desc="cnn vs ncc"):
        want = rec["key"]
        crop = cv2.imread(str(ROOT / rec["crop_path"]))
        if want not in keyset or crop is None:
            continue
        rarity = rec.get("rarity")
        r = int(rec.get("r") or round(min(crop.shape[:2]) / (2 * BOX_K)))
        cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
        iso = D.isolate_node_contents(crop, cx, cy, r, rarity)
        if iso is None:
            continue
        glyph = D.normalize_glyph(iso[4], iso[3], rarity)
        if glyph is None:
            continue
        nrow, nscore, _, _ = D.id_icon_ncc(glyph, rows, Tz)
        q = _cnn_embed(net, glyph)
        s = B @ q
        o = np.argsort(-s)
        out.append({
            "key": want, "rarity": rarity, "source": rec.get("source"),
            "ncc_pred": nrow["key"], "ncc_hit": nrow["key"] == want, "ncc_score": float(nscore),
            "cnn_pred": rows[o[0]]["key"], "cnn_hit": rows[o[0]]["key"] == want,
            "cnn_score": float(s[o[0]]),
        })
    return out


def _mcnemar(cnn_hit, ncc_hit):
    """exact two-sided McNemar on the discordant pairs. returns (cnn_only, ncc_only, p). small N (148
    independent labels) so an exact binomial beats the chi-square approx. p is the prob of a split
    this lopsided if cnn and ncc were equally likely to win a discordant node."""
    from math import comb
    cnn_only = sum(a and not b for a, b in zip(cnn_hit, ncc_hit))     # cnn right, ncc wrong
    ncc_only = sum(b and not a for a, b in zip(cnn_hit, ncc_hit))     # ncc right, cnn wrong
    n = cnn_only + ncc_only
    if n == 0:
        return cnn_only, ncc_only, 1.0
    k = min(cnn_only, ncc_only)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return cnn_only, ncc_only, p


def _top1_conf50(recs, pfx):
    """(top1%, conf50%) for one matcher's prefix ('cnn'/'ncc') over paired per-node dicts."""
    return summarize([(r[f"{pfx}_hit"], r[f"{pfx}_score"]) for r in recs], higher=True)[:2]


def cmd_cnneval(args):
    """the ship-decision readout: cnn vs ncc on real labels, all(211) + independent(148), by rarity,
    a paired McNemar on the honest independent set, and the highest-confidence cnn misses (near-dups
    that slip the ocr gate). conf50 on independent is the headline (fewer ocr hovers).
    bar to beat: ncc 55.4% top1 / 66.2% conf50 on the 148 independent labels."""
    if not CNN_ONNX.is_file():
        print(f"no model at {CNN_ONNX} -- train first (python tools/glyph_cnn.py train)")
        return
    lib = load_matchable()
    allrecs = _paired_eval(lib)
    indep = [r for r in allrecs if r["source"] in SOURCES_INDEP]
    print(f"\ncnn vs ncc on real labels  (all {len(allrecs)} / independent {len(indep)})\n")
    print(f"{'set':14} {'cnn t1':>7} {'cnn c50':>8}   {'ncc t1':>7} {'ncc c50':>8}")
    for name, recs in (("all", allrecs), ("independent", indep)):
        ct1, cc50 = _top1_conf50(recs, "cnn")
        nt1, nc50 = _top1_conf50(recs, "ncc")
        print(f"{name:14} {ct1:6.1f}% {cc50:7.1f}%   {nt1:6.1f}% {nc50:7.1f}%")

    print("\nby rarity (independent, top1):")
    for rar in RARITY_CYCLE + [None]:
        rr = [r for r in indep if r["rarity"] == rar]
        if not rr:
            continue
        ct1, _ = _top1_conf50(rr, "cnn")
        nt1, _ = _top1_conf50(rr, "ncc")
        print(f"   {str(rar):11} n={len(rr):3}  cnn {ct1:5.1f}%  ncc {nt1:5.1f}%")

    cnn_only, ncc_only, p = _mcnemar([r["cnn_hit"] for r in indep], [r["ncc_hit"] for r in indep])
    verdict = "cnn wins" if cnn_only > ncc_only else ("ncc wins" if ncc_only > cnn_only else "tie")
    print(f"\nMcNemar (independent): cnn-only-right={cnn_only}  ncc-only-right={ncc_only}  "
          f"p={p:.3f}  -> {verdict}")

    miss = sorted([r for r in indep if not r["cnn_hit"]], key=lambda r: -r["cnn_score"])[:12]
    print("\nhighest-confidence cnn misses (independent, the ocr-gate slippers):")
    for r in miss:
        nc = "ncc:ok" if r["ncc_hit"] else f"ncc:{r['ncc_pred'][:14]}"
        print(f"   {r['key'][:18]:18} -> {r['cnn_pred'][:18]:18} s={r['cnn_score']:.2f}  {nc}")


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

    s_real = sub.add_parser("real", help="one matcher on the annotator's real labeled set")
    s_real.add_argument("--matcher", choices=MATCHER_NAMES, default="crop_resid")
    s_real.set_defaults(func=cmd_real)

    s_cnn = sub.add_parser("cnneval", help="cnn vs ncc ship readout (all/independent/rarity + McNemar)")
    s_cnn.set_defaults(func=cmd_cnneval)

    args = ap.parse_args()
    args.func(args)
