"""localize nodes, read rarity, identify the icon. the core offline-testable layer.

pipeline: hsv color-segment the rarity disks to get node centers plus a first read of
rarity, then crop + mask each inner icon and match it against the scraped library via
normalized cross-correlation (a perceptual-hash and a masked-ncc variant are also selectable).
every coord is found dynamically (no hardcoded positions) so it works for varying aspect ratios.
"""

import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import imagehash
from PIL import Image

from . import paths
from .node import demote_dead_art
from .resolution import Resolution

USR_HSV = paths.user_base() / "usr" / "rarity-HSVs.json"   # active anchors, evolve per web; user_base writable when frozen
DEFAULT_INDEX = paths.cache_dir() / "icons_index.json"     # scraped library, cache_dir writable when frozen (first-run scrape)

RARITIES = ["common", "uncommon", "rare", "very rare", "ultra rare", "event"]

# empirical rarity disk anchors, opencv hsv [h(0-179),s,v],
# read off tests/fixtures/ screenshots via `src.detect sample <web_path>`.
EMPIRICAL_SEED = {
    "common":     [11, 93, 51],     # brown
    "uncommon":   [61, 166, 72],    # green
    "rare":       [108, 141, 77],   # blue
    "very rare":  [143, 141, 77],   # purple
    "ultra rare": [171, 209, 117],  # pink/iri
    "event":      [21, 213, 172],   # gold/yellow
}

# hue carries the rarity signal (weight most), value is most gamma-sensitive (weight least);
# that weighting is most of our tolerance to dbd's adjustable gamma.
HSV_WEIGHTS = (4.0, 1.0, 0.3)

# normalized glyph canvas size. MUST match scraper.GLYPH_SIZE
GLYPH_SIZE = 128

# icon matcher selection, 'cnn' (learned embedding) is the DEFAULT;
# it beat the classical matchers decisively on real nodes (86.5% vs ncc 55.4% top1, see tools/glyph_cnn + eval_matchers cnneval).
# 'ncc' (plain z-normed cosine) is the fallback cnn degrades to if the model is absent, 'phash' (hamming) the original.
MATCHERS = ("cnn", "ncc", "ncc_masked", "phash")
NCC_RES = 96  # res of the ncc template/query vectors (96 beats 128 on conf50, cheaper vectors)

# learned matcher runtime, the encoder is trained offline (tools/glyph_cnn.py, torch dev-only) and exported to onnx;
# here we only RUN it via cv2.dnn so no torch ships. the onnx is a read-only bundled asset (resource_path),
# the per-sprite embedding bank builds into template_cache_dir on first use like the ncc templates.
# CNN_RES must match the encoder input (tools/glyph_cnn.INPUT_RES).
CNN_ONNX = paths.resource_path("data/models/glyph_encoder.onnx")
CNN_RES = 96

# rarity anchor colors: load from usr (auto-seeded from EMPIRICAL_SEED, hand-editable), then classify.
# per-user auto-calibration REJECTED 2026-07-16: self-poisons (a washed-out ultra rare would classify
# 'common' and drag common's anchor onto it), and gamma tolerance is already HSV_WEIGHTS' job.
_HSVs = None  # cached {rarity: [h, s, v]} for the run

NODE_SHAPE_DICT = { # geometric relationship of node content and node type
    'square': ['item', 'addon'], # hard to identify the '+' distinguishing item from addon, lump them together
    'rhombus': ['perk'],
    'hexagon': ['offering']
}

def _is_nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def _load_hsv(path):
    """read a rarity->hsv map. accepts the bare {rarity:[h,s,v]} or the seed's wrapper
    {"_note":..., "hsv":{...}}. keeps only the known rarity keys."""
    data = json.loads(path.read_text(encoding="utf-8"))
    hsv = data.get("hsv", data)
    return {k: [int(c) for c in v] for k, v in hsv.items() if k in RARITIES}


def _seed_usr_file(out_path=USR_HSV):
    """first run: copy EMPIRICAL_SEED into usr/ so the per-user file exists and can evolve.
    returns the seeded dict."""
    seed = {rar: list(hsv) for rar, hsv in EMPIRICAL_SEED.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    return seed


def get_ref_hsvs():
    """active anchors {rarity:[h,s,v]} (opencv hsv). usr/ if it exists, else auto-seed it from
    EMPIRICAL_SEED on first run. refine keeps usr/ current per web."""
    global _HSVs
    if _HSVs is not None:
        return _HSVs
    if _is_nonempty(USR_HSV):
        _HSVs = _load_hsv(USR_HSV)
        # backfill any anchor the usr file predates (e.g. the 'event' band added later),
        # so a stale usr/ doesn't silently drop a tier
        for rar, hsv in EMPIRICAL_SEED.items():
            _HSVs.setdefault(rar, list(hsv))
    else:
        _HSVs = _seed_usr_file()  # first run: write empirical seed -> usr/ and use it
    return _HSVs


def _hsv_dist(a, b):
    """weighted distance between two opencv hsv colors, hue treated as circular."""
    if a is None or b is None:
        return None
    dh = abs(int(a[0]) - int(b[0]))
    dh = min(dh, 180 - dh)            # hue wraps at 0/180, where brown/red sit
    ds = abs(int(a[1]) - int(b[1]))
    dv = abs(int(a[2]) - int(b[2]))
    wh, ws, wv = HSV_WEIGHTS
    return wh * dh + ws * ds + wv * dv


def hue_circular_delta(hues, ref):
    """signed shortest distance from ref to each hue on opencv's 0..179 wheel, in [-90, 90).
    hue wraps at the red/brown seam (0 and 180 are the same color)
    this is the circularity primitive behind both the disk-hue median and the rarity hue mask."""
    return (np.asarray(hues, dtype=np.float32) - ref + 90.0) % 180.0 - 90.0


def hue_circular_mean(hues):
    """circular mean of a set of opencv hues (0..179)
    average on the unit circle (hue is h*2 degrees) and map the angle back"""
    ang = np.asarray(hues, np.float32) * (np.pi / 90.0)   # 0..179 -> radians
    return (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * (90.0 / np.pi)


def hue_band_mask(hsv, h_ref, h_tol, s_floor=0, v_floor=0):
    """uint8 (h,w) mask of pixels within h_tol of h_ref on the hue wheel and above the s/v floors.
    wraps the 0/180 seam cv2.inRange can't (via hue_circular_delta): refine_node isolates one rarity, disk_color_mask ORs all five.
    hsv is a (h,w,3) opencv-hsv image."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    band = (np.abs(hue_circular_delta(h, h_ref)) <= h_tol) & (s >= s_floor) & (v >= v_floor)
    return band.astype(np.uint8) * 255


def classify_rarity(hsv, anchors=None):
    """nearest-anchor rarity for one sampled disk color. returns (rarity, distance)."""
    anchors = anchors or get_ref_hsvs()
    best, best_d = None, None
    for rar, ref in anchors.items():
        d = _hsv_dist(hsv, ref)
        if d == None:
            return None, None
        if best_d is None or d < best_d:
            best, best_d = rar, d
    return best, best_d


def rarity_candidates(hsv, anchors=None):
    """every rarity ranked by anchor distance, nearest first, [] if the disk read failed.
    classify_rarity only returns the head; detect walks the tail when the head's hue band can't
    isolate a plate (see isolate_node_glyph)."""
    anchors = anchors or get_ref_hsvs()
    d = {rar: _hsv_dist(hsv, ref) for rar, ref in anchors.items()}
    if any(v is None for v in d.values()):
        return []
    return sorted(d, key=d.get)


def disk_color_mask(frame, htol=8, s_floor=62, v_floor=38):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # (H,W,3) uint8, h in 0..179
    anchors = get_ref_hsvs() # rarity: [h,s,v]
    frame_mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
    
    for rarity, (h, s, v) in anchors.items():
        frame_mask |= hue_band_mask(hsv, h, htol, s_floor, v_floor)  # wrap-safe, same util as refine_node

    return frame_mask


def load_rows(index_path=DEFAULT_INDEX):
    """just the metadata rows, no phashes. the ui only ever needs names/rarities/sprite paths, and
    stacking the phashes costs ~20ms of startup it would never look at."""
    rows = json.loads(Path(index_path).read_text(encoding="utf-8"))
    # backfill for indexes scraped before the wiki's "Visceral" tier was recognized: those rows have
    # rarity null but their lead sentence says so ("... is a Visceral Add-on ..."). in-game they draw
    # the iridescent pink disk, so they are our "ultra rare"; fixing here spares users a re-scrape.
    for r in rows:
        if r.get("rarity") is None and "is a Visceral" in (r.get("desc") or ""):
            r["rarity"] = "ultra rare"
    demote_dead_art(rows)  # same spare-a-re-scrape treatment, for renamed perks' orphan uploads
    return rows


def is_matchable(row):
    """can this row ever be a bloodweb node? excludes obtainable=='unavailable' (killer powers,
    retired content) and non-perk rows with no rarity tier (raw-upload twins, in-match pickups).
    applied at load so old indexes heal without a re-scrape."""
    if row.get("obtainable") == "unavailable":
        return False
    return row.get("category") == "perk" or row.get("rarity") is not None


def load_index(index_path=DEFAULT_INDEX):
    """returns (rows, hashes). rows = metadata dicts; hashes = (n, 64) bool array of the
    precomputed phashes, stacked to query with a single vector op."""
    rows = load_rows(index_path)
    hashes = np.stack([
        imagehash.hex_to_hash(r["phash"]).hash.flatten() for r in rows
    ])  # (n, 64)
    return rows, hashes


def id_icon_hamming(icon_bgr, rows, ref_hashes, pool=None):
    """nearest neighbor of an unknown icon via hamming distance between its phash and the library phashes.
    pool is an optional bool mask (n,) restricting the search to a category (once we know the socket shape).
    returns (row, dist, margin, runner_up_row): best row, its hamming distance (0..64, lower better), gap to the 2nd best (small = ambiguous), and that 2nd-best row (debug)."""
    gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY) # same luma weights as PIL 'L'
    q = imagehash.phash(Image.fromarray(gray)).hash.flatten() # (64,) bool
    dists = np.count_nonzero(ref_hashes != q, axis=1) # (n,) hamming: differing bits
    if pool is not None:
        dists = np.where(pool, dists, 65) # 65 > max dist of 64
    order = np.argsort(dists)
    return rows[order[0]], int(dists[order[0]]), int(dists[order[1]] - dists[order[0]]), rows[order[1]]


def _sprite_glyph_gray(path, out_size=GLYPH_SIZE):
    """load a library sprite and frame it like scraper.normalize_sprite (alpha tight-crop -> square-pad on black -> resize), returning a grayscale (h,w) uint8 glyph.
    the ncc templates are built from these so a query glyph (normalize_glyph output) compares cleanly."""
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    g = img.crop(bbox) if bbox else img
    gw, gh = g.size
    side = max(gw, gh)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    canvas.alpha_composite(g, ((side - gw) // 2, (side - gh) // 2))
    return np.array(canvas.resize((out_size, out_size), Image.LANCZOS).convert("L"))


def _glyph_to_vec(gray, res=NCC_RES):
    """gray GLYPH_SIZE glyph -> raw res*res float32 vector"""
    return cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def _library_fingerprint(rows):
    """short content hash of what each row IS (key+phash, in order). the template/bank caches are
    POSITIONAL arrays aligned to rows, and a length+mtime check can't see a re-scrape that keeps the
    count but re-identifies or reorders rows -- a pre-re-scrape bank passed those checks for days and
    scrambled 1012/1648 matches (the 2026-07-16 stale-bank incident). baking this into the cache
    filename makes any library change a guaranteed miss."""
    h = hashlib.sha1()
    for r in rows:
        h.update(f"{r.get('key')}|{r.get('phash')};".encode())
    return h.hexdigest()[:10]


def _write_cache(cache, arr, stale_glob):
    """save a rebuilt template/bank array and drop superseded fingerprints of the same artifact, so
    the cache dir keeps one live copy instead of accreting one file per library state.
    soft on OSError (read-only cache dir when frozen): the array still serves from memory."""
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        for old in cache.parent.glob(stale_glob):
            if old != cache:
                try:
                    old.unlink()
                except OSError:
                    pass  # e.g. held open by another instance; a stale NAME can no longer be served
        np.save(cache, arr)
    except OSError:
        pass  # read-only cache dir (e.g. frozen exe), just rebuild next run


def load_ncc_templates(rows, index_path=DEFAULT_INDEX, res=NCC_RES):
    """build (or load cached) the ncc template matrix: each library sprite -> a res*res grayscale vector,
    returns (T, T2): T = (n, res*res) float32 raw vectors,
    T2 = T**2 (reused for the masked-cosine template norms)"""
    # cache in the disposable template cache (repo data/cache in dev, .../cache/templates when
    # frozen), so the save works even when the bundled index sits in a read-only dir.
    # keyed by CONTENT (see _library_fingerprint), not the index mtime.
    stem = Path(index_path).stem
    cache = paths.template_cache_dir() / f"{stem}.ncc{res}-{_library_fingerprint(rows)}.npy"
    if cache.is_file():
        T = np.load(cache)
        if T.shape == (len(rows), res * res):
            return T, T * T
    base = Path(index_path).parent
    T = np.empty((len(rows), res * res), np.float32)
    for i, r in enumerate(rows):
        T[i] = _glyph_to_vec(_sprite_glyph_gray(base / r["file"]), res)
    _write_cache(cache, T, f"{stem}.ncc{res}*.npy")
    return T, T * T


def id_icon_ncc_masked(icon_bgr, rows, templates, pool=None, res=NCC_RES, fg_frac=0.15, fg_floor=20):
    """masked normalized cross-correlation matcher.
    build a foreground mask from the query glyph's bright strokes, score each template by cosine over just those pixels (templates = (T, T2) from load_ncc_templates).
    returns (row, score, margin, runner_up_row): score = masked cosine in [-1,1] (higher better, unlike phash), margin = best - 2nd, runner_up_row = 2nd-best (debug)."""
    T, T2 = templates
    q = _glyph_to_vec(cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY), res)
    m = (q > max(fg_floor, q.max() * fg_frac)).astype(np.float32)   # query foreground strokes
    if m.sum() < 1:
        m = np.ones_like(q)
    qc = (q - q[m > 0].mean()) * m                                  # center over fg, zero elsewhere
    nq = np.linalg.norm(qc) + 1e-6
    tnorm = np.sqrt(T2 @ m) + 1e-6                                  # ||template * mask|| per row
    scores = (T @ qc) / (tnorm * nq)                               # (n,) masked cosine
    if pool is not None:
        scores = np.where(pool, scores, -2.0)                      # below the min possible cosine
    order = np.argsort(-scores)                                    # higher cosine = better
    return (rows[order[0]], float(scores[order[0]]),
            float(scores[order[0]] - scores[order[1]]), rows[order[1]])


def ncc_plain_templates(templates):
    """z-normed (mean-removed, unit-norm) template matrix for the plain-ncc matcher.
    plain ncc keeps the whole glyph-on-black silhouette rather than masking to bright strokes, so it z-norms the full vector.
    built once per run then reused across nodes."""
    T, _ = templates
    Tz = T - T.mean(axis=1, keepdims=True)         # remove each template's DC level
    Tz /= (np.linalg.norm(Tz, axis=1, keepdims=True) + 1e-6)
    return Tz


def id_icon_ncc(icon_bgr, rows, Tz, pool=None, res=NCC_RES):
    """plain normalized cross-correlation matcher.
    z-normed cosine between the whole query glyph and each z-normed template (Tz from ncc_plain_templates); unlike id_icon_ncc_masked this keeps the full glyph-on-black silhouette rather than masking to bright strokes.
    returns (row, score, margin, runner_up_row): score = cosine in [-1,1] (HIGHER better), margin = gap to 2nd best, runner_up_row = 2nd-best (debug)."""
    q = _glyph_to_vec(cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY), res)
    q = q - q.mean()
    q /= (np.linalg.norm(q) + 1e-6)
    scores = Tz @ q                                                 # (n,) z-normed cosine
    if pool is not None:
        scores = np.where(pool, scores, -2.0)                      # below the min possible cosine
    order = np.argsort(-scores)                                    # higher cosine = better
    return (rows[order[0]], float(scores[order[0]]),
            float(scores[order[0]] - scores[order[1]]), rows[order[1]])


# ---- learned (cnn) matcher: embed the glyph via cv2.dnn, nearest cosine over a sprite-embedding bank

_CNN_NET = None                # lazily loaded cv2.dnn net (one model, reused across nodes)
_CNN_BANK = None               # (fingerprint, (n,128) bank) so repeated detect() calls skip the rebuild
_ONNX_FP = None                # ((mtime, size), sha1[:10]) of CNN_ONNX so the 2mb hash runs once per run


def _onnx_fingerprint():
    """short content hash of the encoder weights. the bank must be rebuilt on a retrain, but the
    bundled onnx's MTIME is worthless for that: pyinstaller stamps it with the exe build time, so
    every rebuild of an unchanged model looked like a retrain (and a genuinely different model
    restored from backup could look fresh). hash the bytes instead; mtime+size only gate re-hashing."""
    global _ONNX_FP
    st = Path(CNN_ONNX).stat()
    key = (st.st_mtime, st.st_size)
    if _ONNX_FP is None or _ONNX_FP[0] != key:
        _ONNX_FP = (key, hashlib.sha1(Path(CNN_ONNX).read_bytes()).hexdigest()[:10])
    return _ONNX_FP[1]


def reset_library_caches():
    """drop the in-memory library-derived cache (the cnn embed bank). the ui calls this after a
    re-scrape: the content-keyed lookups would miss anyway once the rows change, but an explicit
    reset frees the old bank and can't leave a session serving one against a stale rows snapshot."""
    global _CNN_BANK
    _CNN_BANK = None


def _sprite_glyph_color(path):
    """clean color sprite framed like _sprite_glyph_gray but kept in bgr at native square size, the cnn anchor.
    must match the training anchor (tools/synth_glyphs.gallery_glyph) so a bank embedding lands where the encoder learned to map this icon; _cnn_blob does the resize to the encoder input."""
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    g = img.crop(bbox) if bbox else img
    gw, gh = g.size
    side = max(gw, gh)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    canvas.alpha_composite(g, ((side - gw) // 2, (side - gh) // 2))
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cnn_blob(bgr, res=CNN_RES):
    """glyph/anchor bgr -> (1,3,res,res) float32 nchw in [0,1], the cv2.dnn input.
    same INTER_AREA resize + /255 the encoder trained on (tools/synth_glyphs.to_input); bgr order kept on purpose."""
    x = cv2.resize(bgr, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def _cnn_embed(net, bgr):
    """one glyph -> l2-normed 128 embedding.
    the l2-norm is done here in numpy, not in the onnx graph, so the export stays basic-ops for the old opencv (4.6.0) dnn importer."""
    net.setInput(_cnn_blob(bgr))
    e = net.forward().reshape(-1).astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def load_cnn_model():
    """the cv2.dnn encoder, or None if the trained onnx isn't present yet (a fresh dev checkout without a model degrades to ncc instead of crashing).
    re-checks the file until it loads, then caches the net for the run."""
    global _CNN_NET
    if _CNN_NET is not None:
        return _CNN_NET
    if not Path(CNN_ONNX).is_file():
        return None
    _CNN_NET = cv2.dnn.readNetFromONNX(str(CNN_ONNX))
    return _CNN_NET


def load_cnn_bank(rows, index_path=DEFAULT_INDEX):
    """(n,128) l2-normed sprite-embedding bank aligned to rows, cached to template_cache_dir like the ncc templates.
    keyed by CONTENT, not length+mtimes: the filename fingerprints the rows' key+phash sequence and
    the onnx bytes, so a re-scraped/re-identified library or a retrained model can never be served a
    bank built for a different one. the old mtime check let a pre-re-scrape bank pass as fresh and
    scramble 1012/1648 matches (the 2026-07-16 stale-bank incident).
    built over the FULL rows (pool masks the unavailable ones at match time, same as the ncc path)."""
    global _CNN_BANK
    fp = f"{_library_fingerprint(rows)}-{_onnx_fingerprint()}"
    if _CNN_BANK is not None and _CNN_BANK[0] == fp:
        return _CNN_BANK[1]
    stem = Path(CNN_ONNX).stem
    cache = paths.template_cache_dir() / f"embed-{stem}-{CNN_RES}-{fp}.npy"
    B = None
    if cache.is_file():
        B = np.load(cache)
        if B.shape[0] != len(rows):
            B = None
    if B is None:
        net = load_cnn_model()
        base = Path(index_path).parent
        B = np.stack([_cnn_embed(net, _sprite_glyph_color(base / r["file"])) for r in rows]).astype(np.float32)
        _write_cache(cache, B, f"embed-{stem}-{CNN_RES}-*.npy")
    _CNN_BANK = (fp, B)
    return B


def id_icon_cnn(icon_bgr, rows, bank, net, pool=None):
    """learned metric matcher: embed the query glyph, take the nearest cosine over the cached sprite-embedding bank (bank (n,128) l2-normed aligned to rows, net the cv2.dnn encoder).
    pool masks out-of-pool rows like id_icon_ncc.
    returns (row, score, margin, runner_up_row, runner_up_sim); score = cosine in [-1,1] (HIGHER
    better), runner_up_sim = anchor cosine between the top two rows for the near-dup veto (None when
    the runner-up is out-of-pool, i.e. there is no real 2nd candidate to be confused with)."""
    q = _cnn_embed(net, icon_bgr)
    scores = bank @ q                                              # (n,) cosine, both l2-normed
    if pool is not None:
        scores = np.where(pool, scores, -2.0)                      # below the min possible cosine
    order = np.argsort(-scores)
    i0, i1 = order[0], order[1]
    # anchor-anchor cosine of the top two = how intrinsically confusable they are, independent of
    # this query's noise; only meaningful when the runner-up is in-pool (masked rows sit at -2.0).
    runner_sim = float(bank[i0] @ bank[i1]) if scores[i1] > -1.5 else None
    return (rows[i0], float(scores[i0]),
            float(scores[i0] - scores[i1]), rows[i1], runner_sim)


def _crop_glyph_from_frame(frame, x, y, r, r_tol=1):
    """crop frame around glyph bbox"""
    x, y, r, r_eff = int(x), int(y), int(r), int(r * r_tol)
    h, w, = frame.shape[:2]
    
    x0, xf = max(x-r_eff, 0), min(w, x+r_eff)
    y0, yf = max(y-r_eff, 0), min(h, y+r_eff)
    return frame[y0:yf, x0:xf], {'x0': x0, 'y0': y0, 'xf': xf, 'yf': yf}


def _fill_holes(bin_img):
    """fill regions enclosed by a closed rim so each ringed disk becomes a solid blob.
    pad a 1px bg ring first so the flood seed (0,0) is ALWAYS background even when the mask runs to the crop edge;
    without the pad a blob touching (0,0) makes floodfill paint the whole crop white."""
    padded = cv2.copyMakeBorder(bin_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8) # floodFill wants a +2 border mask
    cv2.floodFill(ff, mask, (0, 0), 255) # flood outside bg white from the padded corner
    ff = ff[1:-1, 1:-1] # drop the pad ring back off
    holes = cv2.bitwise_not(ff) # enclosed interiors only
    return bin_img | holes


def _binarize(img, thresh_method='adaptive_gaussian', blur_ksize=5, canny_lo=0, canny_hi=255):
    if thresh_method.lower() == 'adaptive_gaussian':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) # blur a little to tame glyph/web texture
        #bin_frame = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)
        bin_frame = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

    elif thresh_method.lower() == 'otsu':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) # blur a little to tame glyph/web texture
        _, bin_frame = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif thresh_method.lower() == 'canny':
        blur_frame = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0) # binarized canny on bloodweb nodes sucks in b/w, blur on color
        bin_frame = cv2.Canny(blur_frame, canny_lo, canny_hi, L2gradient=True)

    else:
        raise ValueError("Invalid Thresholding Method. Valid types: ['adaptive_gaussian', 'otsu', 'canny']")
    
    return bin_frame


def find_circles(
        frame, blur_ksize=11, open_ksize=11, close_ksize=5,
        thresh_method='adaptive_gaussian', use_hough=False,
        canny_lo=0, canny_hi=255, dp=1.5, circularity_thresh=0.78, r0_floor=0.6,
        rmin=None, rmax=None, min_dist_frac=4, accumulator_thresh=20,
        resolution=None, debug=False
    ):
    """detect bloodweb nodes in a given input frame.

    Preprocessing: choice between 3 different threshold methods (adaptive_gaussian, otsu, canny) to preprocess for contour detection
    morphological closing -> flood fill -> morphological open to close gaps and denoise

    Detection: Either Hough circles (if use_hough=True) or coarse then refine fine pass over the preprocessed contours
        - rough pass to grab the radii of all countour within (rmin,rmax)
        - peak finding to identify node centers

    rmin/rmax default to None -> resolution.rmin/rmax (resolution defaults to Resolution.from_frame(frame)); an explicit rmin/rmax still overrides.
    """
    resolution = resolution or Resolution.from_frame(frame)
    rmin = resolution.rmin if rmin is None else rmin
    rmax = resolution.rmax if rmax is None else rmax

    bin_frame = _binarize(frame, thresh_method=thresh_method, blur_ksize=blur_ksize, canny_lo=canny_lo, canny_hi=canny_hi)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))   
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))  

    # Morphological methods to close gaps and break unwanted contour connections
    if debug: _show(bin_frame, title='find_circles() - 0initial contours', contours=cv2.findContours(bin_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], savefig=True)
    bin_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_CLOSE, close_kernel)
    if debug: _show(bin_frame, title='find_circles() - 1morph close', contours=cv2.findContours(bin_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], savefig=True)
    bin_frame = _fill_holes(bin_frame)
    if debug: _show(bin_frame, title='find_circles() - 2fill holes', contours=cv2.findContours(bin_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], savefig=True)
    bin_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_OPEN, open_kernel)
    if debug: _show(bin_frame, title='find_circles() - 3morph open', contours=cv2.findContours(bin_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], savefig=True)

    circles=[]
    if use_hough:
        hough_circ = cv2.HoughCircles(
            bin_frame, cv2.HOUGH_GRADIENT, dp=dp,
            param1=canny_hi, param2=accumulator_thresh,
            minDist=int(min_dist_frac * rmin),
            minRadius=rmin, maxRadius=rmax
        )
        if hough_circ is not None:
            circles = [] if hough_circ is None else [(float(x), float(y), float(r)) for x, y, r in hough_circ[0]]
    else:
        # first pass to detect approximate core radius size by median majority vote
        contours, _ = cv2.findContours(bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #_show(bin_frame, title=f"contours in find circles() {thresh_method}", contours=contours)
        radii = []
        for c in contours:
            (_, _), r = cv2.minEnclosingCircle(c)
            if rmin <= r <= rmax:
                radii.append(r)
        if not radii:
            # nothing node-sized in the frame (menu, transition, prestige screen);
            # returning empty beats the int(nan) crash the median below would raise
            print("ERROR: no radii found in initial pass")
            return []
        r0 = np.median(radii)
        #print(r0)

        # split nodes with merging contours
        # get image of distances of each pixel from its nearest nonzero pixel (contour bounds)
        # local maxima -> contour peaks; value of local maxima ~ node radius
        dist = cv2.distanceTransform(bin_frame, cv2.DIST_L2, 5) # ~r at each disk center
        dist = cv2.GaussianBlur(dist, (5,5), 0.5)
        max_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r0), int(r0)))
        
        # find the maximum distance of each node contour and set that max dist value to all pixel values within r0
        smoothed_local_maxima = cv2.dilate(dist, max_kernel) # smooths noisy/spiky local maxima to one value
        if debug: _show(smoothed_local_maxima, title="dilate dist transform")

        # find where in the distance transform does the contour area eq that max value
        is_peak = (dist == smoothed_local_maxima) & (dist > r0_floor * r0)
        is_peak = is_peak.astype(np.uint8)
        n, _, _, centroids = cv2.connectedComponentsWithStats(is_peak) # group local maxima blobs into centroids``

        for i in range(1,n):
            cx, cy = centroids[i]
            r = float(dist[int(cy), int(cx)])
            # upper radius gate: nodes on one web are all ~r0 (median),
            # so a peak well past it is a non-node blob (masked-ui rect, campfire texture: dist 79 vs r0 55 on the 005122 fixture).
            # r0-relative since 1.2*rmax let that campfire blob through.
            # the LOWER bound stays the loose is_peak r0_floor on purpose: a partially filled disk (rim gap the flood fill leaked through) reads a LOW dist and must not be dropped.
            if r <= 1.35 * r0:
                circles.append((cx, cy, r))

    if debug: _show(bin_frame, title='find_circles() - final', circles=circles, contours=contours, savefig=False)
    return circles # list of (x, y, r) float


# ---------------------------------------------------------------------------
# slot lattice: bloodweb nodes sit on a FIXED polar grid around the web center (Resolution.LATTICE_*).
# find_circles' contour pass both misses real nodes (dim/unreachable, event golds) and invents circles from fog/ground/web junctions,
# but the two populations separate cleanly by distance to the nearest slot (~10px real vs 40px+ junk on every web measured).
# so: fit the lattice, SNAP what lands on a slot, DROP what doesn't, and run a matched-filter presence test on the empty slots to recover misses.
# ---------------------------------------------------------------------------

LATTICE_SNAP_TOL = 25    # baseline px, max circle-to-slot distance to accept a snap
LATTICE_TPL_TOL = 12     # baseline px, only this-well-centered snaps feed the template
PRESENCE_HALF = 62       # baseline px, half-size of the presence template crop
PRESENCE_SEARCH = 20     # baseline px, +- search window around an empty slot
PRESENCE_THRESH = 0.32   # presence floor: real misses scored >=0.39, empty slots <=0.27
PRESENCE_MIN_N = 40      # nodes the template must average before recovery runs (~2 scans)
PRESENCE_MAX_N = 2000    # stop accumulating here, the mean is long converged
TPL_CANON = 2 * PRESENCE_HALF   # template edge at baseline scale, one cache serves any resolution


def set_presence_thresh(v=PRESENCE_THRESH):
    """runtime override of the presence floor from config (settings ui: presence_thresh), so the
    0.39-vs-0.27 separation measured here can be re-tuned live on machines that read differently."""
    global PRESENCE_THRESH
    PRESENCE_THRESH = float(v)


def lattice_slots(cx, cy, scale):
    """slot centers of the bloodweb lattice at a given center and scale.
    returns ((n,2) float xy, (n,) ring index), n=30 at the full three-ring layout."""
    pts, rings = [], []
    for ri, (rr, ph, ns) in enumerate(zip(
            Resolution.LATTICE_RADII, Resolution.LATTICE_PHASES, Resolution.LATTICE_SLOTS)):
        ang = np.radians(ph + np.arange(ns) * (360.0 / ns))
        pts.append(np.stack(
            [cx + scale * rr * np.cos(ang), cy + scale * rr * np.sin(ang)], axis=1))
        rings.extend([ri] * ns)
    return np.concatenate(pts), np.asarray(rings)


def _center_cost(pts, cx_grid, cy_grid, scale):
    """clipped-and-normalized lattice cost of candidate centers: how far each circle sits from its nearest ring radius, capped at 30px-of-scale.
    so junk circles can't drag the fit and it stays comparable ACROSS scales (an un-normalized clip always favors the smaller scale)."""
    radii = scale * np.asarray(Resolution.LATTICE_RADII)
    d = np.sqrt((pts[:, 0, None, None] - cx_grid) ** 2 + (pts[:, 1, None, None] - cy_grid) ** 2)
    resid = np.min(np.abs(d[..., None] - radii), axis=-1)
    return np.minimum(resid / (30.0 * scale), 1.0).mean(axis=0)


def _fit_center_slots(pts, c0, scale, span, step):
    """center grid refine against the FULL slot positions: the radial cost alone leaves the angle unconstrained (a shifted center can ride circles onto a neighboring ring), the slot distance can't be gamed that way.
    span/step are in baseline px like the other tolerances."""
    offs, _ = lattice_slots(0.0, 0.0, scale)   # slot offsets from a zero center
    xs = np.arange(c0[0] - span * scale, c0[0] + span * scale + 1e-6, step * scale)
    ys = np.arange(c0[1] - span * scale, c0[1] + span * scale + 1e-6, step * scale)
    XX, YY = np.meshgrid(xs, ys)
    mind = None
    for ux, uy in offs:   # running min over slots keeps the arrays (n, ny, nx) flat
        d = np.sqrt((pts[:, 0, None, None] - (XX + ux)) ** 2
                    + (pts[:, 1, None, None] - (YY + uy)) ** 2)
        mind = d if mind is None else np.minimum(mind, d)
    cost = np.minimum(mind / (30.0 * scale), 1.0).mean(axis=0)
    iy, ix = np.unravel_index(np.argmin(cost), cost.shape)
    return float(XX[iy, ix]), float(YY[iy, ix])


def _fit_center_scale_joint(pts, s_lo, s_hi):
    """joint coarse search over (center, scale): the marginals are unstable (a center found at the wrong scale drifts, a scale swept at the wrong center aliases onto the wrong ring), so sweep scale and grid the center together and keep the jointly cheapest cell."""
    c0 = pts.mean(axis=0)
    best = None
    for s in np.arange(s_lo, s_hi, 0.05):
        span, step = 150.0 * s, 5.0 * s
        xs = np.arange(c0[0] - span, c0[0] + span + 1e-6, step)
        ys = np.arange(c0[1] - span, c0[1] + span + 1e-6, step)
        XX, YY = np.meshgrid(xs, ys)
        cost = _center_cost(pts, XX, YY, s)
        iy, ix = np.unravel_index(np.argmin(cost), cost.shape)
        if best is None or cost[iy, ix] < best[0]:
            best = (float(cost[iy, ix]), float(XX[iy, ix]), float(YY[iy, ix]), float(s))
    return best[1], best[2], best[3]


def _scale_search(d, s_lo, s_hi):
    """1-d sweep for the lattice scale: the s that puts the circle-to-center distances nearest the ring radii.
    residuals clip NORMALIZED (resid / clip, like _center_cost) so junk circles can't drag it and a smaller s can't win by shrinking every clip.
    the caller must bound the sweep off the frame scale: with only one ring occupied (early webs) the inner ring fits ring 1 at half scale exactly, and only the bounds break that tie."""
    radii = np.asarray(Resolution.LATTICE_RADII)
    ss = np.arange(s_lo, s_hi, 0.01)
    resid = np.min(np.abs(d[:, None, None] - ss[None, :, None] * radii[None, None, :]), axis=2)
    cost = np.minimum(resid / (30.0 * ss[None, :]), 1.0).mean(axis=0)
    return float(ss[np.argmin(cost)])


def fit_lattice(circles, center=None, scale_hint=1.0):
    """fit (cx, cy, scale) of the slot lattice to find_circles output.
    center is the glow center when the caller has one (find_center_node), else it is grid-searched jointly with the scale.
    scale comes from sweeping the circle-to-ring distances against the known ring radii, bounded off scale_hint (the frame's Resolution.scale): a web-bbox crop under-reads the true scale, so the bracket runs 0.85-1.6x, wide enough for crop-vs-full but too narrow for the half-scale alias a single-ring web would otherwise fit.
    (the detected circle radii are NOT the scale source: dist-transform peaks stop at the socket ring and under-read ~20%.)
    returns None when the circles are too few or the best fit doesn't hold the lattice."""
    if len(circles) < (3 if center is not None else 6):
        return None
    pts = np.asarray([(x, y) for x, y, _ in circles], dtype=np.float32)
    radii = np.asarray(Resolution.LATTICE_RADII)
    s_lo, s_hi = 0.85 * scale_hint, 1.6 * scale_hint

    if center is not None:
        cx, cy = float(center[0]), float(center[1])
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        s = _scale_search(d, s_lo, s_hi)
    else:
        cx, cy, s = _fit_center_scale_joint(pts, s_lo, s_hi)
        cx, cy = _fit_center_slots(pts, (cx, cy), s, span=80.0, step=4.0)
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        s = _scale_search(d, max(s_lo, s - 0.1), min(s_hi, s + 0.1))
    for _ in range(2):   # polish scale on the inlier median, re-center when the center was fitted
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        ring = np.argmin(np.abs(d[:, None] - s * radii[None, :]), axis=1)
        ratio = d / radii[ring]
        ok = np.abs(ratio - s) <= 0.08 * s
        if ok.sum() >= 3:
            s = float(np.median(ratio[ok]))
        if center is None:
            cx, cy = _fit_center_slots(pts, (cx, cy), s, span=8.0, step=1.0)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    inliers = int((np.min(np.abs(d[:, None] - s * radii[None, :]), axis=1)
                   <= LATTICE_SNAP_TOL * s).sum())
    if inliers < 3:
        return None   # the circles don't hold the lattice (not a bloodweb frame?)
    return cx, cy, s


def snap_to_lattice(circles, cx, cy, scale):
    """assign each circle to its nearest slot, one circle per slot (nearest wins).
    snapped nodes take the SLOT position and calibrated radius: a real node sits within ~10px of its slot while a merged-contour centroid can drift 40px off, so the slot is the safer click/crop center.
    returns (snapped [(x,y,r,slot_idx)], taken slot-idx set, dropped [(x,y,r,slot_dist)], tpl_pts well-centered circle centers)."""
    slots, _ = lattice_slots(cx, cy, scale)
    node_r = Resolution.NODE_RADIUS * scale
    best, dropped = {}, []
    for x, y, r in circles:
        d = np.hypot(slots[:, 0] - x, slots[:, 1] - y)
        i = int(np.argmin(d))
        if d[i] > LATTICE_SNAP_TOL * scale:
            dropped.append((x, y, r, float(d[i])))
        elif i not in best or d[i] < best[i][0]:
            best[i] = (float(d[i]), (x, y))
    snapped = [(float(slots[i][0]), float(slots[i][1]), node_r, i) for i in sorted(best)]
    tpl_pts = [xy for dist, xy in best.values() if dist <= LATTICE_TPL_TOL * scale]
    return snapped, set(best), dropped, tpl_pts


_RING_TPL = None    # cached [gray_sum, grad_sum, n] for the run


def _ring_tpl_path():
    return paths.template_cache_dir() / "ring-template.npz"


def _get_ring_tpl():
    global _RING_TPL
    if _RING_TPL is None:
        p = _ring_tpl_path()
        if p.exists():
            z = np.load(p)
            _RING_TPL = [z["gray"], z["grad"], int(z["n"])]
        else:
            _RING_TPL = [np.zeros((TPL_CANON, TPL_CANON), np.float64),
                         np.zeros((TPL_CANON, TPL_CANON), np.float64), 0]
    return _RING_TPL


def _grad_mag(gray32):
    """sobel gradient magnitude, the texture-insensitive half of the presence score."""
    gx = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _crop_square(img, x, y, half):
    """(2*half, 2*half) crop centered on (x, y), or None when it runs off the frame."""
    x, y = int(round(x)), int(round(y))
    if x - half < 0 or y - half < 0 or x + half > img.shape[1] or y + half > img.shape[0]:
        return None
    return np.ascontiguousarray(img[y - half:y + half, x - half:x + half])


def update_ring_template(gray, grad, tpl_pts, scale):
    """fold this scan's well-centered snapped nodes into the running mean node crop.
    over enough nodes the glyphs average out and the ring + plate signature remains, giving a matched filter for 'a node is here' from the USER'S OWN capture (map, gamma, resolution).
    persisted like the ncc cache so it survives runs."""
    tpl = _get_ring_tpl()
    if tpl[2] >= PRESENCE_MAX_N or not tpl_pts:
        return
    half = int(round(PRESENCE_HALF * scale))
    for x, y in tpl_pts:
        cg = _crop_square(gray, x, y, half)
        ce = _crop_square(grad, x, y, half)
        if cg is None or ce is None:
            continue
        tpl[0] += cv2.resize(cg, (TPL_CANON, TPL_CANON)).astype(np.float64)
        tpl[1] += cv2.resize(ce, (TPL_CANON, TPL_CANON)).astype(np.float64)
        tpl[2] += 1
    try:
        p = _ring_tpl_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, gray=tpl[0], grad=tpl[1], n=tpl[2])
    except OSError:
        pass    # cache write failing shouldn't kill a scan, the in-memory template still works


def recover_missed_slots(gray, grad, empty_slots, scale, debug=False):
    """matched-filter presence test on the slots the contour pass left empty.
    scores each slot with z-normed ncc of the mean-node template on gray + gradient crops (mean of the two, +-PRESENCE_SEARCH window); on the validation scans every real miss scored >=0.39 and every empty slot <=0.27.
    empty_slots is [(slot_idx, x, y)]; returns [(slot_idx, x, y, score)] at the ncc peak, empty until the template has seen PRESENCE_MIN_N nodes."""
    tpl = _get_ring_tpl()
    if tpl[2] < PRESENCE_MIN_N:
        if debug:
            print(f"[lattice] presence template still warming ({tpl[2]}/{PRESENCE_MIN_N} nodes)")
        return []
    half = int(round(PRESENCE_HALF * scale))
    search = int(round(PRESENCE_SEARCH * scale))
    tg = cv2.resize((tpl[0] / tpl[2]).astype(np.float32), (2 * half, 2 * half))
    te = cv2.resize((tpl[1] / tpl[2]).astype(np.float32), (2 * half, 2 * half))
    h_img, w_img = gray.shape[:2]
    found = []
    for slot_i, sx, sy in empty_slots:
        # web bbox crop can run right up to the outer ring,
        # so shrink the search window to whatever margin the frame leaves rather than skip the edge slot (peak is within a few px anyway); below zero even the template doesn't fit
        sxi, syi = int(round(sx)), int(round(sy))
        margin = min(sxi, syi, w_img - sxi, h_img - syi)
        s_here = min(search, margin - half)
        if s_here < 0:
            continue
        cg = _crop_square(gray, sxi, syi, half + s_here)
        ce = _crop_square(grad, sxi, syi, half + s_here)
        if cg is None or ce is None:
            continue
        comb = 0.5 * cv2.matchTemplate(cg, tg, cv2.TM_CCOEFF_NORMED) \
             + 0.5 * cv2.matchTemplate(ce, te, cv2.TM_CCOEFF_NORMED)
        iy, ix = np.unravel_index(int(np.argmax(comb)), comb.shape)
        score = float(comb[iy, ix])
        if score >= PRESENCE_THRESH:
            found.append((slot_i, sxi - s_here + int(ix), syi - s_here + int(iy), score))
    return found


# ---------------------------------------------------------------------------
# node state: is this node still buyable, or has it already been taken?
# dbd auto-buys the cheapest PATH to whatever you click and the entity eats nodes mid-web, so a
# once-per-level snapshot goes stale and the spender re-clicks dead nodes (measured: 8 of 22 targets
# on tests/fixtures/web-130708).
#
# the signal is in the socket RING, not the glyph: a taken node keeps its plate and its icon (which
# is why the matcher happily re-identifies it), and normalize_glyph throws the ring away before the
# matcher ever sees it. so this is a color read, and it cannot be a glyph-model read.
#   bought  a BRIGHT red ring hugging the plate  (measured on the 0.74-0.90r band)
#   entity  the ring blacked out under a dark maroon halo (on the 0.90-1.12r band)
# both bands are needed: bought's red is strongest inside the ring, while entity blacks out the ring
# itself, and the band OUTSIDE the ring is map art (a dim node on a dark floor false-reads entity).
# ---------------------------------------------------------------------------

STATE_AVAILABLE = "available"
STATE_BOUGHT = "bought"
STATE_ENTITY = "entity"

# bought: a BRIGHT red ring on the 0.74-0.90r band. dead clean, 0.45 vs 0.09 over every web measured.
BOUGHT_HOT_MIN = 0.22

# entity: black socket OR maroon halo. an OR, not an AND (2026-07-13 retune, 87 labeled entity nodes
# + 642 available): the halo is an ADDITIVE glow, so over the bright campfire at the bottom of the web
# it washes out completely while the ring is still plainly black. the AND was rejecting exactly those
# nodes and cost 16pp of recall for no precision gain.
# the black band is 0.82-1.00r, ENTIRELY inside the node: the entity darkens by alpha, so its soft
# outer edge stays bright over a bright floor and any band reaching past the rim reads background, not
# node. thresholds sit just above the worst available node (black 0.49, gore 0.42).
ENTITY_BLACK_MIN = 0.50   # frac of the 0.82-1.00r socket under ENTITY_BLACK_V
ENTITY_BLACK_V = 38
ENTITY_GORE_MIN = 0.45    # frac of the 0.90-1.12r ring that is dark maroon
# the smoke PULSES and half-renders, so a single frame still misses ~12% of eaten nodes. that is not
# fixed here but in spender.refresh_states, which LATCHES: consumption is monotone, so a miss now is
# caught by the next buy's re-read (per-frame 88% -> 100% cumulative on the 8-frame live capture).

# draw_detections only. the ring color is the only thing still legible at the debug view's fit zoom
# (a 3440px frame scaled to ~26%), so it carries the state; the banner spells out WHO grabbed the node
# once you zoom in. 'us' covers both our own click and the auto-path buys it dragged along with it.
STATE_COLORS = {
    STATE_AVAILABLE: (0, 255, 0),      # green
    STATE_BOUGHT: (0, 0, 255),         # red
    STATE_ENTITY: (255, 0, 255),       # magenta
}
STATE_LABELS = {
    STATE_BOUGHT: "grabbed: us",
    STATE_ENTITY: "grabbed: entity",
}


def _annulus_px(hsv, x, y, r, lo, hi, min_px=20):
    """hsv pixels in the annulus [lo*r, hi*r] around (x, y), or None if too few (node at a frame edge).
    crops a local box before masking so a state read is O(r^2) not O(frame): this runs on every node
    after every buy, unlike sample_disk_hsv which only runs once per scan."""
    x, y = int(round(x)), int(round(y))
    R = int(np.ceil(hi * r)) + 1
    h, w = hsv.shape[:2]
    x0, x1 = max(x - R, 0), min(x + R + 1, w)
    y0, y1 = max(y - R, 0), min(y + R + 1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    px = hsv[y0:y1, x0:x1][(d2 >= (lo * r) ** 2) & (d2 <= (hi * r) ** 2)]
    return px if len(px) >= min_px else None


def read_node_state(hsv, x, y, r):
    """'available' | 'bought' | 'entity' for one node, read off its socket ring (see the block above).
    r is the LATTICE socket radius (Resolution.NODE_RADIUS * scale), not isolate's r_ref: a taken
    node's disk color is meaningless so its r_ref can't be trusted (and isolate often fails outright).
    an unreadable ring returns 'available', i.e. the pre-state behavior of treating every node as buyable."""
    inner = _annulus_px(hsv, x, y, r, 0.74, 0.90)
    socket = _annulus_px(hsv, x, y, r, 0.82, 1.00)
    ring = _annulus_px(hsv, x, y, r, 0.90, 1.12)
    if inner is None or socket is None or ring is None:
        return STATE_AVAILABLE

    h, s, v = inner[:, 0].astype(int), inner[:, 1].astype(int), inner[:, 2].astype(int)
    seam = np.minimum(h, 180 - h) <= 8   # red hue, wrap-safe; tight enough that ultra-rare's pink (h=171) stays out
    if float((seam & (s >= 110) & (v >= 80)).mean()) >= BOUGHT_HOT_MIN:
        return STATE_BOUGHT

    black = float((socket[:, 2] <= ENTITY_BLACK_V).mean())                        # blacked-out socket
    h, s, v = ring[:, 0].astype(int), ring[:, 1].astype(int), ring[:, 2].astype(int)
    gore = float(((np.minimum(h, 180 - h) <= 8) & (s >= 60) & (v < 80)).mean())   # dark maroon halo
    if black >= ENTITY_BLACK_MIN or gore >= ENTITY_GORE_MIN:
        return STATE_ENTITY
    return STATE_AVAILABLE


def read_node_states(frame, xyr):
    """read_node_state for a whole web off one bgr frame: xyr is [(x, y, ring_r)], returns [state].
    one frame convert for the lot, so the per-node cost is just its ring read (see spender.refresh_states)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return [read_node_state(hsv, x, y, r) for x, y, r in xyr]


def sample_disk_hsv(hsv, x, y, r, s_floor=30, v_floor=20, min_px=6):
    """median hsv over an annulus around the disk, away from the glyph.
    annulus not full disk so the center glyph and rim anti-aliasing don't pollute the rarity read."""
    h, w = hsv.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - x) ** 2 + (yy - y) ** 2 # squared dist of every px from center
    ring = (d2 >= (0.2 * r) ** 2) & (d2 <= (0.8*r) ** 2) # restrict to this node's ring
    
    s, v = hsv[..., 1], hsv[..., 2]
    keep = ring & (s >= s_floor) & (v >= v_floor) # drop achromatic px (low s = white/gray, low v = black)
    px = hsv[keep]
    if len(px) <= min_px:
        return None
    # hue is circular (0..179 wraps at the red seam): anchor on the circular mean,
    # then take the median of the wrap-safe offsets from it.
    hue = px[:, 0].astype(np.float32)
    mu = hue_circular_mean(hue)
    h_med = (mu + np.median(hue_circular_delta(hue, mu))) % 180
    return np.array([h_med, np.median(px[:, 1]), np.median(px[:, 2])])


def find_nodes_in_frame(
        frame, debug=False, max_anchor_dist=None, thresh_method='adaptive_gaussian',
        use_hough=False, resolution=None, center=None, use_lattice=True
    ):
    """find all circles in the blood web and clean them up to identify clickable nodes.
    returns [(x, y, r, rarity, slot, state, ring_r)]: slot/ring_r are the lattice index and socket radius (None without a lattice fit), state is read_node_state's available|bought|entity.
    thresh_method picks find_circles' binarization (adaptive_gaussian|otsu|canny), use_hough swaps the localizer between the contour pass (default) and HoughCircles, both threaded from detect() so the settings ui can tune them.
    resolution (a Resolution, default Resolution.from_frame(frame)) scales rmin/rmax to this frame's size.
    use_lattice snaps circles onto the fixed slot grid (dropping off-slot junk) and runs the presence test on empty slots to recover contour misses.
    center is find_center_node's (x, y, r) when the caller already computed it (detect() does), else found here."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # (H,W,3), for the per-circle color read
    circles = find_circles(
        frame, thresh_method=thresh_method, use_hough=use_hough,
        debug=False, resolution=resolution
    ) # list of (x, y, r)
    circles = [(x, y, r, None) for x, y, r in circles]   # (x, y, r, slot); slot filled in by the lattice snap
    ring_r = None   # lattice socket radius, the geometry read_node_state's annuli are fractions of

    if use_lattice and circles:
        if center is None:
            center = find_center_node(frame)
        resolution = resolution or Resolution.from_frame(frame)
        fit = fit_lattice([(x, y, r) for x, y, r, _ in circles], center=center,
                          scale_hint=resolution.scale)
        if fit is None:
            if debug:
                print(f"[lattice] fit skipped ({len(circles)} circles, "
                      f"center {'found' if center else 'missing'}), keeping raw detections")
        else:
            cx, cy, s = fit
            ring_r = Resolution.NODE_RADIUS * s
            snapped, taken, dropped, tpl_pts = snap_to_lattice(
                [(x, y, r) for x, y, r, _ in circles], cx, cy, s)
            if debug:
                print(f"[lattice] center=({cx:.0f},{cy:.0f}) scale={s:.3f}: "
                      f"{len(snapped)} snapped, {len(dropped)} junk dropped")
                for x, y, r, d in dropped:
                    print(f"[lattice]   dropped ({x:.0f},{y:.0f}) r={r:.0f}, {d:.0f}px off-slot")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            grad = _grad_mag(gray)
            update_ring_template(gray, grad, tpl_pts, s)
            circles = [(x, y, r, i) for x, y, r, i in snapped]
            if len(snapped) >= 3:   # a bogus fit snaps almost nothing, don't recover off one
                slots, _ = lattice_slots(cx, cy, s)
                empty = [(i, p[0], p[1]) for i, p in enumerate(slots) if i not in taken]
                for i, x, y, score in recover_missed_slots(gray, grad, empty, s, debug=debug):
                    if debug:
                        print(f"[lattice] recovered miss at ({x},{y}) presence={score:.2f}")
                    circles.append((x, y, ring_r, i))

    nodes = []
    for x, y, r, slot in circles:
        # state first: a taken node's disk color is meaningless, so its rarity read (and isolate, and
        # the matcher) are all garbage. keep it anyway, unrarity'd, so the loop knows the slot is dead
        # and the entity-race tiebreak can measure distance to it.
        state = read_node_state(hsv, x, y, ring_r) if ring_r else STATE_AVAILABLE
        disk_hsv = sample_disk_hsv(hsv, x, y, r)
        rarity, dist = classify_rarity(disk_hsv)
        if state == STATE_AVAILABLE:
            if rarity is None:
                continue # remove circle from list if a rarity wasn't obtained
            if max_anchor_dist is not None and dist > max_anchor_dist:
                continue                           # not near any rarity -> probably not a node
        # runners-up ride along for isolate_node_glyph's retry; appended so index unpackers stay valid.
        nodes.append((int(x), int(y), int(r), rarity, slot, state, ring_r,
                      rarity_candidates(disk_hsv)))

    if debug:
        _show(draw_detections(frame, nodes), "find_nodes")
    return nodes


def find_center_node(
        frame, h_tol=8, s_floor=200, v_lo=22, v_hi=95,
        close_ksize=7, min_area_frac=0.0015, roi_frac=0.30
    ):
    """locate the center auto-spend node (dark red entity hexagon) by its glow color.
    find_circles misses it on ~1/4 of frames and it has no rarity anchor, so detect it off its stable signature instead: hue at the 0/180 seam, high saturation, low (dark) value.
    returns the largest such blob near the frame center as (x, y, r), or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = np.abs(hue_circular_delta(h, 0)) <= h_tol # wrap-safe red seam band
    mask = (red & (s >= s_floor) & (v >= v_lo) & (v <= v_hi)).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel) # bridge the glow ring

    h_img, w_img = mask.shape
    frame_center = np.array([w_img / 2.0, h_img / 2.0])
    roi = roi_frac * min(h_img, w_img)
    min_area = min_area_frac * h_img * w_img
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    best, best_area = None, None
    for i in range(1, n): # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if np.linalg.norm(centroids[i] - frame_center) > roi:
            continue # off-center red (banner, ui) is not the node
        if best_area is None or area > best_area:
            best, best_area = i, area
    if best is None:
        return None

    cx, cy = centroids[best]
    bw, bh = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    return int(round(cx)), int(round(cy)), int(round(max(bw, bh) / 2))


def isolate_node_contents(
        frame, x_hat, y_hat, r_hat, rarity,
        r_tol=1.5, roi_k=1.2, h_tol=8, s_floor=25, v_floor=22,
        close_ksize=5, min_area_frac=0.05
    ):
    """crop a coarse node, color-mask its rarity socket, and pick the central blob.
    the coarse (x,y,r) from find_circles can be off-center/undersized, so crop wide (r_tol) and recenter on the socket centroid; the roi (roi_k*r around the coarse center) keeps hue bleed (bronze ring / dim web) from inflating the socket blob.

    returns (cx, cy, r, contour, crop) or None:
        cx, cy, r: full-frame click center + socket radius
        contour: crop-local socket polygon (feed classify_socket + normalize_glyph)
        crop: the bgr crop those two read from"""
    crop, rel_bbox = _crop_glyph_from_frame(frame, x_hat, y_hat, r_hat, r_tol=r_tol)
    x0, y0 = rel_bbox['x0'], rel_bbox['y0']

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ref = get_ref_hsvs()[rarity][0] # hue anchor for this node's rarity
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    # event/gold shares the bronze ring's hue but the gold disk is far more saturated (S~190+),
    # so lift the saturation floor for it to keep the disk and drop the ring; other tiers keep s_floor
    s_floor_eff = 150 if rarity == "event" else s_floor
    mask = hue_band_mask(hsv, h_ref, h_tol, s_floor_eff, v_floor)

    # cap the mask to a circle around the coarse center so hue bleed can't grow the socket past the node
    # (the bronze ring shares the brown/common hue, the dark web is dim-brown)
    roi = np.zeros_like(mask)
    cv2.circle(roi, (int(x_hat - x0), int(y_hat - y0)), int(roi_k * r_hat), 255, -1)
    mask = cv2.bitwise_and(mask, roi)

    mask = _fill_holes(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel))

    # biggest-enough contour whose centroid sits nearest the crop center
    ch, cw = mask.shape
    center = np.array([cw / 2.0, ch / 2.0])
    best, best_d, best_ctr = None, None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area_frac * ch * cw: # skip glyph specks / ring bleed
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        ctr = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        d = np.linalg.norm(ctr - center)
        if best_d is None or d < best_d:
            best, best_d, best_ctr = c, d, ctr
    if best is None:
        return None

    cx, cy = best_ctr[0] + x0, best_ctr[1] + y0 # centroid -> full-frame
    (_, _), r_ref = cv2.minEnclosingCircle(best) # radius from the socket, not the guess
    return int(round(cx)), int(round(cy)), int(round(r_ref)), best, crop


def normalize_glyph(crop, contour, rarity=None, erode_ksize=3, out_size=GLYPH_SIZE, keep_k=0.65):
    """strip the colored socket from a node crop so only the glyph remains on black, framed to match scraper.normalize_sprite (tight-crop -> square-pad centered -> resize).

    glyph = the BRIGHT pixels inside the socket polygon; the white-ish glyph always sits brighter than the rarity disk fill, so an otsu cut on the value channel keys the fill out without per-rarity hue math (the old hue-subtraction left colored halos and broke on gold/event).
    the contour is convex-hulled first so glyph strokes touching the socket edge don't carve notches out of the interior mask.
    rarity=='event' triggers a clahe contrast boost on the value channel before the otsu cut (the gold disk is too bright for plain otsu to split the white glyph from the fill).
    keep_k prunes mask components that never come within keep_k*r_in of the plate center before the tight bbox: the add-on '+' marker survives otsu at the plate periphery and stretched the bbox, shoving faint art off-center (the focusLens->saboteur drift family, real top1 91.4->94.8%).
    returns a GLYPH_SIZE bgr glyph-on-black square, or None if nothing survives."""
    h, w = crop.shape[:2]
    val = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]   # value channel (brightness), (h,w)

    # event/gold disks are bright enough that plain otsu can't split the white glyph from the gold fill, so the glyph blows out into gold speckle and mis-matches.
    # clahe locally lifts the very bright glyph clear of the bright-but-less fill so the otsu cut lands cleanly.
    # only event needs it, on the darker tiers clahe just amplifies the fill texture and hurts (verified on the 211 real labeled nodes).
    if rarity == "event":
        val = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(val)

    # interior = filled convex socket polygon, eroded a touch to shed the colored rim/anti-alias
    hull = cv2.convexHull(contour)
    inside = np.zeros((h, w), np.uint8)
    cv2.drawContours(inside, [hull], -1, 255, cv2.FILLED)
    if erode_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
        inside = cv2.erode(inside, k)

    # otsu-threshold the value channel within the socket -> splits the bright glyph from the
    # darker colored fill. compute the threshold off the inside pixels only so the dark web
    # outside the socket doesn't drag it.
    vin = val[inside > 0]
    if vin.size == 0:
        return None
    thr, _ = cv2.threshold(vin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    glyph_mask = (inside > 0) & (val >= thr)

    if keep_k:
        # drop bright peripheral junk before the bbox: the '+' marker / corner shine sits at
        # min-dist 0.83-0.98 r_in on the labeled crops while real art, composited centered, always
        # reaches the middle -- 0.65 splits the two populations with margin on both sides.
        # nearest-pixel test (not centroid) so a big ring that spans center to rim is kept whole.
        n_lab, lab = cv2.connectedComponents(glyph_mask.astype(np.uint8))
        hm = cv2.moments(hull)
        if n_lab > 2 and hm["m00"]:
            hx, hy = hm["m10"] / hm["m00"], hm["m01"] / hm["m00"]
            r_in = np.sqrt(cv2.contourArea(hull) / np.pi)
            keep = np.zeros_like(glyph_mask)
            for l in range(1, n_lab):
                comp = lab == l
                cys, cxs = np.nonzero(comp)
                if ((cxs - hx) ** 2 + (cys - hy) ** 2).min() <= (keep_k * r_in) ** 2:
                    keep |= comp
            if keep.any():               # never let the prune empty the glyph outright
                glyph_mask = keep

    ys, xs = np.where(glyph_mask)
    if len(xs) == 0:
        return None

    glyph = np.zeros_like(crop)
    glyph[glyph_mask] = crop[glyph_mask] # keep the bgr glyph pixels, rest stays black
    glyph = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1] # tight glyph bbox

    gh, gw = glyph.shape[:2]
    side = max(gh, gw)
    canvas = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - gh) // 2, (side - gw) // 2
    canvas[y0:y0 + gh, x0:x0 + gw] = glyph

    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)


def isolate_node_glyph(frame, x, y, r, rarities, r_tol=1.5):
    """isolate a node's socket and normalize its glyph, walking the rarity hypotheses in order until
    one lands. returns (rarity, iso, glyph), all None when none of them isolate.
    retry exists because isolate only masks a +-h_tol hue band around one anchor, so a bad first guess
    (e.g. a glyph-dominant plate washing out the disk read) drops the node outright otherwise.
    a hypothesis that isolates but disagrees with the matched icon is reconciled downstream via ocr
    hover, so a wrong guess costs a hover, not a mis-buy."""
    for rar in rarities:
        iso = isolate_node_contents(frame, x, y, r, rar, r_tol=r_tol)
        if iso is None:
            continue
        glyph = normalize_glyph(iso[4], iso[3], rar)
        if glyph is not None:
            return rar, iso, glyph
    return None, None, None


def classify_socket(node_contours, poly_tol=0.05):
    """classify node contents by socket shape: offerings -> hexagon, perks -> rhombus, items -> square, addons -> square with a plus.
    we detect the socket contour's shape and map it to the glyph type.
    """
    hull = cv2.convexHull(node_contours) # drop misc inner glyph noise
    _, _, bw, bh = cv2.boundingRect(hull)
    coverage = cv2.contourArea(hull) / (bw * bh)

    if coverage <= 0.64:
        return 'rhombus'
    if 0.64 < coverage <= 0.8:
        return 'hexagon'
    if 0.8 < coverage:
        return 'square'


def detect(
        frame, rows=None, hashes=None, ncc_templates=None, matcher="cnn", r_tol=1.5,
        debug=False, thresh_method="adaptive_gaussian", use_hough=False, resolution=None,
        row_pool=None, use_lattice=True,
    ):
    """full pipeline; returns per-node dicts {x, y, r, rar, cat, glyph_bgr, match, score, margin, matcher}.
    matcher picks identification: 'cnn' (default, learned embedding cosine, higher=better, falls back to ncc if the model is absent), 'ncc' (plain z-normed cosine, higher=better), 'ncc_masked' (cosine over the query's bright strokes, higher=better), or 'phash' (hamming, lower=better); score's direction follows the matcher (see _score_str).
    thresh_method picks the node-localization binarization (adaptive_gaussian|otsu|canny) and use_hough swaps the localizer (contour pass vs HoughCircles), both passed down to find_circles.
    resolution (a Resolution, default Resolution.from_frame(frame)) scales find_circles' rmin/rmax to this frame's size.
    row_pool (a bool sequence aligned to rows, from node.build_pool_mask) optionally narrows the match library to the priority list's icons/sources; a node whose socket shape has no candidate left in that pool is emitted pooled_out=True (unknown, skipped, never matched or ocr'd).
    use_lattice turns the slot-lattice snap + presence recovery on (see find_nodes_in_frame)."""
    if rows is None:
        rows, hashes = load_index()
    # resolve the learned matcher first so a missing model degrades to ncc before the rest is set up
    cnn_net = cnn_bank = None
    if matcher == "cnn":
        cnn_net = load_cnn_model()
        if cnn_net is None:
            if debug:
                print("cnn model not found, falling back to ncc")
            matcher = "ncc"
        else:
            cnn_bank = load_cnn_bank(rows)
    if matcher == "phash" and hashes is None:
        _, hashes = load_index()
    if matcher in ("ncc", "ncc_masked") and ncc_templates is None:
        ncc_templates = load_ncc_templates(rows)
    ncc_plain_T = ncc_plain_templates(ncc_templates) if matcher == "ncc" else None

    # exclude rows that can never be a bloodweb node (see is_matchable).
    # socket shape is no longer a candidate pool but an agreement check reconciled downstream (Node.category_agrees -> ocr fallback),
    # so every node searches the full matchable library. (n,) bool aligned to rows.
    matchable = np.array([is_matchable(r) for r in rows])
    # optional priority pool: intersect it in so matching only scores the icons/sources we care about.
    # cats is only needed for the per-node pooled-out test below, so build it lazily here.
    if row_pool is not None:
        matchable = matchable & np.asarray(row_pool, dtype=bool)
        cats = np.array([r.get('category') for r in rows])   # (n,) 'item'|'addon'|...
    # the center auto-spend node is found by its own glow color (not the disk pipeline) and gets no glyph match;
    # tagged kind='center' so Node/spender can reference it (see find_center_node).
    # found BEFORE the node pass so the lattice fit can anchor on it instead of re-deriving it.
    center = find_center_node(frame)
    nodes = find_nodes_in_frame(
        frame, debug=debug, thresh_method=thresh_method,
        use_hough=use_hough, resolution=resolution,
        center=center, use_lattice=use_lattice,
    ) # [(x, y, r, rarity), ...]

    res = []
    if center is not None:
        cx, cy, r = center
        res.append({
            'x': cx, 'y': cy, 'r': r,
            'rar': None, 'cat': None, 'kind': 'center',
            'glyph_bgr': None,
            'match': None, 'score': 0.0, 'margin': 0.0, 'matcher': matcher,
        })

    for node in nodes:
        x, y, r, rarity = int(node[0]), int(node[1]), int(node[2]), node[3]
        slot, state, ring_r = node[4], node[5], node[6]
        if center is not None and (x - center[0]) ** 2 + (y - center[1]) ** 2 <= center[2] ** 2:
            continue # this circle is the center node, already emitted above

        # (x, y) is still the lattice slot center here, which is what read_node_state's annuli were
        # calibrated on, and unlike the plate centroid isolate is about to compute it doesn't move when
        # a node's look changes. keep it: a state RE-read (spender.refresh_states) must sample the same
        # spot, and off the centroid the entity ring read lands a few px out and misses (slots 18/29).
        slot_xy = (x, y)

        # already bought (by us, or by dbd auto-pathing through it) or eaten by the entity: it can
        # never be a buy target, so skip isolate + the matcher + the ocr hover it would have routed to.
        if state != STATE_AVAILABLE:
            res.append({
                'x': x, 'y': y, 'r': int(ring_r or r), 'rar': rarity, 'cat': None,
                'slot': slot, 'state': state, 'ring_r': ring_r, 'slot_xy': slot_xy,
                'glyph_bgr': None,
                'match': None, 'score': 0.0, 'margin': 0.0, 'matcher': matcher, 'runner_up': None,
            })
            continue

        # runner-up rarities only retried when slotted (a lattice slot proves the node is real, so an
        # isolate miss there is just a bad rarity guess; off-lattice it resurrects junk instead).
        cands = node[7] if slot is not None else None
        rarity, iso, glyph = isolate_node_glyph(frame, x, y, r, cands or [rarity], r_tol=r_tol)
        if iso is None:
            if debug: print("WARNING: detect() - Node skipped, no rarity hypothesis could isolate a glyph")
            continue

        cx, cy, r_ref, node_contours, crop = iso
        socket_shape = classify_socket(node_contours) # use socket shape geometry to classify node type (item/addon, perk, offering)

        # with a priority pool active, a node whose socket shape has no candidate left in the pool is outside our scope (e.g. a perk node on a survivor run that only lists items);
        # emit it unknown so the loop skips it, no match scored and no ocr hover (see Node.pooled_out).
        # the matcher itself still searches the whole pool so shape stays a downstream agreement check.
        if row_pool is not None:
            allowed = NODE_SHAPE_DICT.get(socket_shape)
            shape_in_pool = matchable if allowed is None else (matchable & np.isin(cats, allowed))
            if not shape_in_pool.any():
                if debug: print(f"pooled out {socket_shape} node @ {cx},{cy}")
                res.append({
                    'x': cx, 'y': cy, 'r': r_ref, 'rar': rarity, 'cat': socket_shape,
                    'slot': slot, 'state': state, 'ring_r': ring_r, 'slot_xy': slot_xy,
                    'glyph_bgr': glyph, 'pooled_out': True,
                    'match': None, 'score': 0.0, 'margin': 0.0, 'matcher': matcher, 'runner_up': None,
                })
                continue

        # identify the glyph against the full matchable library (unavailable excluded);
        # socket shape is cross-checked downstream, not used to prune candidates here.
        runner_sim = None                                          # cnn-only, for the near-dup veto
        if matcher == "cnn":
            best_match_row, score, margin, runner, runner_sim = id_icon_cnn(glyph, rows, cnn_bank, cnn_net, pool=matchable)
        elif matcher == "ncc":
            best_match_row, score, margin, runner = id_icon_ncc(glyph, rows, ncc_plain_T, pool=matchable)
        elif matcher == "ncc_masked":
            best_match_row, score, margin, runner = id_icon_ncc_masked(glyph, rows, ncc_templates, pool=matchable)
        else:
            best_match_row, score, margin, runner = id_icon_hamming(glyph, rows, hashes, pool=matchable)
        if debug:
            print(matcher, best_match_row['key'], round(score, 3), round(margin, 3),
                  "vs", runner['key'])

        # observed attrs vs matched icon attrs are reconciled in spender (see node.needs_resolution)
        res.append({
            'x': cx, 'y': cy, 'r': r_ref,
            'rar': rarity,
            'cat': socket_shape,
            'slot': slot, 'state': state, 'ring_r': ring_r, 'slot_xy': slot_xy,
            'glyph_bgr': glyph,
            'match': best_match_row, 'score': score, 'margin': margin, 'matcher': matcher,
            'runner_up': runner.get('name') or runner.get('key'),  # 2nd-best, debug-only
            'runner_up_sim': runner_sim,  # top1<->top2 anchor cos, cnn near-dup veto (None otherwise)
        })
    return res


# debugging sample disk colors and visualize detections on fixtures.
def _score_str(n):
    """compact match-score label: 'd<ham>' for phash (lower=better), 's<cosine>' for ncc
    (higher=better). reads the generic 'score'/'matcher' keys off a detect() result dict."""
    s = n.get("score")
    if s is None:
        return "?"
    return f"d{int(s)}" if n.get("matcher", "phash") == "phash" else f"s{s:.2f}"


def draw_detections(frame, nodes):
    """draw each node (circle + label) onto a copy of the frame. accepts detect() result dicts,
    the find_nodes_in_frame tuples, or spender's Node objects, so it works at any stage of the pipeline.
    a grabbed node (bought / entity) keeps its identity lines and gains a 'grabbed: by-who' banner on
    top, colored + ringed by state: a node grabbed MID-WEB was identified back when it was still
    available, and that read is exactly what says whether the thing we just lost was one we wanted.
    (a node already grabbed at the first scan never got identified, since detect skips the matcher on
    it, so it has nothing under the banner to show.)"""
    out = frame.copy()
    for n in nodes:
        state = STATE_AVAILABLE
        if isinstance(n, dict):
            x, y, r = n["x"], n["y"], n["r"]
            state = n.get("state", STATE_AVAILABLE)
            match = n.get("match")
            if n.get("kind") == "center":
                lines = ["autospend"]
            else:
                lines = [STATE_LABELS[state]] if state != STATE_AVAILABLE else []
                if match or state == STATE_AVAILABLE:
                    m = match or {}   # match is a library row dict (or None)
                    # name on its own line so long icon names stay legible over the busy web
                    lines += [f"{n.get('rar', '?')}/{n.get('cat', '?')} {_score_str(n)}",
                              m.get("name") or m.get("key") or "?"]
        elif isinstance(n, tuple):
            x, y, r, rarity = n[:4]              # find_nodes_in_frame tuples carry slot/state/ring_r too
            state = n[5] if len(n) > 5 else STATE_AVAILABLE
            lines = ([STATE_LABELS[state]] if state != STATE_AVAILABLE else []) + [str(rarity)]
        else:
            # spender.Node: the boundary object the live loop resolves nodes into, post-ocr
            x, y, r = n.x, n.y, n.r
            state = n.state
            if n.is_center:
                lines = ["autospend"]
            else:
                lines = [STATE_LABELS[n.state]] if n.taken else []
                if not n.taken or n.matched_name or n.name:
                    # matched_name/score are the matcher's own read, frozen before ocr can overwrite
                    # node.name; ocr read is only shown when it actually settled the node (resolved_by).
                    ocr_read = n.name if n.resolved_by == "ocr" else "na"
                    lines += [
                        f"{n.rarity}/{n.socket_shape}",
                        f"{n.matcher}: {n.matched_name or '?'} {n.score:.2f}",
                        f"ocr: {ocr_read}",
                    ]
        col = STATE_COLORS.get(state, (0, 255, 0))   # taken nodes stand out at a glance in the debug view
        cv2.circle(out, (x, y), r, col, 2)
        # stack the lines just above the circle, last line nearest the rim
        for i, line in enumerate(lines):
            org = (x - r, y - r - 6 - (len(lines) - 1 - i) * 14)
            # black underlay then the state color so the label stays readable over the busy web background
            cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


# matplotlib, not cv2 highgui: the conda opencv build ships without a gui backend, and we
# don't want highgui in the frozen exe anyway. matplotlib also gives zoom/pan for free and
# maps clicks back to image coords even when zoomed, which the 3440x1440 frames need.
def _plt():
    # lazy import, debug draw only, so import detect and the frozen exe skip matplotlib
    import matplotlib.pyplot as plt
    return plt


def _sample_window(fixture_path):
    """open a fixture and print the hsv under each click, how we read real disk colors to set/verify the rarity anchors.
    median over a small patch so one noisy pixel doesn't mislead."""
    plt = _plt()
    img = cv2.imread(str(fixture_path))
    if img is None:
        raise FileNotFoundError(fixture_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]

    def on_click(event):
        if event.xdata is None:  # click landed outside the axes
            return
        x = min(max(int(event.xdata), 0), w_img - 1)
        y = min(max(int(event.ydata), 0), h_img - 1)
        patch = hsv[max(0, y - 3):y + 4, max(0, x - 3):x + 4].reshape(-1, 3)  # (<=49, 3)
        h, s, v = np.median(patch, axis=0).astype(int)
        print(f"({x},{y}) hsv=[{h},{s},{v}] bgr={img[y, x].tolist()}")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # matplotlib wants rgb
    ax.set_title("click disks to read hsv (toolbar zoom/pan); close window to quit")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def _show(
        img, title="detect", savefig=False,
        contours=None, edges=None, circles=None,
        contour_color=(0, 0, 255), edge_color=(0, 255, 0), circle_color=(255, 0, 0),
    ):
    """show an image in a matplotlib window, with optional contour/edge/circle overlays for find_circles debugging.
    this conda opencv build ships no highgui backend, so cv2.imshow throws 'function not implemented' (same reason _sample_window uses matplotlib).

    img is a bgr frame OR a single-channel gray/edge/mask (ndim==2, promoted to bgr so overlays can be colored).
    contours is a list of cv2 contours (drawn), edges a binary single-channel map (nonzero pixels painted on), circles a list of (x, y, r) like find_circles returns (drawn as outline + center dot, floats cast to int).
    overlay colors are bgr, composited with cv2 onto a copy then handed to imshow."""
    plt = _plt()
    # matplotlib clips float rgb to [0,1], so a scalar float map (e.g. the distance transform) sent through gray2bgr shows up solid white;
    # normalize any non-uint8 2d input to 0-255 first so its gradient is visible (binary uint8 masks pass through as-is)
    if img.ndim == 2 and img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if edges is not None:
        vis[edges > 0] = edge_color # binary-edge pixels -> overlay color

    if contours is not None:
        cv2.drawContours(vis, contours, -1, contour_color, 1)

    if circles is not None:
        for x, y, r in circles:
            c = (int(round(x)), int(round(y)))
            cv2.circle(vis, c, int(round(r)), circle_color, 2)  # the found disk
            cv2.circle(vis, c, 2, circle_color, -1) # center dot

    fig = plt.figure()
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)) # matplotlib wants rgb
    plt.title(title)

    if savefig:
        paths.debug_dir().mkdir(parents=True, exist_ok=True)
        plt.savefig(str(paths.debug_dir() / f"{title}.png"), dpi=200)
        plt.close(fig)   # close it, else a later plt.show() from another _show pops this up too
    else:
        plt.show()


def _show_gallery(items, title="glyphs", cols=6, savefig=False):
    """tile a set of (image, caption) pairs in a grid, each captioned.
    for eyeballing the per-node crops/normalized glyphs next to what they read as (rarity, socket, match), instead of the whole annotated frame.
    images are bgr or single-channel like everywhere else here; a None image draws a blank cell (e.g. normalize_glyph returned None).
    dev-only, matplotlib since this conda cv2 has no highgui backend."""
    plt = _plt()
    items = [it for it in items]
    if not items:
        print("gallery: nothing to show")
        return
    cols = min(cols, len(items))
    rows = (len(items) + cols - 1) // cols
    # squeeze=False so axes is always 2d, then flatten -> uniform handling for any grid size
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(cols * 2.0, rows * 2.3))
    axes = axes.ravel()
    for ax, (img, cap) in zip(axes, items):
        if img is not None and img.size:
            rgb = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2
                   else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.imshow(rgb)
        else:
            ax.imshow(np.zeros((8, 8, 3), np.uint8)) # blank cell for a missing glyph
        ax.set_title(cap, fontsize=7)
        ax.axis("off")
    for ax in axes[len(items):]: # blank the unused tail cells
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if savefig:
        paths.debug_dir().mkdir(parents=True, exist_ok=True)
        fig.savefig(str(paths.debug_dir() / f"{title}.png"), dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="detect debug cockpit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_sample = sub.add_parser("sample", help="click a fixture to read disk hsv (tune anchors)")
    s_sample.add_argument("fixture", type=Path)
    s_detect = sub.add_parser("detect", help="run detect on a fixture and show the result")
    s_detect.add_argument("fixture", type=Path)
    s_detect.add_argument("--save", type=Path, default=None)
    s_detect.add_argument("--matcher", choices=MATCHERS, default="ncc", help="icon matcher (default ncc)")
    s_glyphs = sub.add_parser("glyphs", help="gallery of per-node crops + normalized glyphs")
    s_glyphs.add_argument("fixture", type=Path)
    s_glyphs.add_argument(
        "--match", action="store_true",
        help="also run id_icon per node (needs the re-hashed index to mean anything)"
    )
    s_glyphs.add_argument("--matcher", choices=MATCHERS, default="ncc", help="icon matcher (default ncc)")
    s_glyphs.add_argument("--save", action="store_true", help="save the gallery to .tmp/ instead of showing")
    args = ap.parse_args()

    # TODO: replace with auto bbox (detect-then-bound on the node cluster, no hardcoded coords)
    # web_bbox_fallback_px is Resolution's WEB_BBOX_FALLBACK sized to the loaded fixture, so this
    # stays correct even on a fixture that isn't the 3440x1440 baseline.
    if args.cmd == "sample":
        _sample_window(args.fixture)
    elif args.cmd == "detect":
        frame = cv2.imread(str(args.fixture))
        web_bbox = Resolution.from_frame(frame).web_bbox_fallback_px()
        frame = frame[web_bbox['y0']:web_bbox['yf'], web_bbox['x0']:web_bbox['xf']]
        nodes = detect(frame, matcher=args.matcher, debug=True)
        viz = draw_detections(frame, nodes)
        if args.save:
            cv2.imwrite(str(args.save), viz)
        _show(viz, "detect")

    elif args.cmd == "glyphs":
        # captioned with output of detections (rarity/socket-shape, matched name, hamming dist + margin).
        # shows the per-node glyph/shape/match quality
        frame = cv2.imread(str(args.fixture))
        web_bbox = Resolution.from_frame(frame).web_bbox_fallback_px()
        frame = frame[web_bbox['y0']:web_bbox['yf'], web_bbox['x0']:web_bbox['xf']]
        results = detect(frame, matcher=args.matcher, debug=True)
        items = []
        for n in results:
            match = n.get("match") or {}
            name = match.get("name") or match.get("key") or "?"
            cap = f"{n['rar']}/{n['cat']}\n{name}\n{_score_str(n)} m{n['margin']:.2f}"
            items.append((n["glyph_bgr"], cap))
        _show_gallery(items, title=f"detections-{args.fixture.stem}", savefig=args.save)
