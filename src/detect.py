"""localize nodes, read rarity, identify the icon. the core offline-testable layer.

pipeline: hsv color-segment the rarity disks to get node centers plus a first read of
rarity, then crop + mask each inner icon and match it against the scraped library via
normalized cross-correlation (a perceptual-hash and a masked-ncc variant are also selectable).
every coord is found dynamically (no hardcoded positions) so it works for varying aspect ratios.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import imagehash
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
USR_HSV = ROOT / "usr" / "rarity-HSVs.json"         # active anchors, evolve per user each web
DEFAULT_INDEX = ROOT / "data" / "icons_index.json"

RARITIES = ["common", "uncommon", "rare", "very rare", "ultra rare", "event"]

# empirical rarity disk anchors, opencv hsv [h(0-179),s,v].
# used bloodweb screenshots from tests/fixtures/ with src.detect sample <web_path>
# to click nodes and print HSV vals
EMPIRICAL_SEED = {
    "common":     [11, 93, 51],     # brown
    "uncommon":   [61, 166, 72],    # green
    "rare":       [108, 141, 77],   # blue
    "very rare":  [143, 141, 77],   # purple
    "ultra rare": [171, 209, 117],  # pink/iri
    "event":      [21, 213, 172],   # gold/yellow
}

# hue carries the rarity signal so weight it most
# value is the most gamma/brightness sensitive so weight it least.
# that weighting is most of our gamma tolerance (since gamma is adjustable param in dbd graphics settings)
HSV_WEIGHTS = (4.0, 1.0, 0.3)

# normalized glyph canvas size. MUST match scraper.GLYPH_SIZE
GLYPH_SIZE = 128

# icon matcher selection, 'ncc' (plain z-normed cosine over the whole glyph) is the default
# 'phash' (perceptual-hash hamming) is the original matcher, kept for comparison
MATCHERS = ("ncc", "ncc_masked", "phash")
NCC_RES = 128  # res of the ncc template/query vectors

# rarity anchor colors: load (usr, auto-seeded from EMPIRICAL_SEED), classify,
# TODO: User calibration for hsv colors
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
        # backfill any anchor the usr file predates (e.g. the 'event' band added to the seed
        # after an earlier file was written) so a stale usr/ doesn't silently drop a tier
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
    """uint8 (h,w) mask of pixels within h_tol of h_ref on the hue wheel and above the s/v
    floors. wraps the 0/180 seam that cv2.inRange can't (via hue_circular_delta), so this is
    the one place band+inrange lives: refine_node isolates one rarity, disk_color_mask ORs it
    over all five. hsv is a (h,w,3) opencv-hsv image."""
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


def disk_color_mask(frame, htol=8, s_floor=62, v_floor=38):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # (H,W,3) uint8, h in 0..179
    anchors = get_ref_hsvs() # rarity: [h,s,v]
    frame_mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
    
    for rarity, (h, s, v) in anchors.items():
        frame_mask |= hue_band_mask(hsv, h, htol, s_floor, v_floor)  # wrap-safe, same util as refine_node

    return frame_mask


def refine_ref_hsvs(disk_samples, seed=None, out_path=USR_HSV, min_samples=3):
    """auto-update anchors from real disk colors. disk_samples is a list of (h,s,v) read off a frame (localize feeds these). each sample groups to its nearest seed anchor,
    then the anchor becomes the median of its group. median, not mean,
    so a stray glyph/glow pixel or a mid-animation disk doesn't drag it.
    a rarity with too few samples keeps the seed.
    writes {rarity:[h,s,v]} to usr/ and refreshes the cache.

    groups against the stable EMPIRICAL_SEED (a hue-band label) so the reference can't drift
    while the written values still track the user's monitor over webs."""
    seed = seed or {rar: list(hsv) for rar, hsv in EMPIRICAL_SEED.items()}
    groups = {rar: [] for rar in seed}
    for hsv in disk_samples:
        rar, _ = classify_rarity(hsv, seed)
        groups[rar].append(hsv)
    refined = {}
    for rar, ref in seed.items():
        samples = groups[rar]
        if len(samples) >= min_samples:
            refined[rar] = [int(c) for c in np.median(np.array(samples), axis=0)]  # (k,3)->(3,)
        else:
            refined[rar] = list(ref)  # not enough evidence yet, keep the seed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refined, indent=2), encoding="utf-8")
    global _HSVs
    _HSVs = refined
    return refined


def load_index(index_path=DEFAULT_INDEX):
    """returns (rows, hashes). rows = metadata dicts; hashes = (n, 64) bool array of the
    precomputed phashes, stacked to query with a single vector op."""
    rows = json.loads(Path(index_path).read_text(encoding="utf-8"))
    hashes = np.stack([
        imagehash.hex_to_hash(r["phash"]).hash.flatten() for r in rows
    ])  # (n, 64)
    return rows, hashes


def id_icon_hamming(icon_bgr, rows, ref_hashes, pool=None):
    """nearest neighbor of an unknown icon via hamming distance between its phash and the library phashes.
    pool is an optional bool mask (n,) to restrict the search to a category,
    once we read the node's socket shape, giving a smaller cleaner candidate set.
    returns (row, dist, margin): best row, its hamming distance (0..64, lower is better),
    and the gap to the 2nd best (a small gap means an ambiguous match)."""
    gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY) # same luma weights as PIL 'L'
    q = imagehash.phash(Image.fromarray(gray)).hash.flatten() # (64,) bool
    dists = np.count_nonzero(ref_hashes != q, axis=1) # (n,) hamming: differing bits
    if pool is not None:
        dists = np.where(pool, dists, 65) # 65 > max dist of 64
    order = np.argsort(dists)
    return rows[order[0]], int(dists[order[0]]), int(dists[order[1]] - dists[order[0]])


def _sprite_glyph_gray(path, out_size=GLYPH_SIZE):
    """load a library sprite and frame it exactly like scraper.normalize_sprite
    (alpha tight-crop -> square-pad on black -> resize), returning a grayscale glyph (h,w) uint8
    the ncc templates are built from these so a query glyph (normalize_glyph output) compares effectively"""
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


def load_ncc_templates(rows, index_path=DEFAULT_INDEX, res=NCC_RES):
    """build (or load cached) the ncc template matrix: each library sprite -> a res*res grayscale vector,
    returns (T, T2): T = (n, res*res) float32 raw vectors,
    T2 = T**2 (reused for the masked-cosine template norms)"""
    cache = Path(index_path).with_suffix(f".ncc{res}.npy")
    if cache.is_file() and cache.stat().st_mtime >= Path(index_path).stat().st_mtime:
        T = np.load(cache)
        if T.shape == (len(rows), res * res):
            return T, T * T
    base = Path(index_path).parent
    T = np.empty((len(rows), res * res), np.float32)
    for i, r in enumerate(rows):
        T[i] = _glyph_to_vec(_sprite_glyph_gray(base / r["file"]), res)
    try:
        np.save(cache, T)
    except OSError:
        pass # read-only data dir (e.g. frozen exe) -- just rebuild next run
    return T, T * T


def id_icon_ncc_masked(icon_bgr, rows, templates, pool=None, res=NCC_RES, fg_frac=0.15, fg_floor=20):
    """masked normalized cross-correlation matcher
    build a foreground mask from the query glyph (its bright strokes)
    and score each template by cosine over just those pixels
    templates = (T, T2) from load_ncc_templates.
    returns (row, score, margin)
    score = masked cosine in [-1, 1] (higher better, unlike phash), margin = best - 2nd best."""
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
    return rows[order[0]], float(scores[order[0]]), float(scores[order[0]] - scores[order[1]])


def ncc_plain_templates(templates):
    """z-normed (mean-removed, unit-norm) template matrix for the plain-ncc matcher
    plain ncc keeps the whole glyph-on-black silhouette rather than masking to bright strokes,
    so it z-norms the full vector. built once per run then reused across nodes"""
    T, _ = templates
    Tz = T - T.mean(axis=1, keepdims=True)         # remove each template's DC level
    Tz /= (np.linalg.norm(Tz, axis=1, keepdims=True) + 1e-6)
    return Tz


def id_icon_ncc(icon_bgr, rows, Tz, pool=None, res=NCC_RES):
    """plain normalized cross-correlation matcher
    z-normed cosine between the whole query glyph and each z-normed template (Tz from ncc_plain_templates).
    unlike id_icon_ncc_masked this keeps the full glyph-on-black silhouette,
    instead of masking to bright strokes
    returns (row, score, margin)
    score = cosine in [-1, 1] (HIGHER is better), margin = gap to the 2nd best."""
    q = _glyph_to_vec(cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY), res)
    q = q - q.mean()
    q /= (np.linalg.norm(q) + 1e-6)
    scores = Tz @ q                                                 # (n,) z-normed cosine
    if pool is not None:
        scores = np.where(pool, scores, -2.0)                      # below the min possible cosine
    order = np.argsort(-scores)                                    # higher cosine = better
    return rows[order[0]], float(scores[order[0]]), float(scores[order[0]] - scores[order[1]])


def _crop_glyph_from_frame(frame, x, y, r, r_tol=1):
    """crop frame around glyph bbox"""
    x, y, r, r_eff = int(x), int(y), int(r), int(r * r_tol)
    h, w, = frame.shape[:2]
    
    x0, xf = max(x-r_eff, 0), min(w, x+r_eff)
    y0, yf = max(y-r_eff, 0), min(h, y+r_eff)
    return frame[y0:yf, x0:xf], {'x0': x0, 'y0': y0, 'xf': xf, 'yf': yf}


def _fill_holes(bin_img):
    """fill regions enclosed by a closed rim so each ringed disk becomes a solid blob.
    pad a 1px background ring first so the flood seed (0,0) is ALWAYS background,
    even when the mask runs to the crop edge.
    without the pad, a blob touching (0,0) makes the floodfill paint the whole crop white"""
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
        rmin=30, rmax=100, min_dist_frac=4, accumulator_thresh=20,
        debug=False
    ):
    """detect bloodweb nodes in a given input frame.

    Preprocessing: choice between 3 different threshold methods (adaptive_gaussian, otsu, canny) to preprocess for contour detection
    morphological closing -> flood fill -> morphological open to close gaps and denoise
    
    Detection: Either Hough circles (if use_hough=True) or coarse then refine fine pass over the preprocessed contours
        - rough pass to grab the radii of all countour within (rmin,rmax)
        - peak finding to identify node centers
    """    
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
            # TODO: implement error handling for if no radii found (fail loudly)
            print("ERROR: no radii found in initial pass")
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
            circles.append((cx, cy, r)) 

    if debug: _show(bin_frame, title='find_circles() - final', circles=circles, contours=contours, savefig=False)
    return circles # list of (x, y, r) float


def sample_disk_hsv(hsv, x, y, r, s_floor=30, v_floor=20, min_px=6):
    """median hsv over an annulus around the disk, away from the glyph. annulus not full disk
    so the center glyph and rim anti-aliasing don't pollute the rarity read."""
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


def find_nodes_in_frame(frame, debug=False, max_anchor_dist=None):
    """find all circles in the blood web and clean them up to identify clickable nodes"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # (H,W,3), for the per-circle color read
    circles = find_circles(frame, debug=False) # list of (x, y, r)

    nodes = []
    for x, y, r in circles:
        disk_hsv = sample_disk_hsv(hsv, x, y, r)
        rarity, dist = classify_rarity(disk_hsv)
        if rarity is None:
            continue # remove circle from list if a rarity wasn't obtained
        if max_anchor_dist is not None and dist > max_anchor_dist:
            continue                               # not near any rarity -> probably not a node
        nodes.append((int(x), int(y), int(r), rarity))

    if debug:
        _show(draw_detections(frame, nodes), "find_nodes")
    return nodes


def find_center_node(
        frame, h_tol=8, s_floor=200, v_lo=22, v_hi=95,
        close_ksize=7, min_area_frac=0.0015, roi_frac=0.30
    ):
    """locate the center auto-spend node (dark red entity hexagon) by its glow color.
    find_circles misses it on ~1/4 of frames and there's no rarity anchor for it, so detect it
    off its stable signature instead: hue at the 0/180 seam, high saturation, low (dark) value.
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
    """crop a coarse node, color-mask its rarity socket, and pick the central blob 
    the coarse (x,y,r) from find_circles can be off-center/undersized, so crop wide (r_tol)
    and recenter on the socket centroid; the roi (roi_k*r around the coarse center) keeps hue
    bleed (bronze ring / dim web) from inflating the socket blob
    
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


def normalize_glyph(crop, contour, erode_ksize=3, out_size=GLYPH_SIZE):
    """strip the colored socket from a node crop so only the glyph remains on black, framed to
    match scraper.normalize_sprite (tight-crop -> square-pad centered -> resize)

    glyph = the BRIGHT pixels inside the socket polygon. the white-ish glyph always sits brighter than the rarity disk fill,
    otsu cut the socket's value channel keys the fill out without any per-rarity hue math
    (the old hue-subtraction left colored halos and broke on gold/event) 
    the contour is convex-hulled first so glyph strokes that touch the socket edge don't carve notches out of the interior mask.
    returns a GLYPH_SIZE bgr glyph-on-black square, or None if nothing survives."""
    h, w = crop.shape[:2]
    val = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]   # value channel (brightness), (h,w)

    # interior = filled convex socket polygon, eroded a touch to shed the colored rim/anti-alias
    inside = np.zeros((h, w), np.uint8)
    cv2.drawContours(inside, [cv2.convexHull(contour)], -1, 255, cv2.FILLED)
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


def classify_socket(node_contours, poly_tol=0.05):
    """find the category of the node contents (socket) by exploiting a pattern,
    offerings -> hexagon; perks -> rhombus; items -> square, addons -> square with a plus
    using the contour of the socket we detect it's shape and classify the type of glyph in the node.
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


def detect(frame, rows=None, hashes=None, ncc_templates=None, matcher="ncc", r_tol=1.5, debug=False):
    """full pipeline; returns per-node dicts {x, y, r, rar, cat, glyph_bgr, match, score, margin,
    matcher}. matcher picks identification: 'ncc' (default, plain z-normed cosine, higher=better),
    'ncc_masked' (cosine over the query's bright strokes, higher=better), or 'phash' (hamming,
    lower=better). score's direction follows the matcher (see _score_str)."""
    if rows is None:
        rows, hashes = load_index()
    if matcher == "phash" and hashes is None:
        _, hashes = load_index()
    if matcher in ("ncc", "ncc_masked") and ncc_templates is None:
        ncc_templates = load_ncc_templates(rows)
    ncc_plain_T = ncc_plain_templates(ncc_templates) if matcher == "ncc" else None

    cats = np.array([r['category'] for r in rows]) # (n,) strings of 'item', 'perk', ...
    nodes = find_nodes_in_frame(frame, debug=debug) # [(x, y, r, rarity), ...]

    res = []
    # the center auto-spend node is found by its own glow color, not the disk pipeline, and gets
    # no glyph match. tagged kind='center' so Node/spender can reference it (see find_center_node).
    center = find_center_node(frame)
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
        if center is not None and (x - center[0]) ** 2 + (y - center[1]) ** 2 <= center[2] ** 2:
            continue # this circle is the center node, already emitted above
        iso = isolate_node_contents(frame, x, y, r, rarity, r_tol=r_tol)
        if iso is None:
            if debug: print("WARNING: detect() - Node skipped after failing to isolate node contents")
            continue
        
        cx, cy, r_ref, node_contours, crop = iso
        socket_shape = classify_socket(node_contours) # use socket shape geometry to classify node type (item/addon, perk, offering)
        glyph = normalize_glyph(crop, node_contours) # standardize glyph sizes to match with indexed icons
        if glyph is None:
            if debug: print("WARNING: detect() - Node skipped after not finding a glyph in the socket")
            continue

        # identify the glyph against the library, restricted to the socket-shape category pool
        pool = None if socket_shape is None else np.isin(cats, NODE_SHAPE_DICT[socket_shape])
        if matcher == "ncc":
            best_match_row, score, margin = id_icon_ncc(glyph, rows, ncc_plain_T, pool=pool)
        elif matcher == "ncc_masked":
            best_match_row, score, margin = id_icon_ncc_masked(glyph, rows, ncc_templates, pool=pool)
        else:
            best_match_row, score, margin = id_icon_hamming(glyph, rows, hashes, pool=pool)
        if debug: print(matcher, best_match_row['key'], round(score, 3), round(margin, 3))

        # TODO: resolve descrepancies in observed attrs vs matched icon attrs (from wiki)
        # TODO Moved to spender logic

        res.append({
            'x': cx, 'y': cy, 'r': r_ref,
            'rar': rarity,
            'cat': socket_shape,
            'glyph_bgr': glyph,
            'match': best_match_row, 'score': score, 'margin': margin, 'matcher': matcher,
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
    """draw each node (circle + label) onto a copy of the frame. accepts detect() result dicts
    or the (x, y, r, rarity) tuples from find_nodes_in_frame, so it works at either stage.
    dict label is two lines, 'rarity/socket score' then the matched name on its own line
    keys come straight off detect()'s output (rar, cat, score, matcher, match row)."""
    out = frame.copy()
    for n in nodes:
        if isinstance(n, dict):
            x, y, r = n["x"], n["y"], n["r"]
            if n.get("kind") == "center":
                lines = ["autospend"]
            else:
                match = n.get("match") or {} # match is a library row dict (or None)
                name = match.get("name") or match.get("key") or "?"
                # name on its own line so long icon names stay legible over the busy web
                lines = [f"{n.get('rar', '?')}/{n.get('cat', '?')} {_score_str(n)}", name]
        else:
            x, y, r, rarity = n
            lines = [str(rarity)]
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        # stack the lines just above the circle, last line nearest the rim
        for i, line in enumerate(lines):
            org = (x - r, y - r - 6 - (len(lines) - 1 - i) * 14)
            # black underlay then green so the label stays readable over the busy web background
            cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# matplotlib, not cv2 highgui: the conda opencv build ships without a gui backend, and we
# don't want highgui in the frozen exe anyway. matplotlib also gives zoom/pan for free and
# maps clicks back to image coords even when zoomed, which the 3440x1440 frames need.
def _sample_window(fixture_path):
    """open a fixture and print the hsv under each click. this is how we read real disk
    colors to set/verify the rarity anchors. median over a small patch so one noisy pixel
    doesn't mislead."""
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
    """show an image in a matplotlib window, with optional contour/edge/circle overlays for
    the find_circles debugging. this conda opencv build ships no highgui backend, so cv2.imshow
    throws 'function not implemented' (same reason _sample_window uses matplotlib).

    img is a bgr frame OR a single-channel gray/edge/mask (ndim==2, promoted to bgr so the
    overlays can be colored). contours is a list of cv2 contours (drawn), edges is a binary
    single-channel map (its nonzero pixels painted on), circles is a list of (x, y, r) like
    find_circles returns (drawn as outline + center dot, floats cast to int). overlay colors
    are bgr; composited with cv2 onto a copy, then handed to imshow."""
    # matplotlib clips float rgb to [0,1], so a scalar float map (e.g. the distance transform)
    # sent straight through gray2bgr shows up as solid white. normalize any non-uint8 2d input
    # to 0-255 first so its gradient is actually visible (binary uint8 masks pass through as-is).
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
        plt.savefig(f".tmp/{title}.png", dpi=200)
        plt.close(fig)   # close it, else a later plt.show() from another _show pops this up too
    else:
        plt.show()


def _show_gallery(items, title="glyphs", cols=6, savefig=False):
    """tile a set of (image, caption) pairs in a grid, each captioned. for eyeballing the
    per-node crops/normalized glyphs next to what they got read as (rarity, socket, match),
    instead of squinting at the whole annotated frame. images are bgr or single-channel like
    everywhere else here; a None image draws a blank cell (e.g. normalize_glyph returned None).
    dev-only; matplotlib since this conda cv2 has no highgui backend."""
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
        fig.savefig(f".tmp/{title}.png", dpi=200)
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
    web_bbox = {'x0': 300, 'y0': 200, 'xf': 1500, 'yf': 1300}

    if args.cmd == "sample":
        _sample_window(args.fixture)
    elif args.cmd == "detect":
        frame = cv2.imread(str(args.fixture))
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
        frame = frame[web_bbox['y0']:web_bbox['yf'], web_bbox['x0']:web_bbox['xf']]
        results = detect(frame, matcher=args.matcher, debug=True)
        items = []
        for n in results:
            match = n.get("match") or {}
            name = match.get("name") or match.get("key") or "?"
            cap = f"{n['rar']}/{n['cat']}\n{name}\n{_score_str(n)} m{n['margin']:.2f}"
            items.append((n["glyph_bgr"], cap))
        _show_gallery(items, title=f"detections-{args.fixture.stem}", savefig=args.save)
