"""localize nodes, read rarity, identify the icon. the core offline-testable layer.

pipeline: hsv color-segment the rarity disks to get node centers plus a first read of
rarity, then crop + mask each inner icon and match it against the scraped library via
perceptual hash, with normalized-correlation template matching as a tie-breaker. every
coord is found dynamically (no hardcoded positions) so it works for varying aspect ratios.
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

RARITIES = ["common", "uncommon", "rare", "very rare", "ultra rare"]

# empirical rarity disk anchors, opencv hsv [h(0-179),s,v].
# used bloodweb screenshots from tests/fixtures/ with src.detect sample <web_path>
# to click nodes and print HSV vals
EMPIRICAL_SEED = {
    "common":     [11, 93, 51],     # brown
    "uncommon":   [61, 166, 72],    # green
    "rare":       [108, 141, 77],   # blue
    "very rare":  [143, 141, 77],   # purple
    "ultra rare": [171, 209, 117],  # pink/iri
}

# hue carries the rarity signal so weight it most; value is the most gamma/brightness
# sensitive so weight it least. that weighting is most of our gamma tolerance.
HSV_WEIGHTS = (4.0, 1.0, 0.3)

# normalized glyph canvas size. MUST match scraper.GLYPH_SIZE: query and template glyphs have
# to be framed identically or their phashes aren't comparable. phash downsamples to 32x32
# internally so the exact value barely matters, only that both sides agree.
GLYPH_SIZE = 128

# rarity anchor colors: load (usr, auto-seeded from EMPIRICAL_SEED), classify, refine (part 4)
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
    hue wraps at the red/brown seam (0 and 180 are the same color), so a plain subtract is
    wrong near it. vectorized: pass a scalar or an ndarray of hues, get the same shape back.
    this is the circularity primitive behind both the disk-hue median and the rarity hue mask."""
    return (np.asarray(hues, dtype=np.float32) - ref + 90.0) % 180.0 - 90.0


def hue_circular_mean(hues):
    """circular mean of a set of opencv hues (0..179). a plain mean is wrong across the 0/180
    seam, so average on the unit circle (hue is h*2 degrees) and map the angle back. used as
    the stable anchor the disk-hue median offsets from."""
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


def id_icon(icon_bgr, rows, ref_hashes, pool=None):
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


def _crop_glyph_from_frame(frame, x, y, r, r_tol=1):
    """crop frame around glyph bbox"""
    x, y, r, r_eff = int(x), int(y), int(r), int(r * r_tol)
    h, w, = frame.shape[:2]
    
    x0, xf = max(x-r_eff, 0), min(w, x+r_eff)
    y0, yf = max(y-r_eff, 0), min(h, y+r_eff)
    return frame[y0:yf, x0:xf]



def _fill_holes(bin_img):
    """fill regions enclosed by a closed rim so each ringed disk becomes a solid blob.
    floodfill the outside background from a corner, invert -> only enclosed holes remain,
    OR them back in. needs (0,0) to actually be background (guard if your border has noise)."""
    h, w = bin_img.shape
    ff = bin_img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8) # floodFill wants a +2 border mask
    cv2.floodFill(ff, mask, (0, 0), 255) # flood outside bg white
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
    """part 5 (localize) + part 6 (classify rarity). geometry-first: find the equal-radius
    disk cluster, then read rarity per circle. returns [(x, y, r, rarity), ...], coords
    dynamic so it survives the 21:9 frame. max_anchor_dist (when set) drops circles whose
    color isn't near any anchor -> a color sanity check on the geometry; left off until the
    real distances are eyeballed on fixtures."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # (H,W,3), for the per-circle color read
    circles = find_circles(frame, debug=False) # list of (x, y, r)

    nodes = []
    for i, (x, y, r) in enumerate(circles):
        disk_hsv = sample_disk_hsv(hsv, x, y, r)
        rarity, dist = classify_rarity(disk_hsv)
        if rarity is None:
            circles.pop(i) # remove circle from list if a rarity wasn't obtained
        if max_anchor_dist is not None and dist > max_anchor_dist:
            continue                               # not near any rarity -> probably not a node
        nodes.append((int(x), int(y), int(r), rarity))

    if debug:
        _show(draw_detections(frame, nodes), "find_nodes")
    return nodes


def refine_node(
        frame, x_hat, y_hat, r_hat, rarity,
        r_tol=1.25, close_ksize=5,
        h_tol=8, s_floor=25, v_floor=22, min_px=50
    ):
    """for each node, rerun contour detection to refine node circle more precisely at a local level
    takes in full frame and initial estimate of circle bounds, returns refined circle bounds
    """
    node_crop = _crop_glyph_from_frame(frame, x_hat, y_hat, r_hat, r_tol=r_tol)
    node_hsv = cv2.cvtColor(node_crop, cv2.COLOR_BGR2HSV)
    h_ref = get_ref_hsvs()[rarity][0] # hue anchor of this node's rarity
    mask = hue_band_mask(node_hsv, h_ref, h_tol, s_floor, v_floor)  # wrap-safe color isolation
    if cv2.countNonZero(mask) < min_px:  # countNonZero, not len(): masked px, not crop rows
        print("WARNING: Node Color not in rarity band; Ideally this is due to node refinement removing a non-existent but detected node.")
        return None
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    _show(mask, title='refine_node() - initial mask')
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel) # close gaps in color mask
    _show(mask, title='refine_node() - morph close')
    mask = _fill_holes(mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _show(mask, title='refine_node() - fill holes + contours', contours=contours)


def normalize_glyph(frame, rarity=None, htol=12, s_floor=40, out_size=GLYPH_SIZE):
    """strip the colored rarity disk from a node crop so only the glyph remains on black, matching the scraped sprites
    then tight-crop + square-pad so the framing matches.
    phash squishes whatever it gets to 32x32,
    so what makes a query hash comparable to a template hash is identical framing, not size.

    kept separate from the scraper's normalize on purpose: detection isn't pixel-perfect
    (off-center crop, ragged contour, ring bleed), so this side does the disk removal and
    cleanup the sprite never needs. the two only have to agree on the final framing

    returns a bgr glyph-on-black square, or None if nothing survives the mask"""
    
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cy, cx, r = h / 2, w / 2, min(h, w) / 2         # crop is centered on the node

    # drop the ring and the out-of-disk corners; shrink a little since the crop/center isn't exact
    yy, xx = np.ogrid[:h, :w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.80 * r) ** 2

    # disk px = near the rarity hue and colored enough; glyph = everything else inside the disk.
    # color-key to black mirrors the sprite's black bg (a gray disk bg would shift the dct coeffs).
    anchors = get_ref_hsvs()
    if rarity in anchors:
        href = anchors[rarity][0]
        dh = np.abs(hsv[..., 0].astype(int) - href)
        dh = np.minimum(dh, 180 - dh) # hue is circular
        is_disk = (dh <= htol) & (hsv[..., 1] >= s_floor)
    else:
        is_disk = np.zeros((h, w), bool) # no rarity -> fall back to inner crop only

    glyph_mask = inside & ~is_disk
    ys, xs = np.where(glyph_mask)
    if len(xs) == 0:
        return None

    glyph = np.zeros_like(frame)
    glyph[glyph_mask] = glyph[glyph_mask] # glyph px on black
    glyph = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1] # tight-crop to glyph bbox

    # square-pad (preserve aspect) so a wide vs tall glyph stays distinguishable after phash's
    # square resize, then resize to a fixed size (also lets the ncc tie-breaker compare same-size).
    gh, gw = glyph.shape[:2]
    side = max(gh, gw)
    canvas = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - gh) // 2, (side - gw) // 2
    canvas[y0:y0 + gh, x0:x0 + gw] = glyph
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)


def read_socket_shape(frame, x, y, r):
    node = _crop_glyph_from_frame(frame, x, y, r)
    bin_node = _binarize(node)
    contours = cv2.findContours(bin_node, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shape = None
    r_closest = np.inf
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        r_cur = np.sqrt(4 * area / np.pi) / 2
        if np.abs(r_cur - r) < np.abs(r_closest - r):
            r_closest = r_cur
            chosen_contour_idx = i

    chosen_contour = contours[chosen_contour_idx]
    approx_contour = cv2.approxPolyDP(chosen_contour, 0.01 * cv2.arcLength(chosen_contour, True), True)
    print(len(approx_contour))
    if len == 6:
        shape = 'hexagon'
    if len == 4:
        shape = 'square-rhombus'

    return shape


def detect(frame, rows=None, hashes=None, r_tol=1.2, debug=False):
    """full pipeline; returns [{name, category, rarity, x, y, radius, dist, margin}]"""
    
    if rows is None or hashes is None:
        rows, hashes = load_index()
    
    cats = np.array([r['category'] for r in rows]) # (n,) strings of 'item', 'perk', ...
    nodes = find_nodes_in_frame(frame, debug=debug)
    
    res = []
    for node in nodes:
        x, y, r, rarity = int(node[0]), int(node[1]), int(node[2]), node[3]
        refined_node = refine_node(frame, x, y, r, rarity, r_tol=1.2)
        if refined_node is None: continue
        x, y, r = refined_node
        node_refined_crop = _crop_glyph_from_frame(frame, x, y, r, r_tol=1) # recrop with exact radius of refined node detection
        _show(node_refined_crop)
        
        # use node contents geometry to classify node type (item/addon, perk, offering)
        node_cat_list = read_socket_shape(node_refined_crop, x, y, r)
        glyph = normalize_glyph(node_refined_crop, rarity) # standardize glyph sizes to match with indexed icons
        #_show(node_refined)
        _show(glyph)

        # classifying glyph with wiki ref icons
        pool = None if node_cat_list is None else np.isin(cats, NODE_SHAPE_DICT[node_cat_list]) # reduce comparison pool given geometric glyph cat
        best_match_row, match_ham_dist, margin = id_icon(glyph, rows, hashes, pool=pool) # phashing glyphs with indexed icons
        print(best_match_row, match_ham_dist, margin)

        # TODO: resolve descrepancies in observed attrs vs matched icon attrs (from wiki)

        res.append({
            'x': x, 'y': y, 'r': r,
            'rar': rarity,
            'cat': node_cat_list,
            'glyph_bgr': glyph,
            'match': best_match_row,
            'ham_dist': match_ham_dist,
            'dist_margin': margin,
        })
    return res


# part 10: debug cockpit. sample disk colors and visualize detections on fixtures.
def draw_detections(frame, nodes):
    """draw each node (circle + label) onto a copy of the frame. accepts detect() dicts or
    (x, y, r, rarity) tuples from find_nodes_in_frame, so it works at either stage."""
    out = frame.copy()
    for n in nodes:
        if isinstance(n, dict):
            x, y, r = n["x"], n["y"], n["r"]
            label = f"{n.get('rarity', '?')} {n.get('name', '?')} d{n.get('dist', '?')}"
        else:
            x, y, r, rarity = n
            label = str(rarity)
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.putText(
            out, label, (x - r, y - r - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
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
            ax.imshow(np.zeros((8, 8, 3), np.uint8))   # blank cell for a missing glyph
        ax.set_title(cap, fontsize=7)
        ax.axis("off")
    for ax in axes[len(items):]:                       # blank the unused tail cells
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
    s_glyphs = sub.add_parser("glyphs", help="gallery of per-node crops + normalized glyphs")
    s_glyphs.add_argument("fixture", type=Path)
    s_glyphs.add_argument(
        "--match", action="store_true",
        help="also run id_icon per node (needs the re-hashed index to mean anything)"
    )
    s_glyphs.add_argument("--save", action="store_true", help="save the gallery to .tmp/ instead of showing")
    args = ap.parse_args()

    # TODO: replace with auto bbox (detect-then-bound on the node cluster, no hardcoded coords)
    web_bbox = {'x0': 300, 'y0': 200, 'xf': 1500, 'yf': 1300}

    if args.cmd == "sample":
        _sample_window(args.fixture)
    elif args.cmd == "detect":
        frame = cv2.imread(str(args.fixture))
        frame = frame[web_bbox['y0']:web_bbox['yf'], web_bbox['x0']:web_bbox['xf']]
        nodes = detect(frame, debug=True)
        viz = draw_detections(frame, nodes)
        if args.save:
            cv2.imwrite(str(args.save), viz)
        _show(viz, "detect")
