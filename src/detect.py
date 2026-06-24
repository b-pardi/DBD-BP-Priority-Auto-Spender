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

# rarity anchor colors: load (usr, auto-seeded from EMPIRICAL_SEED), classify, refine (part 4)
_HSVs = None  # cached {rarity: [h, s, v]} for the run


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
        lo, hi = max(0, h - htol), min(h + htol, 179) # chop off tol at H bounds, may consider wrapping band around back to 0 but fine for now
        cur_rarity_mask = cv2.inRange(hsv, (lo, s_floor, v_floor), (hi, 255, 255))
        frame_mask |= cur_rarity_mask
    
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
    gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)          # same luma weights as PIL 'L'
    q = imagehash.phash(Image.fromarray(gray)).hash.flatten()  # (64,) bool
    dists = np.count_nonzero(ref_hashes != q, axis=1)          # (n,) hamming: differing bits
    if pool is not None:
        dists = np.where(pool, dists, 65)                      # 65 > max dist of 64
    order = np.argsort(dists)
    return rows[order[0]], int(dists[order[0]]), int(dists[order[1]] - dists[order[0]])


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
    if thresh_method.lower() == 'adaptive_gaussian':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) # blur a little to tame glyph/web texture
        #bin_frame = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)
        bin_frame = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

    elif thresh_method.lower() == 'otsu':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) # blur a little to tame glyph/web texture
        _, bin_frame = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif thresh_method.lower() == 'canny':
        blur_frame = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0) # binarized canny on bloodweb nodes sucks in b/w, blur on color
        bin_frame = cv2.Canny(blur_frame, canny_lo, canny_hi, L2gradient=True)

    else:
        raise ValueError("Invalid Thresholding Method. Valid types: ['adaptive_gaussian', 'otsu', 'canny']")

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
    # hue is circular (0..179 wraps at the red seam)
    # unwrap to the branch nearest the circular mean, then median there.
    hue = px[:, 0].astype(np.float32)
    ang = hue * (np.pi / 90.0) # 0..179 -> radians (h*2 degrees)
    mu = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * (90.0 / np.pi)
    h_med = (mu + np.median((hue - mu + 90) % 180 - 90)) % 180
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


def normalize_glyph(img_bgr, rarity=None, htol=12, s_floor=40, out_size=128):
    """strip the colored rarity disk from a node crop so only the glyph remains on black, matching the scraped sprites
    then tight-crop + square-pad so the framing matches.
    phash squishes whatever it gets to 32x32,
    so what makes a query hash comparable to a template hash is identical framing, not size.

    kept separate from the scraper's normalize on purpose: detection isn't pixel-perfect
    (off-center crop, ragged contour, ring bleed), so this side does the disk removal and
    cleanup the sprite never needs. the two only have to agree on the final framing

    returns a bgr glyph-on-black square, or None if nothing survives the mask"""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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
        dh = np.minimum(dh, 180 - dh)               # hue is circular
        is_disk = (dh <= htol) & (hsv[..., 1] >= s_floor)
    else:
        is_disk = np.zeros((h, w), bool)            # no rarity -> fall back to inner crop only

    glyph_mask = inside & ~is_disk
    ys, xs = np.where(glyph_mask)
    if len(xs) == 0:
        return None

    glyph = np.zeros_like(img_bgr)
    glyph[glyph_mask] = img_bgr[glyph_mask]                       # glyph px on black
    glyph = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # tight-crop to glyph bbox

    # square-pad (preserve aspect) so a wide vs tall glyph stays distinguishable after phash's
    # square resize, then resize to a fixed size (also lets the ncc tie-breaker compare same-size).
    gh, gw = glyph.shape[:2]
    side = max(gh, gw)
    canvas = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - gh) // 2, (side - gw) // 2
    canvas[y0:y0 + gh, x0:x0 + gw] = glyph
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)


def detect(frame, rows=None, hashes=None, debug=False):
    """full pipeline; returns [{name, category, rarity, x, y, radius, dist, margin}].

    pseudocode:
      rows, hashes = (rows, hashes) or load_index()
      for (x, y, r, rarity) in find_nodes_in_frame(frame):
        crop = frame[y-r:y+r, x-r:x+r]
        category = read_socket_shape(frame, x, y, r)     # may be None
        glyph = normalize_glyph(crop, rarity)
        pool = bool mask of rows in `category` (or None)
        row, dist, margin = id_icon(glyph, rows, hashes, pool=pool)
        # reconcile: trust shape over color; use the matched row's wiki rarity (when not
        # null) to sanity-check or override the disk-color rarity, especially blue vs purple
        results.append({...})
      return results
    """
    nodes = find_nodes_in_frame(frame, debug=True)

    return nodes



# part 10: debug cockpit. sample disk colors and visualize detections on fixtures.
def draw_detections(frame, nodes):
    """draw each node (circle + label) onto a copy of the frame. accepts detect() dicts or
    (x, y, r, rarity) tuples from find_nodes_in_frame, so it works at either stage."""
    out = frame.copy()
    for n in nodes:
        if isinstance(n, dict):
            x, y, r = n["x"], n["y"], n["radius"]
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
        vis[edges > 0] = edge_color                    # binary-edge pixels -> overlay color
    if contours is not None:
        cv2.drawContours(vis, contours, -1, contour_color, 1)
    if circles is not None:
        for x, y, r in circles:
            c = (int(round(x)), int(round(y)))
            cv2.circle(vis, c, int(round(r)), circle_color, 2)  # the found disk
            cv2.circle(vis, c, 2, circle_color, -1)             # center dot
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))   # matplotlib wants rgb
    plt.title(title)
    if savefig:
        plt.savefig(f".tmp/{title}.png", dpi=200)
        plt.close(fig)   # close it, else a later plt.show() from another _show pops this up too
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
    args = ap.parse_args()

    if args.cmd == "sample":
        _sample_window(args.fixture)
    elif args.cmd == "detect":
        frame = cv2.imread(str(args.fixture))
        # TODO: find a way to automatically determine bounds
        web_bbox = {
            'x0': 300,
            'y0': 200,
            'xf': 1500,
            'yf': 1300
        }
        frame = frame[web_bbox['y0']:web_bbox['yf'], web_bbox['x0']: web_bbox['xf']]
        nodes = detect(frame)  # raises NotImplementedError until parts 5-9 are filled
        viz = draw_detections(frame, nodes)
        if args.save:
            cv2.imwrite(str(args.save), viz)
        _show(viz, "detect")
