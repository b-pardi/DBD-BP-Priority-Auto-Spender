"""ocr helpers: the optional bp-threshold read, and the node-identity hover scan.

tesserocr over fixed text regions, loaded lazily through ocr_runtime.get_tesserocr so its
conda dlls resolve in dev and in a frozen exe (see ocr_runtime).
read_bp backs the optional stop threshold, reading the top-bar bloodpoint total.
find_node_tooltip is the fallback identity source for nodes detect could not trust, it hovers
the node so dbd shows the name tooltip and reads that instead of guessing.
see node.Node.needs_resolution for what gets routed here.
"""

import os
import re
import sys
import time
import difflib

import cv2
import numpy as np
from PIL import Image

from .node import normalize_name
from .ocr_runtime import get_tesserocr

# bp value sits right anchored in the top bar, as fractions of frame w/h (calibrated 3440x1440).
BP_REGION = (0.8765, 0.0472, 0.9215, 0.0764) # x0, y0, x1, y1
BP_THRESH = 150 # binarize cutoff that keeps only the bright digits

HOVER_DELAY_S = 0.1 # wait after move_to for dbd's tooltip to fade in
PARK_XY = (0.52, 0.42) # neutral cursor rest in the empty fog right of the web, clears any tooltip
DIFF_THRESH = 25 # binarize cutoff on the before/after change
MIN_BOX_FRAC = 0.02 # the tooltip blob must cover at least this fraction of the frame
NAME_BAND = 0.45 # ocr the top this fraction of the located box, where name and subhead live
FUZZY_CUTOFF = 0.8 # difflib ratio floor when no exact index hit

# bloodweb auto-crop anchors: fixed ui labels whose ocr'd pixel boxes bound the web, so detection
# can be cropped to the web and stray ui icons (settings/friends/prestige) never become fake nodes.
# the boxes are stable per resolution, so find_web_bbox reads them once and the source caches it.
ANCHOR_TOP_ZONE = (0.08, 0.03, 0.52, 0.24)  # SHARED PERKS (web left+top) + SPEND BLOODPOINTS (right)
ANCHOR_BL_ZONE = (0.0, 0.82, 0.28, 1.0)     # BACK [ESC] button, sits just below the web bottom
CROP_PAD_FRAC = 0.02                         # outward pad on left/top/right (bottom pads inward)

_fold_cache = None # {normalized name: row}, built once from the index rows
_apis = {} # cached PyTessBaseAPI per (psm, whitelist), reused so tesseract inits once


def _tessdata():
    """path to the eng tessdata, the conda share dir in dev or the bundle dir when frozen."""
    cands = []
    if getattr(sys, "frozen", False):
        cands.append(os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "tessdata"))
    cands += [
        os.path.join(sys.prefix, "share", "tessdata"),
        os.path.join(sys.prefix, "Library", "share", "tessdata"),
        os.environ.get("TESSDATA_PREFIX", ""),
    ]
    for p in cands:
        if p and os.path.isfile(os.path.join(p, "eng.traineddata")):
            return p
    return get_tesserocr().get_languages()[0].rstrip("/\\") # fall back to tesserocr's own guess


def _api(psm, whitelist=None):
    """a cached tesserocr api for the given page-seg mode and optional char whitelist."""
    key = (psm, whitelist)
    if key not in _apis:
        t = get_tesserocr()
        api = t.PyTessBaseAPI(psm=psm, path=_tessdata())
        if whitelist:
            api.SetVariable("tessedit_char_whitelist", whitelist)
        _apis[key] = api
    return _apis[key]


def _ocr_word_boxes(frame, zone, scale=2):
    """ocr a fractional zone (fx0,fy0,fx1,fy1) of the frame, returning [(UPPER_TEXT, (x0,y0,x1,y1))]
    with the word boxes in FULL-FRAME pixel coords. sparse-text psm to catch scattered ui labels,
    word-level boxes via the result iterator. backs find_web_bbox's anchor search."""
    h, w = frame.shape[:2]
    zx0, zy0, zx1, zy1 = int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)
    g = cv2.cvtColor(frame[zy0:zy1, zx0:zx1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # tesseract likes big glyphs
    t = get_tesserocr()
    api = _api(t.PSM.SPARSE_TEXT)
    api.SetImage(Image.fromarray(g))
    api.Recognize()
    out = []
    for word in t.iterate_level(api.GetIterator(), t.RIL.WORD):
        txt = (word.GetUTF8Text(t.RIL.WORD) or "").strip()
        if not txt:
            continue
        bx0, by0, bx1, by1 = word.BoundingBox(t.RIL.WORD)            # zone-local, at `scale`
        out.append((txt.upper(), (zx0 + bx0 // scale, zy0 + by0 // scale,
                                  zx0 + bx1 // scale, zy0 + by1 // scale)))
    return out


def find_web_bbox(frame, pad_frac=CROP_PAD_FRAC):
    """auto-locate the bloodweb's bounding box by ocr-ing fixed ui anchor labels, so detection can be
    cropped to the web (keeping stray ui icons like settings/friends/prestige out of the node
    detector). anchors, stable per resolution: 'SHARED PERKS' marks the web's left edge + top,
    'SPEND BLOODPOINTS' the right edge, 'BACK [ESC]' the bottom. returns (x0, y0, x1, y1) in
    full-frame pixels, or None when too few anchors are found (caller then uses the full frame).
    left/top/right pad OUTWARD; the bottom pads INWARD, because the settings/friends icon row sits
    just below the BACK button and an outward pad would re-include exactly those stray icons.
    frame is the full bgr grab (h, w, 3)."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    top = _ocr_word_boxes(frame, ANCHOR_TOP_ZONE)
    bl = _ocr_word_boxes(frame, ANCHOR_BL_ZONE)

    def box_of(words, *needles):
        return next((b for txt, b in words if any(n in txt for n in needles)), None)

    shared = box_of(top, "SHARED")               # web left edge + a top reference
    back = box_of(bl, "BACK", "ESC")             # sits just below the web bottom
    # 'SPEND BLOODPOINTS' is two words; the web's right edge is BLOODPOINTS' (rightmost) edge, so
    # take the max right edge over both rather than the first match (SPEND alone stops short).
    right_boxes = [b for txt, b in top if "BLOODPOINTS" in txt or "SPEND" in txt]

    left = shared[0] if shared else None
    right = max((b[2] for b in right_boxes), default=None)
    bottom = back[1] if back else None
    tops = [b[1] for b in right_boxes] + ([shared[1]] if shared else [])
    top_y = min(tops, default=None)
    if None in (left, right, bottom, top_y):
        return None                              # not enough anchors, fall back to the full frame

    pad = int(pad_frac * h)
    x0, y0 = max(left - pad, 0), max(top_y - pad, 0)
    x1, y1 = min(right + pad, w), min(bottom - pad, h)  # bottom pads inward, clearing the icon row
    if x1 - x0 < 0.2 * w or y1 - y0 < 0.2 * h:
        return None                              # implausibly small, treat as a failed read
    return (x0, y0, x1, y1)


def read_bp(frame):
    """current bloodpoint total from the top bar, or None if it cannot be read.
    crops the right anchored bp value, keeps the bright digits, ocrs them digit only.
    the loop compares this against config stop_bp_threshold to optionally stop spending.
    frame is the full bgr grab (h, w, 3).
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(BP_REGION[0] * w), int(BP_REGION[1] * h),
                      int(BP_REGION[2] * w), int(BP_REGION[3] * h))
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # upscale, tesseract likes big glyphs
    _, th = cv2.threshold(g, BP_THRESH, 255, cv2.THRESH_BINARY)
    api = _api(get_tesserocr().PSM.SINGLE_LINE, "0123456789 ")
    api.SetImage(Image.fromarray(th))
    digits = re.sub(r"\D", "", api.GetUTF8Text())
    return int(digits) if len(digits) >= 4 else None # ignore stray short misreads


def read_tooltip(tooltip_crop_bgr):
    """ocr a located tooltip box and return its top text lines (the name then the subhead).
    crops the top NAME_BAND where the name and subhead sit, above the description, and reads
    that block as lines so the caller can match the name line against the index.
    the box-top can wobble up into the entity thorns, but that just adds junk lines that never
    match a name key, so the real name line still wins.
    tooltip_crop_bgr is the bgr tooltip box from _locate_tooltip.
    """
    bh = tooltip_crop_bgr.shape[0]
    g = cv2.cvtColor(tooltip_crop_bgr[:int(NAME_BAND * bh)], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) # name caps are smallish
    api = _api(get_tesserocr().PSM.SINGLE_BLOCK)
    api.SetImage(Image.fromarray(g))
    return [ln.strip() for ln in api.GetUTF8Text().splitlines() if ln.strip()]


def _locate_tooltip(before, after):
    """bbox (x, y, w, h) of the tooltip that appeared between two grabs, or None.
    the tooltip is the largest dense rectangular change, found by thresholding the abs diff and
    taking the biggest blob with a high fill extent.
    the only real competitor is the character model idling between grabs, which is tall and low
    extent so the extent gate drops it.
    before and after are full bgr grabs (h, w, 3) of the same region.
    """
    gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ga = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = cv2.GaussianBlur(np.abs(ga - gb).astype(np.uint8), (7, 7), 0)
    _, th = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)) # merge the panel into one blob
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) # drop the speckle
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    h, w = th.shape
    best = None # (bbox, area)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        extent = area / float(bw * bh)
        if extent > 0.5 and area > MIN_BOX_FRAC * h * w and (best is None or area > best[1]):
            best = ((int(x), int(y), int(bw), int(bh)), int(area))
    return best[0] if best else None


def _fold(rows):
    """index lookup map, normalized name to row, built once and cached."""
    global _fold_cache
    if _fold_cache is None:
        _fold_cache = {normalize_name(r["name"]): r for r in rows}
    return _fold_cache


def _match_name(lines, rows):
    """the index row whose name matches one of the ocr'd lines, or None.
    tries an exact normalized hit first since the subhead and description lines never normalize
    to a name key, then a difflib pass for the noisier reads like the busy event headers.
    """
    fold = _fold(rows)
    for ln in lines:
        row = fold.get(normalize_name(ln))
        if row:
            return row
    keys = list(fold)
    best, best_ratio = None, 0.0
    for ln in lines:
        nl = normalize_name(ln)
        if not nl:
            continue
        m = difflib.get_close_matches(nl, keys, n=1, cutoff=FUZZY_CUTOFF)
        if m:
            ratio = difflib.SequenceMatcher(None, nl, m[0]).ratio()
            if ratio > best_ratio:
                best, best_ratio = fold[m[0]], ratio
    return best


def find_node_tooltip(node, frame, region, rows, hover_delay_s=None):
    """identify a node by hovering it and reading dbd's name tooltip, mutating node in place.
    detect routes here the nodes it could not trust (node.needs_resolution) rather than guessing.
    the tooltip is anchored to the node and flips side to stay on screen, so we park the cursor,
    grab a clean before frame, hover the node, grab the after, and read the box that appeared.
    on a read it sets node.name, node.match and node.resolved_by 'ocr', else leaves node as is.
    live only, the caller skips this when frame is None (the sim path).
    frame is the detection grab (h, w, 3) and region maps frame coords to screen.
    hover_delay_s overrides HOVER_DELAY_S (the tooltip fade-in wait); raise it if reads fail because
    the tooltip hadn't appeared yet. None uses the default.
    """
    from . import capture, input_control # live deps imported lazily so ocr stays importable for tests
    hover_delay_s = HOVER_DELAY_S if hover_delay_s is None else hover_delay_s
    h, w = frame.shape[:2]

    # park off any node so the before frame holds no stale tooltip from a previous hover.
    px, py = capture.frame_to_screen(int(PARK_XY[0] * w), int(PARK_XY[1] * h), region)
    input_control.move_to(px, py)
    before, _ = capture.grab_with_region(region)

    # hover the node and let its tooltip fade in, then grab the after frame.
    sx, sy = capture.frame_to_screen(node.x, node.y, region)
    input_control.move_to(sx, sy)
    time.sleep(hover_delay_s)
    after, _ = capture.grab_with_region(region)

    box = _locate_tooltip(before, after)
    if box is None:
        return node # tooltip never localized, leave the node for the rules to skip
    x, y, bw, bh = box
    row = _match_name(read_tooltip(after[y:y + bh, x:x + bw]), rows)
    if row:
        node.name = row["name"]
        node.match = row
        node.resolved_by = "ocr"
    return node
