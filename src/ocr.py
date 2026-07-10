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
from .resolution import Resolution

# bp value sits right anchored in the top bar. the fraction itself lives on Resolution
# (Resolution.BP_REGION) since it is one of the frame-fraction regions centralized there; read_bp
# resolves resolution.BP_REGION off the frame it is given.
BP_THRESH = 150 # binarize cutoff that keeps only the bright digits

HOVER_DELAY_S = 0.1 # wait after move_to for dbd's tooltip to fade in
DIFF_THRESH = 25 # binarize cutoff on the before/after change
MIN_BOX_FRAC = 0.02 # the tooltip blob must cover at least this fraction of the frame
NAME_BAND = 0.45 # ocr the top this fraction of the located box, where name and subhead live
FUZZY_CUTOFF = 0.8 # difflib ratio floor when no exact index hit

# bloodweb auto-crop anchors: fixed ui labels whose ocr'd pixel boxes bound the web, so detection
# can be cropped to the web and stray ui icons (settings/friends/prestige) never become fake nodes.
# the zones are also frame fractions on Resolution (ANCHOR_TOP_ZONE/ANCHOR_BL_ZONE), stable per
# resolution, so find_web_bbox reads them once off the frame and the source caches the result.
CROP_PAD_FRAC = 0.02 # outward pad on left/top/right (bottom pads inward)

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


def _union_box(boxes):
    """smallest box (x0,y0,x1,y1) covering all given word boxes, or None when empty.
    used to rebuild a multi-word label's full box from its per-word ocr boxes."""
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _ui_masks(top_words):
    """rectangles (full-frame px) to blank out before detection so ui glyphs sitting just under the
    top anchor labels can't be mistaken for bloodweb nodes: the character 'SHARED PERKS' row (its 3
    perk icons, informational only) and the 'SPEND BLOODPOINTS' button. each rect is anchored to its
    label's ocr'd text box so it tracks resolution. top_words is _ocr_word_boxes' top-zone output."""
    masks = []
    perks = _union_box([b for txt, b in top_words if "SHARED" in txt or "PERK" in txt])
    if perks:
        tx0, ty0, tx1, ty1 = perks
        wdt, hgt = tx1 - tx0, ty1 - ty0
        # from the text's bottom-left: right 1.5x its width, down 3x its height (the perk-icon row)
        masks.append((tx0, ty1, int(tx0 + 1.5 * wdt), int(ty1 + 3 * hgt)))
    spend = _union_box([b for txt, b in top_words if "SPEND" in txt or "BLOODPOINTS" in txt])
    if spend:
        tx0, ty0, tx1, ty1 = spend
        hgt = ty1 - ty0
        mid = (tx0 + tx1) // 2
        # from the text's bottom-midpoint to its right edge, down 4x its height (the spend button)
        masks.append((mid, ty1, tx1, int(ty1 + 4 * hgt)))
    return masks


def apply_ui_masks(sub, masks, origin=(0, 0)):
    """blank out the _ui_masks rectangles on a detection crop so their glyphs can't become fake
    nodes. masks are full-frame (x0,y0,x1,y1); origin is the crop's top-left (x0,y0) so the rects
    map into sub's local coords. copies first (sub is a view of the returned full frame), fills the
    rects black, and returns the copy; a no-op (returns sub unchanged) when there are no masks."""
    if not masks:
        return sub
    sub = sub.copy()
    h, w = sub.shape[:2]
    ox, oy = origin
    for mx0, my0, mx1, my1 in masks:
        x0, y0 = max(mx0 - ox, 0), max(my0 - oy, 0)
        x1, y1 = min(mx1 - ox, w), min(my1 - oy, h)
        if x1 > x0 and y1 > y0:
            sub[y0:y1, x0:x1] = 0
    return sub


def find_web_bbox(frame, pad_frac=CROP_PAD_FRAC, resolution=None):
    """auto-locate the bloodweb's bounding box by ocr-ing fixed ui anchor labels, so detection can be
    cropped to the web (keeping stray ui icons like settings/friends/prestige out of the node
    detector). anchors, stable per resolution: 'SHARED PERKS' marks the web's left edge + top,
    'SPEND BLOODPOINTS' the right edge, 'BACK [ESC]' the bottom. returns (bbox, masks): bbox is
    (x0, y0, x1, y1) in full-frame pixels, or None when too few anchors are found (caller then uses
    the full frame); masks is a list of _ui_masks rects (full-frame px) the caller blanks out via
    apply_ui_masks so the shared-perks row and spend button never register as nodes.
    left/top/right pad OUTWARD; the bottom pads INWARD, because the settings/friends icon row sits
    just below the BACK button and an outward pad would re-include exactly those stray icons.
    frame is the full bgr grab (h, w, 3). resolution (a Resolution, default
    Resolution.from_frame(frame)) supplies the anchor search zones."""
    if frame is None:
        return None, []
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]
    top = _ocr_word_boxes(frame, resolution.ANCHOR_TOP_ZONE)
    bl = _ocr_word_boxes(frame, resolution.ANCHOR_BL_ZONE)

    def box_of(words, *needles):
        return next((b for txt, b in words if any(n in txt for n in needles)), None)

    # the perks label reads 'SHARED PERKS' at prestige >=1 but 'SHAREABLE PERKS' at prestige 0, so
    # match on 'SHARE' (a substring of both) with 'PERK' as a fallback, else prestige-0 webs never crop.
    shared = box_of(top, "SHARE", "PERK")        # web left edge + a top reference
    back = box_of(bl, "BACK", "ESC")             # sits just below the web bottom
    # 'SPEND BLOODPOINTS' is two words; the web's right edge is BLOODPOINTS' (rightmost) edge, so
    # take the max right edge over both rather than the first match (SPEND alone stops short).
    right_boxes = [b for txt, b in top if "BLOODPOINTS" in txt or "SPEND" in txt]

    masks = _ui_masks(top)                        # perk row + spend button, independent of the bbox

    left = shared[0] if shared else None
    right = max((b[2] for b in right_boxes), default=None)
    bottom = back[1] if back else None
    tops = [b[1] for b in right_boxes] + ([shared[1]] if shared else [])
    top_y = min(tops, default=None)
    if None in (left, right, bottom, top_y):
        return None, masks                       # not enough anchors, fall back to the full frame

    pad = int(pad_frac * h)
    x0, y0 = max(left - pad, 0), max(top_y - pad, 0)
    x1, y1 = min(right + pad, w), min(bottom - pad, h)  # bottom pads inward, clearing the icon row
    if x1 - x0 < 0.2 * w or y1 - y0 < 0.2 * h:
        return None, masks                       # implausibly small, treat as a failed read
    return (x0, y0, x1, y1), masks


def read_bp(frame, resolution=None):
    """current bloodpoint total from the top bar, or None if it cannot be read.
    crops the right anchored bp value, keeps the bright digits, ocrs them digit only.
    the loop compares this against config stop_bp_threshold to optionally stop spending.
    frame is the full bgr grab (h, w, 3). resolution (a Resolution, default
    Resolution.from_frame(frame)) supplies the bp region.
    """
    if frame is None:
        return None
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]
    bp_region = resolution.BP_REGION
    x0, y0, x1, y1 = (int(bp_region[0] * w), int(bp_region[1] * h),
                      int(bp_region[2] * w), int(bp_region[3] * h))
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # upscale, tesseract likes big glyphs
    _, th = cv2.threshold(g, BP_THRESH, 255, cv2.THRESH_BINARY)
    api = _api(get_tesserocr().PSM.SINGLE_LINE, "0123456789 ")
    api.SetImage(Image.fromarray(th))
    digits = re.sub(r"\D", "", api.GetUTF8Text())
    return int(digits) if len(digits) >= 4 else None # ignore stray short misreads


def _read_region_text(frame, region, scale=3, psm=None, binarize=False):
    """ocr a fractional region (fx0,fy0,fx1,fy1) of the frame and return its text UPPERCASED.
    a small generic reader for the fixed-label reads (prestige tooltip, ok button); sparse-text psm
    by default so a lone word like 'OK' still registers. binarize (otsu) rescues dim text like the
    dimmed rewards-screen OK button, which reads as garbage off the raw grayscale. frame is the full
    bgr grab (h, w, 3)."""
    t = get_tesserocr()
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(region[0] * w), int(region[1] * h),
                      int(region[2] * w), int(region[3] * h))
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if binarize:
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    api = _api(psm if psm is not None else t.PSM.SPARSE_TEXT)
    api.SetImage(Image.fromarray(g))
    return (api.GetUTF8Text() or "").upper()


def read_bloodweb_level(frame, resolution=None):
    """current bloodweb level (1..50) from the strip under the character name, or None.
    reads the whole 'BLOODWEB LEVEL n' line and takes the trailing number, so a garbled 'BLOODWEB'
    doesn't matter as long as the digits read. backs the level goal-stop and the prestige-ready
    trigger (level 50 = the web the prestige star replaces). frame is the full bgr grab (h, w, 3).
    resolution (default Resolution.from_frame(frame)) supplies LEVEL_REGION."""
    if frame is None:
        return None
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]
    region = resolution.LEVEL_REGION
    x0, y0, x1, y1 = (int(region[0] * w), int(region[1] * h),
                      int(region[2] * w), int(region[3] * h))
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)  # keep the bright text
    api = _api(get_tesserocr().PSM.SINGLE_LINE)
    api.SetImage(Image.fromarray(th))
    nums = re.findall(r"\d+", api.GetUTF8Text())
    if not nums:
        return None
    lvl = int(nums[-1])
    return lvl if 1 <= lvl <= 50 else None  # bloodweb levels only run 1..50


def read_prestige_level(frame, resolution=None):
    """current prestige level from the crest to the left of the name, or None if it can't be read.
    the crest digit sits on a noisy stone texture, so crop tight to its center, upscale hard, otsu,
    and open/close away the speckle before a digit-only ocr. an empty crest (no digit) is prestige 0.
    verified against the prestige 0/1 fixtures; higher and two-digit prestige are unproven, so this is
    the best-effort read the user opted into (used for the prestige goal-stop and the debug readout,
    never to decide whether to prestige, which keys off level 50 + a hover confirm instead).
    frame is the full bgr grab (h, w, 3). resolution supplies PRESTIGE_CREST_REGION."""
    if frame is None:
        return None
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]
    region = resolution.PRESTIGE_CREST_REGION
    x0, y0, x1, y1 = (int(region[0] * w), int(region[1] * h),
                      int(region[2] * w), int(region[3] * h))
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # kill speckle
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # solidify strokes
    api = _api(get_tesserocr().PSM.SINGLE_LINE, "0123456789")
    api.SetImage(Image.fromarray(th))
    digits = re.sub(r"\D", "", api.GetUTF8Text())
    return int(digits) if digits else 0  # no digit found = empty crest = prestige 0


def read_center_hover_text(frame, region, center_xy, resolution=None, hover_delay_s=None):
    """hover the web center and return the tooltip text UPPERCASED, for the prestige-ready confirm.
    center_xy is the full-frame web center remembered from the filled level-50 web (the prestige star
    sits there, and the empty prestige screen's own center detection is unreliable). parks first so no
    stale tooltip lingers, hovers, lets the tooltip fade in, then ocrs PRESTIGE_TOOLTIP_REGION; the
    caller checks for 'PRESTIGE'. live only. frame is the last grab (for shape + resolution)."""
    from . import capture, input_control  # live deps imported lazily so ocr stays test-importable
    hover_delay_s = HOVER_DELAY_S if hover_delay_s is None else hover_delay_s
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]
    park_xy = resolution.PARK_XY
    px, py = capture.frame_to_screen(int(park_xy[0] * w), int(park_xy[1] * h), region)
    input_control.move_to(px, py)
    sx, sy = capture.frame_to_screen(int(center_xy[0]), int(center_xy[1]), region)
    input_control.move_to(sx, sy)
    time.sleep(hover_delay_s)
    after, _ = capture.grab_with_region(region)
    return _read_region_text(after, resolution.PRESTIGE_TOOLTIP_REGION)


def find_ok_button(frame, resolution=None):
    """the REWARDS UNLOCKED screen's OK button click point (full-frame px), or None if it isn't up.
    ocrs OK_REGION and, when 'OK' is present, returns OK_CLICK_XY in pixels so do_prestige can dismiss
    the rewards screen after a prestige. frame is the full bgr grab (h, w, 3)."""
    if frame is None:
        return None
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]
    if "OK" in _read_region_text(frame, resolution.OK_REGION, binarize=True):
        cx, cy = resolution.OK_CLICK_XY
        return int(cx * w), int(cy * h)
    return None


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


def find_node_tooltip(node, frame, region, rows, hover_delay_s=None, resolution=None):
    """identify a node by hovering it and reading dbd's name tooltip, mutating node in place.
    detect routes here the nodes it could not trust (node.needs_resolution) rather than guessing.
    the tooltip is anchored to the node and flips side to stay on screen, so we park the cursor,
    grab a clean before frame, hover the node, grab the after, and read the box that appeared.
    on a read it sets node.name, node.match and node.resolved_by 'ocr', else leaves node as is.
    live only, the caller skips this when frame is None (the sim path).
    frame is the detection grab (h, w, 3) and region maps frame coords to screen.
    hover_delay_s overrides HOVER_DELAY_S (the tooltip fade-in wait); raise it if reads fail because
    the tooltip hadn't appeared yet. None uses the default. resolution (a Resolution, default
    Resolution.from_frame(frame)) supplies the park spot.
    """
    from . import capture, input_control # live deps imported lazily so ocr stays importable for tests
    hover_delay_s = HOVER_DELAY_S if hover_delay_s is None else hover_delay_s
    resolution = resolution or Resolution.from_frame(frame)
    h, w = frame.shape[:2]

    # park off any node so the before frame holds no stale tooltip from a previous hover.
    park_xy = resolution.PARK_XY
    px, py = capture.frame_to_screen(int(park_xy[0] * w), int(park_xy[1] * h), region)
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
