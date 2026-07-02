"""dev-only live bloodweb node annotator.

capture the current web from the screen (game must be visible and focused), detect all
bloodweb nodes, ocr-hover each non-center one for an identity read, and show a gallery of
node crops for review. click cells to flag for relabeling; close the gallery to confirm.
flagged cells get a terminal prompt for the correct key. results are appended (idempotent by
crop path) to data/labels/real_nodes.json.

run: python tools/annotate.py [--matcher ncc] [--hover-delay 0.1]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import capture, detect, ocr
from src.node import normalize_name, Node

LABELS_DIR = ROOT / "data" / "labels"
CROPS_DIR = LABELS_DIR / "crops"
LABELS_JSON = LABELS_DIR / "real_nodes.json"

BOX_K = 1.3       # node crop half-width in node radii, matches eval_matchers.BOX_K
GALLERY_COLS = 8   # columns in the review gallery

STARTUP_S = 3.0    # seconds to switch focus to the game before capture fires


# ----------------------------------------------------------------- file helpers

def _crop_box(frame, x, y, r):
    """bgr square crop centered on (x, y) with half-width BOX_K*r, clamped to frame bounds."""
    h, w = frame.shape[:2]
    hw = int(round(BOX_K * r))
    x0, y0 = max(x - hw, 0), max(y - hw, 0)
    x1, y1 = min(x + hw, w), min(y + hw, h)
    return frame[y0:y1, x0:x1].copy()


def _save_crop(crop_bgr, web_id, idx):
    """write a node crop to data/labels/crops/ and return the path relative to the repo root."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{web_id}_{idx:02d}.png"
    out = CROPS_DIR / fname
    cv2.imwrite(str(out), crop_bgr)
    return str(out.relative_to(ROOT)).replace("\\", "/")  # forward slashes for cross-platform keys


def _load_labels():
    """existing label records keyed by crop_path, or an empty dict if the file doesn't exist."""
    if not LABELS_JSON.is_file():
        return {}
    rows = json.loads(LABELS_JSON.read_text("utf-8"))
    return {r["crop_path"]: r for r in rows}


def _save_labels(existing, new_entries):
    """merge new_entries into the existing dict (keyed by crop_path) and write to disk.
    idempotent: a duplicate crop_path overwrites the old record rather than duplicating it."""
    merged = dict(existing)
    for e in new_entries:
        merged[e["crop_path"]] = e
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    LABELS_JSON.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
    return merged


def _lookup_key(text, rows):
    """resolve a user-typed string to an index key.
    tries exact key match first, then normalize_name match against both key and name.
    returns (key, name) or (None, None) if nothing in the index fits."""
    if not text:
        return None, None
    for r in rows:
        if r["key"] == text:
            return r["key"], r["name"]
    needle = normalize_name(text)
    for r in rows:
        if normalize_name(r["key"]) == needle or normalize_name(r["name"]) == needle:
            return r["key"], r["name"]
    return None, None


# --------------------------------------------------------------- gallery review

def _gallery_review(entries, rows):
    """interactive matplotlib gallery for reviewing detected nodes before saving labels.

    each cell shows the crop plus a caption with the final label (ocr when resolved, else
    UNRESOLVED), the detector's guess and score, and rarity/shape. the final label is always
    the ocr read when it resolved; the detector guess is shown for comparison only and is never
    auto-promoted (it may be wrong).

    two click actions:
      left click  = wrong label (orange border). node is a real bloodweb node but the identity
                    is incorrect. fires a terminal prompt after the gallery closes.
      right click = false detection (red border). not a real bloodweb node at all (background
                    object, ui chrome, etc.). saved as source='false_detection', key=null.

    close the window to confirm all current states. then terminal prompts fire for each
    orange (relabel) cell: enter=keep current label, type a name/key=relabel, 'skip'=exclude.

    entries is the list of record dicts, each with a 'crop_bgr' key for the image.
    rows is the full index, for key lookup in the terminal prompts.
    returns the same list with 'source' and 'key' updated per the user's choices."""

    n = len(entries)
    cols = min(GALLERY_COLS, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        nrows, cols, squeeze=False,
        figsize=(cols * 3.5, nrows * 4.0)
    )
    axes_flat = axes.ravel()
    # per-cell state: None=ok, 'relabel'=wrong label (left click), 'false'=false detection (right click)
    state = [None] * n

    def _cell_caption(e):
        label = e["key"] or "UNRESOLVED"
        det = (e["matcher_guess"] or "?")[:20]
        return f"label: {label}\ndet: {det} s{e['score']:.2f}\n{e['rarity']}/{e['shape']}"

    def _redraw(i):
        ax = axes_flat[i]
        ax.clear()
        e = entries[i]
        img = e.get("crop_bgr")
        if img is not None and img.size:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(np.zeros((8, 8, 3), np.uint8))
        ax.set_title(_cell_caption(e), fontsize=6, pad=2)
        ax.axis("off")
        color = {"relabel": "orange", "false": "red"}.get(state[i])
        if color:
            rect = mpatches.Rectangle(
                [0, 0], 1, 1, fill=False,
                edgecolor=color, linewidth=4,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)

    for i in range(n):
        _redraw(i)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "left click = wrong label (orange)  |  right click = false detection (red)  |  close to confirm",
        fontsize=9
    )
    # skip tight_layout entirely — it fights wspace=0 and re-introduces horizontal gaps.
    # set all margins explicitly so wspace=0 actually sticks.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.35, wspace=0.0)

    def on_click(event):
        if event.inaxes is None or event.button not in (1, 3):
            return
        action = "relabel" if event.button == 1 else "false"
        for i, ax in enumerate(axes_flat[:n]):
            if event.inaxes is ax:
                state[i] = None if state[i] == action else action  # toggle off if already set
                _redraw(i)
                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()   # blocks until the window is closed

    # handle false detections first: mark and move on, no prompt needed
    for i, e in enumerate(entries):
        if state[i] == "false":
            e["key"] = None
            e["source"] = "false_detection"

    # terminal prompts for relabel cells, each showing the crop in a small window
    relabel_cells = [(i, e) for i, e in enumerate(entries) if state[i] == "relabel"]
    for n_done, (i, e) in enumerate(relabel_cells):
        ocr_lbl = e["key"] or "UNRESOLVED"
        det_lbl = e["matcher_guess"] or "?"

        # open a small crop window so the user can see what they're labeling
        crop_fig, crop_ax = plt.subplots(figsize=(3.5, 3.5))
        img = e.get("crop_bgr")
        if img is not None and img.size:
            crop_ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            crop_ax.imshow(np.zeros((8, 8, 3), np.uint8))
        crop_ax.set_title(
            f"node {i:02d}  ({n_done + 1}/{len(relabel_cells)})\n"
            f"label: {ocr_lbl}\ndet: {det_lbl} s{e['score']:.2f}",
            fontsize=8
        )
        crop_ax.axis("off")
        crop_fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)  # pump the event loop long enough for the window to appear

        print(f"\n--- node {i:02d}: {e['crop_path']} ---")
        print(f"  current label: {ocr_lbl}")
        print(f"  detector guess: {det_lbl} s{e['score']:.2f}")
        print(f"  rarity/shape:   {e['rarity']}/{e['shape']}")
        keep_hint = f"keep as '{ocr_lbl}'" if e["key"] else "keep as unresolved"
        print(f"  enter={keep_hint}  |  type a name/key=relabel  |  'skip'=exclude")
        resp = input("  > ").strip()
        plt.close(crop_fig)

        if resp == "skip":
            e["_skip"] = True
        elif resp:
            found_key, found_name = _lookup_key(resp, rows)
            if found_key:
                print(f"    resolved to: {found_key}  ({found_name})")
                e["key"] = found_key
            else:
                print(f"    '{resp}' not found in index, saving raw text as key")
                e["key"] = resp
            e["source"] = "manual"
        # enter with no text: keep e["key"] and e["source"] as-is

    return entries


# ------------------------------------------------------------- main annotator

def annotate_web(matcher="ncc", hover_delay_s=None, use_hough=False):
    """capture the current web, detect + ocr each node, review in a gallery, append labels."""
    print("[annotate] loading index and ncc templates...")
    rows, _ = detect.load_index()
    ncc_templates = detect.load_ncc_templates(rows)

    print(f"[annotate] starting in {STARTUP_S:.0f}s, switch to the game now...")
    time.sleep(STARTUP_S)

    web_id = f"web_{datetime.now():%H%M%S}"
    print(f"[annotate] capturing web {web_id}...")

    frame, region = capture.grab_with_region()
    bbox, masks = ocr.find_web_bbox(frame)
    if bbox:
        x0, y0, x1, y1 = bbox
        sub = frame[y0:y1, x0:x1]
        print(f"[annotate] web bbox {bbox}")
    else:
        x0, y0 = 0, 0
        sub = frame
        print("[annotate] no bbox found, using full frame (stray detections possible)")
    sub = ocr.apply_ui_masks(sub, masks, origin=(x0, y0))  # drop the perk row / spend button

    finder = "hough" if use_hough else "contours"
    print(f"[annotate] running detect pipeline (node_finder={finder})...")
    dets = detect.detect(
        sub, rows=rows, ncc_templates=ncc_templates,
        matcher=matcher, use_hough=use_hough, debug=False
    )
    for d in dets:
        d["x"] += x0
        d["y"] += y0   # crop-local -> full-frame, so ocr hover and clicking align

    non_center = [d for d in dets if d.get("kind") != "center"]
    print(f"[annotate] {len(non_center)} non-center nodes detected, hovering for ocr...")

    hover_s = hover_delay_s if hover_delay_s is not None else ocr.HOVER_DELAY_S
    existing = _load_labels()
    entries = []

    for idx, d in enumerate(non_center):
        x, y, r = d["x"], d["y"], d["r"]
        rarity = d.get("rar") or "unknown"
        shape = d.get("cat") or "unknown"
        match_row = d.get("match") or {}
        matcher_guess = match_row.get("key")
        score = float(d.get("score", 0.0))

        crop_bgr = _crop_box(frame, x, y, r)
        crop_path = _save_crop(crop_bgr, web_id, idx)

        node = Node.from_detection(d)
        ocr.find_node_tooltip(node, frame, region, rows, hover_delay_s=hover_s)

        if node.resolved_by == "ocr" and node.match:
            proposed_key = node.match.get("key")
            source = "ocr"
        elif matcher_guess:
            proposed_key = matcher_guess
            source = "matcher"
        else:
            proposed_key = None
            source = "unresolved"

        print(
            f"[annotate]   node {idx:02d}: ocr={proposed_key or 'UNRESOLVED':20s} "
            f"det={matcher_guess or '?':20s} s={score:.2f}  {rarity}/{shape}"
        )

        entries.append({
            "crop_path": crop_path,
            "key": proposed_key,
            "rarity": rarity,
            "shape": shape,
            "web_id": web_id,
            "xy": [x, y],
            "r": r,                 # node radius in pixels, lets eval re-run isolate_node_contents
            "source": source,
            "matcher_guess": matcher_guess,
            "score": score,
            "crop_bgr": crop_bgr,   # in-memory only, stripped before writing to json
        })

    if not entries:
        print("[annotate] no nodes detected, nothing to label")
        return

    print(f"\n[annotate] {len(entries)} nodes ready for review...")
    entries = _gallery_review(entries, rows)

    to_save = []
    for e in entries:
        if e.get("_skip"):
            continue
        rec = {k: v for k, v in e.items() if k not in ("crop_bgr", "_skip")}
        to_save.append(rec)

    merged = _save_labels(existing, to_save)
    n_resolved = sum(1 for e in to_save if e.get("key") is not None)
    print(
        f"\n[annotate] saved {len(to_save)} records "
        f"({n_resolved} resolved, {len(to_save) - n_resolved} unresolved) "
        f"to {LABELS_JSON}"
    )
    print(f"[annotate] {len(merged)} total records in file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="live bloodweb node annotator (dev only)")
    ap.add_argument(
        "--matcher", choices=detect.MATCHERS, default="ncc",
        help="icon matcher for the detection pass (default ncc)"
    )
    ap.add_argument(
        "--hover-delay", type=float, default=None,
        help="seconds to wait after hovering for dbd tooltip fade-in (default uses ocr.HOVER_DELAY_S)"
    )
    ap.add_argument(
        "--node-finder", choices=("contours", "hough"), default="contours",
        help="circle detection method: contours (default) or hough"
    )
    args = ap.parse_args()
    annotate_web(
        matcher=args.matcher,
        hover_delay_s=args.hover_delay,
        use_hough=(args.node_finder == "hough")
    )
