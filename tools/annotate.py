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
GALLERY_COLS = 6   # columns in the review gallery

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
    """interactive matplotlib gallery: show crops + captions, let the user click cells to flag
    them for relabeling (red border), close the window to confirm. then for each flagged cell
    a terminal prompt asks for the correct key (enter=keep, 'drop'=null, 'skip'=exclude).
    matplotlib only; this conda opencv build has no highgui backend.
    entries is the list of record dicts, each with a 'crop_bgr' key for the image.
    rows is the full index, needed for key lookup after the terminal prompts.
    returns the same list with 'source' and 'key' updated per the user's choices."""

    n = len(entries)
    cols = min(GALLERY_COLS, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        nrows, cols, squeeze=False,
        figsize=(cols * 2.2, nrows * 2.8)
    )
    axes_flat = axes.ravel()
    marked = [False] * n

    def _cell_caption(e):
        ocr_lbl = e["key"] or "UNRESOLVED"
        det_key = (e["matcher_guess"] or "?")[:20]
        sc = e["score"]
        return f"ocr: {ocr_lbl}\ndet: {det_key} s{sc:.2f}\n{e['rarity']}/{e['shape']}"

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
        if marked[i]:
            rect = mpatches.Rectangle(
                [0, 0], 1, 1, fill=False,
                edgecolor="red", linewidth=4,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)

    for i in range(n):
        _redraw(i)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "click cells to flag for relabeling (red border)  |  close window to confirm",
        fontsize=9
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def on_click(event):
        if event.inaxes is None:
            return
        for i, ax in enumerate(axes_flat[:n]):
            if event.inaxes is ax:
                marked[i] = not marked[i]
                _redraw(i)
                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()   # blocks until the window is closed

    # terminal prompts for flagged cells, after the matplotlib event loop exits
    for i, e in enumerate(entries):
        if not marked[i]:
            continue
        print(f"\n--- node {i:02d}: {e['crop_path']} ---")
        print(f"  proposed (ocr): {e['key'] or 'UNRESOLVED'}")
        print(f"  matcher guess:  {e['matcher_guess'] or '?'} s{e['score']:.2f}")
        print(f"  rarity/shape:   {e['rarity']}/{e['shape']}")
        print("  enter=keep  |  type a name/key=relabel  |  'drop'=null key  |  'skip'=exclude")
        resp = input("  > ").strip()
        if resp == "skip":
            e["_skip"] = True
        elif resp == "drop":
            e["key"] = None
            e["source"] = "manual"
        elif resp:
            found_key, found_name = _lookup_key(resp, rows)
            if found_key:
                print(f"    resolved to: {found_key}  ({found_name})")
                e["key"] = found_key
            else:
                print(f"    '{resp}' not found in index, saving raw text as key")
                e["key"] = resp
            e["source"] = "manual"
        # enter with no text keeps the current e["key"] and e["source"] unchanged

    return entries


# ------------------------------------------------------------- main annotator

def annotate_web(matcher="ncc", hover_delay_s=None):
    """capture the current web, detect + ocr each node, review in a gallery, append labels."""
    print("[annotate] loading index and ncc templates...")
    rows, _ = detect.load_index()
    ncc_templates = detect.load_ncc_templates(rows)

    print(f"[annotate] starting in {STARTUP_S:.0f}s, switch to the game now...")
    time.sleep(STARTUP_S)

    web_id = f"web_{datetime.now():%H%M%S}"
    print(f"[annotate] capturing web {web_id}...")

    frame, region = capture.grab_with_region()
    bbox = ocr.find_web_bbox(frame)
    if bbox:
        x0, y0, x1, y1 = bbox
        sub = frame[y0:y1, x0:x1]
        print(f"[annotate] web bbox {bbox}")
    else:
        x0, y0 = 0, 0
        sub = frame
        print("[annotate] no bbox found, using full frame (stray detections possible)")

    print("[annotate] running detect pipeline...")
    dets = detect.detect(
        sub, rows=rows, ncc_templates=ncc_templates,
        matcher=matcher, debug=False
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
    args = ap.parse_args()
    annotate_web(matcher=args.matcher, hover_delay_s=args.hover_delay)
