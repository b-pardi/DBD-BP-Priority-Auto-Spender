"""ocr helpers: the optional bp-threshold read, and the node-identity hover scan.

tesserocr over fixed text regions, loaded lazily through ocr_runtime.get_tesserocr so
its conda dlls resolve in dev and in a frozen exe (see ocr_runtime). bp reading backs
the optional stop threshold; the hover scan is the fallback identity source when the
matcher and the observed disk reads disagree or the match is weak (see
node.Node.needs_resolution).
"""

from .ocr_runtime import get_tesserocr


# TODO: read_bp(frame) -> int bloodpoints, or None if unreadable
#   tesserocr = get_tesserocr(); digits = tesserocr.image_to_text(bp_crop)


def read_bp(frame):
    raise NotImplementedError


def read_tooltip(tooltip_crop_bgr):
    # read item type using ocr
    raise NotImplementedError


def find_node_tooltip(node, frame, region, rows):
    """identify a node by hovering it and ocr-reading the name tooltip.

    detect() keeps both the observed reads (disk rarity, socket shape) and the matched icon.
    a node is trusted when those agree and the match is confident. when they don't, we don't
    guess: we hover the node so dbd shows its name tooltip and read that text instead. this is
    live-only (it moves the real cursor), so the loop skips it when there is no frame (sim).

    steps:
        1. sx, sy = capture.frame_to_screen(node.x, node.y, region)
        2. input_control.move_to(sx, sy); wait briefly for dbd's hover tooltip to appear.
        3. re-grab the frame (or a region around the tooltip) and crop the tooltip text box.
           the tooltip sits at a roughly fixed offset from the node, calibrate that once.
        4. tesserocr = get_tesserocr(); text = tesserocr.image_to_text(tooltip_crop) -> raw name.
        5. row = lookup by node.normalize_name() against the index (exact first, then fuzzy).
        6. on success write it back: node.name, node.match = row; node.resolved_by = "ocr".
           on failure leave node untouched (it just won't satisfy specific-item rules).
        return node  (mutated in place)
    """
    raise NotImplementedError
