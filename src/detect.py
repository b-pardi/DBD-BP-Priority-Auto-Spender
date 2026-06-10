"""localize nodes, read rarity, identify the icon. the core offline-testable layer.

pipeline: hsv color-segment the rarity disks to get node centers plus a first read of
rarity, then crop + mask each inner icon and match it against the scraped library via
perceptual hash, with normalized-correlation template matching as a tie-breaker. every
coord is found dynamically (no hardcoded positions) so it survives the 21:9 frame.
"""

# TODO: localize_nodes(frame) -> list of (x, y, radius, rarity)
# TODO: identify_icon(crop) -> (name, category, score)
# TODO: detect(frame) -> detected nodes with name / category / rarity / pos


def detect(frame):
    """given a bgr frame (h, w, 3), return the list of detected bloodweb nodes."""
    raise NotImplementedError
