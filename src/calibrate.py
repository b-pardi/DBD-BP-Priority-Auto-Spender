"""one-time in-game rarity color calibration (deferred -- skeleton for now).

the shipped product runs this once with the bloodweb open: the user clicks each rarity
disk in turn, we sample the clicked region's hsv, and write it to usr/HSVs.json, which
detect then reads as its anchor colors. exists because the rendered disk color drifts
from the wiki's design hex with each user's gamma/monitor, so sampling real pixels beats
any baked-in value. supersedes the auto-refine-from-frames bootstrap once a user runs it.
"""

from pathlib import Path

# the five bloodweb rarities in tier order. same keys the scraper seed and usr/HSVs.json
# use, so the whole pipeline speaks one vocabulary.
RARITIES = ["common", "uncommon", "rare", "very rare", "ultra rare"]

# usr/ holds per-user runtime config; HSVs.json is the active anchor set detect reads.
ROOT = Path(__file__).resolve().parent.parent
HSV_PATH = ROOT / "usr" / "HSVs.json"


def sample_hsv_at(frame, x, y, patch=5):
    """median hsv over a small patch around a click, not a single pixel.
    frame is bgr (h, w, 3); returns (h, s, v). median so glyph/glow/anti-alias noise
    near the click doesn't skew the read."""
    # TODO: crop the patch around (x, y), cvtColor bgr->hsv, np.median over the patch
    raise NotImplementedError


def calibrate(out_path=HSV_PATH):
    """walk the user through clicking each rarity disk; write anchors to usr/HSVs.json.
    needs a mouse-click listener (the capture lib only grabs the screen, it can't read
    clicks), so the input dependency is still TBD -- decide at build time (pynput / mouse
    / win32)."""
    # TODO: for each rarity in RARITIES:
    #   prompt "click a <rarity> node"
    #   wait for a click, grab the current frame (capture.grab_bloodweb)
    #   anchors[rarity] = sample_hsv_at(frame, click_x, click_y)
    # TODO: write {rarity: [h, s, v]} to out_path as json (mkdir usr/ if missing)
    raise NotImplementedError


if __name__ == "__main__":
    calibrate()
