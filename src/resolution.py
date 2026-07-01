"""single source of truth for every resolution-dependent constant in the detect/ocr pipeline.

everything here was originally hard-tuned against one 3440x1440 capture: find_circles' hough/contour
rmin/rmax, and the fractional ui regions ocr.py reads for the bp total, the web-bbox anchor zones,
and the tooltip-hover park spot. Resolution centralizes them off that baseline so a different capture
resolution can scale the pixel sizes (rmin/rmax scale by h/1440, since node size tracks vertical
resolution more directly than width, aspect ratio varies e.g. ultrawide) while the fractional regions
stay as they are (a fraction of frame is already resolution-independent, that is why they were
expressed that way to begin with).

GLYPH_SIZE (detect.py / scraper.py, the library/query glyph canvas) is deliberately not here: it is
a fixed template size the matcher compares against, not a screen measurement, so it never scales.

Resolution.from_frame(frame) builds one off an actual bgr capture; the bare Resolution() baseline
default reproduces every constant's original hardcoded value, so any caller that does not pass a
resolution keeps behaving exactly as before on the 3440x1440 fixtures.
"""

from dataclasses import dataclass

BASELINE_W = 3440
BASELINE_H = 1440


@dataclass(frozen=True)
class Resolution:
    """frame pixel size plus the constants find_circles/ocr key off it.
    w, h default to the 3440x1440 baseline the pipeline was originally tuned on."""
    w: int = BASELINE_W
    h: int = BASELINE_H

    # fractional ui regions (fx0, fy0, fx1, fy1) or (fx, fy), stable across resolutions since a
    # fraction of frame already is the resolution-independent form. centralized here so ocr.py and
    # detect's debug cockpit read one definition instead of each keeping their own literal.
    BP_REGION = (0.8765, 0.0472, 0.9215, 0.0764)           # ocr.read_bp: top-bar bp total
    ANCHOR_TOP_ZONE = (0.08, 0.03, 0.52, 0.24)             # ocr.find_web_bbox: SHARED PERKS / SPEND BLOODPOINTS
    ANCHOR_BL_ZONE = (0.0, 0.82, 0.28, 1.0)                # ocr.find_web_bbox: BACK [ESC]
    PARK_XY = (0.0, 0.0)                                   # ocr.find_node_tooltip: neutral cursor rest, off the web entirely
    WEB_BBOX_FALLBACK = (300 / BASELINE_W, 200 / BASELINE_H,
                          1500 / BASELINE_W, 1300 / BASELINE_H)  # detect.py debug cockpit's fixed test crop

    @classmethod
    def from_frame(cls, frame):
        """Resolution sized to an actual bgr capture (frame.shape is (h, w, 3) or (h, w))."""
        h, w = frame.shape[:2]
        return cls(w=w, h=h)

    @property
    def scale(self):
        """pixel-size scale vs the baseline capture, height-based since node size tracks vertical
        resolution more directly than width (aspect ratio varies across monitors)."""
        return self.h / BASELINE_H

    @property
    def rmin(self):
        """find_circles' minimum node radius in this Resolution's pixels."""
        return round(30 * self.scale)

    @property
    def rmax(self):
        """find_circles' maximum node radius in this Resolution's pixels."""
        return round(100 * self.scale)

    def web_bbox_fallback_px(self):
        """WEB_BBOX_FALLBACK in this Resolution's own pixels, {'x0','y0','xf','yf'}
        (detect.py's debug cockpit crops fixtures with this before running detect on them)."""
        fx0, fy0, fx1, fy1 = self.WEB_BBOX_FALLBACK
        return {'x0': round(fx0 * self.w), 'y0': round(fy0 * self.h),
                'xf': round(fx1 * self.w), 'yf': round(fy1 * self.h)}
