"""single source of truth for every resolution-dependent constant in the detect/ocr pipeline.

everything here was hard-tuned against one 3440x1440 capture: find_circles' rmin/rmax and the ui
regions ocr.py reads. rmin/rmax scale by h/1440 (node size tracks height, not width). web-relative
zones (anchors, tooltip, park) stay frame fractions since the web is centered. but the top-bar reads
(bp / level / prestige crest) are EDGE-anchored in dbd's hud, so a width-fraction crop only lands
right at the baseline aspect ratio: on 16:9 the bp crop slid off the right-aligned number and read
only its trailing digits. those three are baseline px offsets from their anchor edge (scaled by
height, resolved by *_region_px); at the baseline they equal the old fractional crops, so ultrawide
is untouched.

GLYPH_SIZE (detect.py / scraper.py) is deliberately not here: it's a fixed template size the matcher
compares against, not a screen measurement, so it never scales.

Resolution.from_frame(frame) builds one off a real bgr capture; the bare Resolution() baseline
reproduces every constant's original hardcoded value, so a caller that passes no resolution behaves
exactly as before on the 3440x1440 fixtures.
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

    # top-bar reads: baseline px offsets from an anchor edge (scaled by height in *_region_px), NOT
    # width fractions. left-anchored give (x0, x1) px from the LEFT edge; bp is (left, right) px from
    # the RIGHT edge (the counter is right-aligned, grows leftward with more digits). y is a fraction
    # of height. at the baseline these equal the old fractions (LEVEL 0.145/0.245, CREST 0.113/0.142,
    # BP 0.8765/0.9215 of w 3440), so ultrawide reads exactly as before.
    LEVEL_X = (499, 843)              # ocr.read_bloodweb_level: "BLOODWEB LEVEL n" strip below the name
    LEVEL_Y = (0.065, 0.088)
    PRESTIGE_CREST_X = (389, 489)    # ocr.read_prestige_level: crest digit left of the name (empty crest = prestige 0)
    PRESTIGE_CREST_Y = (0.045, 0.088)
    BP_X_FROM_RIGHT = (402, 240)     # ocr.read_bp: top-bar bp total
    BP_Y = (0.0472, 0.0764)

    # fractional ui regions (fx0, fy0, fx1, fy1) or (fx, fy): web-centered or generous zones, close
    # enough as a fraction of frame. centralized here so ocr.py and detect's debug cockpit share them.
    PRESTIGE_TOOLTIP_REGION = (0.29, 0.42, 0.55, 0.58)    # ocr.read_center_hover_text: where the hovered center's "PRESTIGE LEVEL n" tooltip lands
    OK_REGION = (0.87, 0.81, 0.96, 0.89)                  # ocr.find_ok_button: the REWARDS UNLOCKED screen's OK button
    OK_CLICK_XY = (0.910, 0.852)                          # ok button center, clicked to dismiss the rewards screen
    ANCHOR_TOP_ZONE = (0.08, 0.03, 0.52, 0.24)             # ocr.find_web_bbox: SHARED/SHAREABLE PERKS / SPEND BLOODPOINTS
    ANCHOR_BL_ZONE = (0.0, 0.82, 0.28, 1.0)                # ocr.find_web_bbox: BACK [ESC]
    PARK_XY = (0.01, 0.01)                                 # neutral cursor rest, off the web entirely. NOT (0,0): the exact corner is pydirectinput's failsafe point, so the first input call after parking there raises FailSafeException and kills the run
    WEB_BBOX_FALLBACK = (300 / BASELINE_W, 200 / BASELINE_H,
                          1500 / BASELINE_W, 1300 / BASELINE_H)  # detect.py debug cockpit's fixed test crop

    # bloodweb slot lattice in baseline px: nodes sit on three fixed rings around the web center
    # (inner ring only has the 6 slots at 30+60k degrees; the even-60 junctions binarize circle-ish
    # but never hold a node). measured off 17 webs: every real node lands within ~10px of a slot,
    # every junk detection 40px+ off, so snapping separates them cleanly. detect fits center + scale
    # per frame from the circles, so these need no per-resolution tuning.
    LATTICE_RADII = (163.5, 330.5, 474.5)                  # ring radii
    LATTICE_PHASES = (30.0, 15.0, 0.0)                     # first-slot angle per ring, degrees
    LATTICE_SLOTS = (6, 12, 12)                            # slots per ring, evenly spaced
    NODE_RADIUS = 49.5                                     # node disk radius, snapped detections are forced to this

    @property
    def web_span_px(self):
        """web's full diameter in px (outer ring + a node disk either side).
        backstops ocr.find_web_bbox's right edge, which is unreliable off the baseline aspect ratio."""
        return 2 * (self.LATTICE_RADII[-1] + self.NODE_RADIUS) * self.scale

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

    def _left_region_px(self, xoff, yfrac):
        """a left-anchored top-bar crop {x0,y0,x1,y1}: x offsets are baseline px from the LEFT edge
        scaled by height, y is a fraction of height."""
        return {'x0': round(xoff[0] * self.scale), 'y0': round(yfrac[0] * self.h),
                'x1': round(xoff[1] * self.scale), 'y1': round(yfrac[1] * self.h)}

    def level_region_px(self):
        """read_bloodweb_level's crop box in this frame's px."""
        return self._left_region_px(self.LEVEL_X, self.LEVEL_Y)

    def prestige_crest_region_px(self):
        """read_prestige_level's crop box in this frame's px."""
        return self._left_region_px(self.PRESTIGE_CREST_X, self.PRESTIGE_CREST_Y)

    def bp_region_px(self):
        """read_bp's crop box in this frame's px, right-anchored (x = width - baseline offset scaled by
        height) so it tracks the right-aligned counter on any aspect ratio instead of drifting off it."""
        return {'x0': self.w - round(self.BP_X_FROM_RIGHT[0] * self.scale),
                'y0': round(self.BP_Y[0] * self.h),
                'x1': self.w - round(self.BP_X_FROM_RIGHT[1] * self.scale),
                'y1': round(self.BP_Y[1] * self.h)}

    def web_bbox_fallback_px(self):
        """WEB_BBOX_FALLBACK in this Resolution's own pixels, {'x0','y0','xf','yf'}
        (detect.py's debug cockpit crops fixtures with this before running detect on them)."""
        fx0, fy0, fx1, fy1 = self.WEB_BBOX_FALLBACK
        return {'x0': round(fx0 * self.w), 'y0': round(fy0 * self.h),
                'xf': round(fx1 * self.w), 'yf': round(fy1 * self.h)}
