"""single source of truth for the default *settings* values: every knob the Settings screen edits.

these are the per-key config defaults an older/incomplete file falls back to (load_config sets them)
and the "Restore all defaults" button resets to. priority lists / profiles are NOT here: they live in
the seeded config file and have their own editor. the engine-tuning gates stay owned by their module
(detect/node/ocr) and are only referenced here so a value lives in one place; the timing + hotkey
literals are owned here (spender imports them back) since nothing outside the spender consumed them.

note this is the code-intended default set, deliberately decoupled from whatever personal values the
shipped seed config happens to carry.
"""

from . import detect, node, ocr

# hotkeys
KILL_KEY = "f8"        # dedicated always-stop panic hotkey, only ever stops, never resumes
START_KEY = "f7"       # start/pause toggle, idle -> running -> paused -> running

# timing (seconds), all tunable live from the settings screen
SETTLE_S = 0.6         # post-buy wait before the rescan
ADVANCE_S = 3.0        # post-auto-spend wait for the fill + level transition to play out
PRESTIGE_WAIT_S = 5.0  # post-prestige-click wait before the rewards OK button appears
ENTITY_SETTLE_S = 0.4  # smoke-render wait before the post-buy state re-read (latching recovers misses)

# pristine copies of the engine-tuning gates, read at import before any runtime override mutates the
# module globals (detect.set_presence_thresh / node.set_rescue_gate).
PRESENCE_THRESH_DEFAULT = detect.PRESENCE_THRESH
RESCUE_MIN_DEFAULT = node.CNN_RESCUE_MIN
RESCUE_MARGIN_DEFAULT = node.CNN_RESCUE_MARGIN
NEARDUP_VETO_DEFAULT = node.NEARDUP_VETO

# config-key -> default value for everything the settings screen owns. keep in sync with the widgets
# in ui/screens/settings.py; the restore button walks exactly this map.
DEFAULT_SETTINGS = {
    # display & accessibility
    "ui_scale": 1.0,            # app-wide ctk widget scaling, 1.0 = default size
    "debug": False,             # show the Debug view
    "show_tooltips": True,      # hover tooltips across the ui
    # hotkeys
    "start_key": START_KEY,
    "kill_key": KILL_KEY,
    # detection & matching
    "matcher": "cnn",           # learned matcher; ncc/ncc_masked/phash are the classical alternates
    "thresh_method": "adaptive_gaussian",
    "node_finder": "contours",  # the contour pass; "hough" is the opencv alternate
    "presence_thresh": PRESENCE_THRESH_DEFAULT,
    "matcher_rescue_min": RESCUE_MIN_DEFAULT,
    "matcher_rescue_margin": RESCUE_MARGIN_DEFAULT,
    "matcher_neardup_veto": NEARDUP_VETO_DEFAULT,
    # match pool (exclusive is a strict subset of inferred, so the ui forces inferred on with it)
    "pool_inferred": True,
    "pool_exclusive": False,
    "weak_match_fallback": True,  # an unread node falls back to its weak icon match rather than skip
    # spend order: reordering ties toward the entity changes which node a tie picks, so off by default
    "entity_race": False,
    # timing
    "settle_s": SETTLE_S,
    "entity_settle_s": ENTITY_SETTLE_S,
    "ocr_hover_s": ocr.HOVER_DELAY_S,
    "advance_s": ADVANCE_S,
    # stops & prestige (all live-only; 0 disables a threshold)
    "auto_prestige": False,       # prestiging spends 20k bp and resets the character, so opt-in
    "prestige_wait_s": PRESTIGE_WAIT_S,
    "stop_bp_threshold": 0,
    "stop_prestige": 0,
    "stop_level": 0,
}
