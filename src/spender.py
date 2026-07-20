"""main loop: scan a level -> resolve -> click priorities -> auto-spend -> advance, with a start/kill switch.

scans once per level, not per buy: dbd auto-buys the cheapest path to any node and node positions
stay put within a level, so we click all our priorities off one snapshot (see run). dry-run mode
logs what it WOULD click and sends no input, for safe testing, and pairs with --sim to exercise
the whole decide step offline with no game open. run control = two global hotkeys (win32
RegisterHotKey, so they fire over a focused fullscreen dbd), a start/pause toggle (launches idle,
no capture or input until pressed) and a dedicated kill, plus the optional bp-threshold stop.

the loop is source-agnostic: a source is a callable returning (frame, region, nodes), where
frame/region are None in sim. live builds nodes from capture + detect, sim from the simulator,
and both feed the identical resolve + choose_next path.
"""



import argparse
import threading
import time
import json
import math
import random
import sys
import ctypes
from ctypes import wintypes
from pathlib import Path

from . import capture, detect, input_control, ocr, paths
from .defaults import (
    ADVANCE_S, DEFAULT_SETTINGS, ENTITY_SETTLE_S, KILL_KEY, NEARDUP_VETO_DEFAULT,
    PRESENCE_THRESH_DEFAULT, PRESTIGE_WAIT_S, RESCUE_MARGIN_DEFAULT, RESCUE_MIN_DEFAULT,
    SETTLE_S, START_KEY,
)
from .node import (
    Node, build_pool_mask, set_neardup_veto, set_rescue_gate, tier_is_ordered, tier_rules,
)
from .resolution import Resolution
from .sim import sim_source

# the tunable settings defaults (keys + values) live in defaults.py, imported above so there is one
# source of truth; only the loop-internal constants nothing else configures stay local here.
IDLE_POLL_S = 0.05   # how often the loop re-checks the switch while idle or paused
REPICK_TOL_PX = 20   # spot-match tolerance for the don't-repick guard, tune with node spacing
PARK_TOL_PX = 25     # how far the cursor may drift from PARK_XY before a scan re-parks it
PARK_FADE_S = 0.5    # post-re-park wait for a hovered tooltip to fade before the re-grab

DEFAULT_CONFIG = paths.config_path()   # frozen-aware: repo config/ in dev, %APPDATA%/dbdbp-pas when frozen
VALID_CATEGORIES = {"item", "addon", "offering", "perk", "power"}
VALID_RARITIES = {"common", "uncommon", "rare", "very rare", "ultra rare", "event"}

# rough per-node bloodpoint cost by rarity, for the bp-floor stop's mid-web estimate. dbd's real cost
# also drifts with the web-level bracket, but the user asked for a rarity-only approximation and an
# imprecise stop, so these are round averages (override via config 'rarity_bp_cost' if a run drifts).
RARITY_BP_COST = {
    "common": 3000, "uncommon": 3250, "rare": 3500,
    "very rare": 4500, "ultra rare": 6000, "event": 4000,
}
DEFAULT_NODE_COST = 3500   # fallback for an unknown/None rarity

PRESTIGE_LEVEL = 50   # the bloodweb level at which the center becomes the prestige star
# win32 global-hotkey plumbing for the run control. RegisterHotKey reserves the key system-wide and
# posts WM_HOTKEY to our own message loop, so it fires no matter which window has focus (including a
# foreground dbd we're clicking in); the old keyboard-lib hook only landed while the terminal had
# focus. caveat: if dbd runs elevated, launch this elevated too or windows (UIPI) blocks the keys.
_MOD_NOREPEAT = 0x4000   # don't auto-repeat the hotkey while the key is held
_WM_HOTKEY = 0x0312
_user32 = ctypes.windll.user32
_user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
_user32.RegisterHotKey.restype = wintypes.BOOL
_user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_user32.GetMessageW.restype = ctypes.c_int   # returns -1 on error, 0 on WM_QUIT, else >0


def _vk(key):
    """map a hotkey name like 'f8' or 'a' to a win32 virtual-key code for RegisterHotKey.
    covers the f1-f24 keys we default to plus single letters/digits, enough for config rebinds."""
    k = key.strip().lower()
    if k.startswith('f') and k[1:].isdigit() and 1 <= int(k[1:]) <= 24:
        return 0x70 + (int(k[1:]) - 1)         # VK_F1..VK_F24
    if len(k) == 1 and k.isalnum():
        return ord(k.upper())                  # '0'-'9' -> 0x30.., 'a'-'z' -> 0x41..
    raise ValueError(f"unsupported hotkey {key!r}; use f1-f24 or a single letter/digit")


class Switch:
    """global-hotkey run control: a start/pause toggle plus a dedicated kill, both backed by
    threading.Events the loop polls cheaply. a daemon thread registers the keys via win32
    RegisterHotKey and pumps the message loop that flips the events, so the hotkeys fire over a
    focused, fullscreen dbd (the old keyboard-lib hook only worked while the terminal had focus).

    three states the loop reads off .running / .killed:
      idle     launch default, armed but doing nothing (no capture, no input)
      running  start pressed, the loop does its work
      paused   start pressed again, the loop idles but stays ready to resume
    one class not two because the states are coupled: start toggles running<->paused, while kill
    force-idles from anywhere and latches, so the panic key only ever stops. arm() starts the
    listener and returns self, leaving the control idle; the loop polls."""

    def __init__(self, start_key=START_KEY, kill_key=KILL_KEY):
        self.start_key = start_key
        self.kill_key = kill_key
        self._running = threading.Event()   # set = running, clear = idle/paused
        self._killed = threading.Event()    # set = hard stop latched, terminal
        # fired (from the listener thread) whenever the start hotkey lands, on top of the toggle.
        # lets the ui treat the key as a start BUTTON too: with no run thread alive the toggle alone
        # flips an event nobody is reading, so the ui watches this to spawn a fresh run instead —
        # the whole point being you can start from inside the game without alt-tabbing back.
        self.on_start = None

    def arm(self):
        # RegisterHotKey binds the hotkey to the calling thread, so register + pump in the
        # listener itself, not here on the main thread. daemon so it dies with the process.
        threading.Thread(target=self._listen, daemon=True).start()
        return self

    def _listen(self):
        start_id, kill_id = 1, 2
        if not _user32.RegisterHotKey(None, start_id, _MOD_NOREPEAT, _vk(self.start_key)):
            print(f"warning: start hotkey {self.start_key!r} already in use, couldn't register")
        if not _user32.RegisterHotKey(None, kill_id, _MOD_NOREPEAT, _vk(self.kill_key)):
            print(f"warning: kill hotkey {self.kill_key!r} already in use, couldn't register")
        msg = wintypes.MSG()
        while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == _WM_HOTKEY:
                if msg.wParam == start_id:
                    self.toggle()
                    cb = self.on_start
                    if cb is not None:
                        cb()   # must stay thread-safe: this is the listener thread, not the ui's
                elif msg.wParam == kill_id:
                    self.kill()

    def toggle(self):
        """start/pause: flip running on/off, but a killed control stays idle (kill wins)."""
        if self._killed.is_set():
            return
        if self._running.is_set():
            self._running.clear()
        else:
            self._running.set()

    def kill(self):
        """hard stop from any state: drop out of running and latch killed, the panic path."""
        self._running.clear()
        self._killed.set()

    def reset(self):
        """clear the latched kill (and running) so a ui can start a fresh run after a stop without
        re-registering the global hotkeys. the listener thread and key bindings stay armed."""
        self._killed.clear()
        self._running.clear()

    @property
    def running(self):
        """is the loop cleared to do work this iteration? false while idle, paused, or killed."""
        return self._running.is_set() and not self._killed.is_set()

    @property
    def killed(self):
        """has a hard stop been latched? the loop's outer exit condition."""
        return self._killed.is_set()


def _validate_rule(rule, where="rule"):
    """validate one priority rule dict, raising ValueError with a locating prefix on a bad field.
    a rule is an item (name + optional rarity) or a category (category + optional rarity)."""
    kind = rule.get("type")
    if kind == "item":
        if not rule.get("name"):
            raise ValueError(f"{where}: item rule needs a 'name'")
    elif kind == "category":
        if rule.get("category") not in VALID_CATEGORIES:
            raise ValueError(f"{where}: category must be one of {sorted(VALID_CATEGORIES)}")
    else:
        raise ValueError(f"{where}: type must be 'item' or 'category', got {kind!r}")
    rarity = rule.get("rarity")
    if rarity is not None and rarity not in VALID_RARITIES:
        raise ValueError(f"{where}: bad rarity {rarity!r}")


def normalize_tier(tier):
    """coerce one serialized tier into the canonical in-memory shape {"rules": [...], "ordered": bool}.
    accepts every shape the file format has carried, so old configs and hand-edits load unchanged:
      v1: a bare rule dict            -> its own singleton (unordered) tier
      v2: a list of rule dicts        -> an unordered tier
      v3: {"rules": [...], "ordered"} -> passed through (the within-tier ordering feature)."""
    if isinstance(tier, dict) and "rules" in tier:
        return {"rules": list(tier.get("rules") or []), "ordered": bool(tier.get("ordered", False))}
    if isinstance(tier, dict):                      # a single v1 rule dict
        return {"rules": [tier], "ordered": False}
    return {"rules": list(tier or []), "ordered": False}   # v2 list tier


def normalize_tiers(tiers):
    """normalize a whole priority list (list of tiers) to the canonical per-tier shape."""
    return [normalize_tier(t) for t in (tiers or [])]


def copy_tier(tier):
    """canonical-shape copy of a tier with fresh rule dicts, for the ui's edit buffer."""
    return {"rules": [dict(r) for r in tier_rules(tier)], "ordered": tier_is_ordered(tier)}


def copy_tiers(tiers):
    """copy a whole priority list, each tier deep-ish copied (see copy_tier)."""
    return [copy_tier(t) for t in (tiers or [])]


def serialize_tier(tier):
    """compact on-disk form for one tier: a plain list of rules when unordered (so existing files
    stay byte-for-byte the same shape they were), a {"rules": [...], "ordered": true} dict only when
    the tier is ordered, so the flag round-trips without cluttering every other tier."""
    rules = [dict(r) for r in tier_rules(tier)]
    return {"rules": rules, "ordered": True} if tier_is_ordered(tier) else rules


def serialize_tiers(tiers):
    """serialize a whole priority list to its compact on-disk form (see serialize_tier)."""
    return [serialize_tier(t) for t in (tiers or [])]


def load_config(path=None):
    """read and lightly validate the priority config (schema v3: tiered, tiers optionally ordered).
    `priorities` is an ordered list of tiers; each tier is {"rules": [rule, ...], "ordered": bool}.
    tiers rank strictly high to low. within a tier, an ordered tier prefers the earliest matching
    rule (top = first pick) while a normal tier picks at random across every match (see choose_next).
    a v1 config (flat list of rule dicts) or a v2 config (list of bare-list tiers) is migrated on
    read by normalize_tiers, so old files still load with identical semantics (all tiers unordered).
    returns the parsed dict; raises ValueError on a malformed rule and lets a missing file raise
    FileNotFoundError, so a bad config fails fast at startup instead of silently never matching."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["priorities"] = normalize_tiers(cfg.get("priorities", []))
    for t, tier in enumerate(cfg["priorities"]):
        for i, rule in enumerate(tier_rules(tier)):
            _validate_rule(rule, where=f"priorities[{t}][{i}]")

    # every settings-screen knob defaults from the one map in defaults.py (matcher, timing, hotkeys,
    # pool flags, stops & prestige, ui_scale, tooltips, the detection gates run() pushes into
    # detect/node, ...), so an older/incomplete file fills in and the "Restore defaults" button and
    # this loader can never disagree on a value.
    for key, val in DEFAULT_SETTINGS.items():
        cfg.setdefault(key, val)
    # the rest are not settings-screen knobs, so they live here rather than in DEFAULT_SETTINGS:
    cfg.setdefault("dry_run", True)              # run-screen safety toggle
    # library reveal filters: event glyphs show by default, retired/power ("n/a") rows stay hidden.
    # (supersedes the single hide_unavailable flag, which lumped both behind one checkbox.)
    cfg.setdefault("show_event", True)
    cfg.setdefault("show_na", False)
    return cfg


def save_config(cfg, path=None):
    """write the priority+settings config back as json (schema v3), validating first so we never
    persist a broken file. round-trips what load_config returns and stays human-editable (indent=2).
    every tier (in `priorities` and any named `profiles`) is written through serialize_tier, so an
    unordered tier stays a plain list and only ordered tiers gain the {"rules","ordered"} wrapper.
    the passed cfg is not mutated (the on-disk shapes are built into a copy). single serializer for
    the file; the ui calls it on Save."""
    path = Path(path) if path else DEFAULT_CONFIG
    for t, tier in enumerate(cfg.get("priorities", [])):
        for i, rule in enumerate(tier_rules(tier)):
            _validate_rule(rule, where=f"priorities[{t}][{i}]")
    for name, tiers in (cfg.get("profiles") or {}).items():
        for t, tier in enumerate(tiers):
            for i, rule in enumerate(tier_rules(tier)):
                _validate_rule(rule, where=f"profiles[{name}][{t}][{i}]")
    out = dict(cfg)
    out["priorities"] = serialize_tiers(cfg.get("priorities", []))
    if cfg.get("profiles"):
        out["profiles"] = {name: serialize_tiers(tiers) for name, tiers in cfg["profiles"].items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def park_xy(frame, region):
    """screen coords of the off-web cursor rest spot for this frame (Resolution.PARK_XY)."""
    h, w = frame.shape[:2]
    return capture.frame_to_screen(
        int(Resolution.PARK_XY[0] * w), int(Resolution.PARK_XY[1] * h), region)


def park_cursor(frame, region):
    """move the cursor off the web, so the next grab is a clean read of the web.

    THE INVARIANT: every frame the loop reads must be grabbed with the cursor parked. left on a node,
    dbd draws its tooltip panel over the web, and that panel covers other nodes: it hides their socket
    rings, and the ring is the whole state read (detect.read_node_state), so a bought node under a
    tooltip reads back as available and we re-click a node we already own. it also lands in the debug
    view's annotated frame, over the detections you're trying to look at.
    """
    px, py = park_xy(frame, region)
    input_control.move_to(px, py)
    return px, py


# node sources from live capture
def live_source(
        rows, ncc_templates, region=None, matcher="cnn",
        thresh_method="adaptive_gaussian", use_hough=False, auto_crop=True, web_bbox=None,
        crop_pad_frac=ocr.CROP_PAD_FRAC, debug=False, row_pool=None,
    ):
    """() -> (frame, region, nodes) source backed by the real screen + detect().
    detection runs on a bloodweb crop so stray ui icons can't become fake click targets, but the
    full frame is returned with coords mapped back to it so ocr and clicking are unaffected.
    crop = web_bbox if given, else ocr.find_web_bbox once (cached), else the full frame.
    row_pool (a bool sequence aligned to rows, or None) narrows detect's match library to the
    priority list's icons/sources; snapshotted at run-start, not rebuilt mid-run.
    """
    state = {"bbox": tuple(web_bbox) if web_bbox else None, "masks": None, "logged": False,
             "skip_logged": False}

    def _source():
        frame, reg = capture.grab_with_region(region)
        # park guard: when a level transition outlasts advance_s, dbd re-centers the cursor onto the
        # new center node, undoing do_auto_spend's park and leaving its tooltip over the fresh web.
        # if the cursor drifted off the park spot, re-park, let the tooltip fade, and re-grab a clean
        # frame. also covers a run started with the cursor on a node.
        px, py = park_xy(frame, reg)
        cx, cy = input_control.get_position()
        if abs(cx - px) > PARK_TOL_PX or abs(cy - py) > PARK_TOL_PX:
            input_control.move_to(px, py)
            time.sleep(PARK_FADE_S)
            frame, reg = capture.grab_with_region(region)
        if state["masks"] is None and auto_crop:  # anchor ocr each scan until it reads, then cached
            found, masks = ocr.find_web_bbox(frame, pad_frac=crop_pad_frac)
            if found is not None:
                if state["bbox"] is None:        # keep a preset web_bbox but still take the masks
                    state["bbox"] = found
                state["masks"] = masks
            elif state["bbox"] is None:
                # no anchors and no preset crop: bloodweb probably not on screen (menu, transition,
                # early start). detecting on the full frame is how ui buttons became fake clicked
                # nodes, and the old code CACHED this first failure so one bad frame poisoned the run.
                # skip the scan, return no nodes, retry the anchors next scan.
                if debug and not state["skip_logged"]:
                    print("[crop] anchors not found (bloodweb not visible?), "
                          "skipping scans until they read")
                    state["skip_logged"] = True
                return frame, reg, []
            # preset bbox but flaky anchors this frame, so crop with the preset now,
            # and retry the masks next scan rather than caching a partial or empty mask list.
        state["skip_logged"] = False
        bbox, masks = state["bbox"], state["masks"] or []
        if not state["logged"]:                  # log the crop decision once
            if debug:
                print(f"[crop] web bbox {bbox}" if bbox
                      else "[crop] auto-crop off, using full frame (strays possible)")
                if masks:
                    print(f"[crop] masking {len(masks)} ui region(s) (perk row / spend button)")
            state["logged"] = True

        x0, y0 = (bbox[0], bbox[1]) if bbox else (0, 0)
        sub = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        sub = ocr.apply_ui_masks(sub, masks, origin=(x0, y0))
        # detect's own debug stays off: it pops blocking matplotlib windows, fatal in a live loop.
        dets = detect.detect(
            sub, rows=rows, ncc_templates=ncc_templates, matcher=matcher,
            debug=False, thresh_method=thresh_method, use_hough=use_hough, row_pool=row_pool,
        )
        for d in dets:
            d["x"] += x0
            d["y"] += y0                         # crop-local -> full-frame, so clicks + ocr line up
            if d.get("slot_xy"):                 # a frame coord too, so a state re-read lands on the node
                d["slot_xy"] = (d["slot_xy"][0] + x0, d["slot_xy"][1] + y0)
        if masks:
            # a blanked mask rect binarizes into a solid blob that becomes a fake circle;
            # drop any detection whose center landed in a masked ui region (perk row / spend button).
            dets = [d for d in dets
                    if not any(mx0 <= d["x"] <= mx1 and my0 <= d["y"] <= my1
                               for mx0, my0, mx1, my1 in masks)]
        nodes = [Node.from_detection(d) for d in dets]
        return frame, reg, nodes
    return _source


def _node_tag(n):
    """compact one-line node label for the debug logs: rarity/shape, matched name, the runner-up
    (when present, to show near-tie ambiguity), the raw score+margin, and the click spot."""
    runner = f" vs {n.runner_up!r}" if n.runner_up else ""
    return (f"{n.rarity}/{n.socket_shape} {str(n.name)[:22]!r}{runner} "
            f"s={n.score:.2f} m={n.margin:.3f} @{n.x},{n.y}")


def _log_scan_summary(nodes):
    """one line per scan: how many nodes are trusted vs routed to ocr, and a tally of WHY they
    routed (weak match / category / rarity). this is the at-a-glance answer to 'why is it using
    ocr so much' that the per-node [ocr] lines then itemize."""
    real = [n for n in nodes if not n.is_center]
    ncenter = len(nodes) - len(real)
    taken = [n for n in real if n.taken]           # already bought or entity-eaten: never a target
    real = [n for n in real if not n.taken]
    pooled = [n for n in real if n.pooled_out]     # outside the priority pool: unknown, skipped
    scored = [n for n in real if not n.pooled_out]
    need = [n for n in scored if n.needs_resolution]
    tally = {}
    for n in need:
        for r in n.resolution_reasons:
            head = r.split(" (")[0]          # group "weak match (margin ...)" under "weak match"
            tally[head] = tally.get(head, 0) + 1
    summary = ", ".join(f"{k} x{v}" for k, v in
                        sorted(tally.items(), key=lambda kv: -kv[1])) or "none"
    pooled_str = f", {len(pooled)} pooled-out" if pooled else ""
    if taken:
        bought = sum(1 for n in taken if n.state == detect.STATE_BOUGHT)
        pooled_str += f", {bought} bought + {len(taken) - bought} entity-eaten (skipped)"
    # confident matches whose socket-shape read disagreed: the match won (no ocr routed), but the
    # shape misread stays visible here so a drift in the shape classifier doesn't hide behind it.
    overridden = [n for n in scored if n.shape_overridden]
    if overridden:
        pooled_str += f", {len(overridden)} shape-overridden"
    print(f"[scan] {len(real)} nodes (+{ncenter} center): "
          f"{len(scored) - len(need)} trusted, {len(need)} -> ocr{pooled_str} | reasons: {summary}")
    for n in overridden:
        print(f"[scan]   shape override {_node_tag(n)}: socket read {n.socket_shape!r}, "
              f"trusting the {n.matched_category!r} match")


def resolve_uncertain(nodes, frame, region, rows, debug=False, hover_delay_s=None,
                      weak_match_fallback=True, switch=None):
    """settle identity for nodes flagged needs_resolution via an ocr hover scan.
    a node is trusted when its observed reads and matched icon agree and the match is confident,
    otherwise we hover it and read the name tooltip rather than guess.
    when weak_match_fallback (the default), a hover that reads no tooltip falls back to the weak icon
    match for item rules (ocr_failed set) rather than skipping the node.
    live only, so skip entirely when frame is None (the sim path, nodes stay as detected).
    hover_delay_s is the tooltip fade-in wait before the read; None uses ocr's default (raise it from
    the settings ui if reads fail because the tooltip hadn't appeared yet).
    when debug, log each routed node's reason and the ocr outcome (see also _log_scan_summary).
    switch (a Switch, optional) lets the panic or pause key abort mid-scan: each hover moves the
    mouse and pays the tooltip wait, so a web full of uncertain nodes would otherwise keep hovering
    long after a kill or pause landed. safe to cut short on a pause: the paused loop never reaches
    the pick stage, and the resume's fresh scan re-resolves whatever was left unhovered.
    """
    if frame is None:
        return nodes
    for n in nodes:
        if switch is not None and not switch.running:
            break                        # kill or pause landed mid-scan: stop hovering, caller bails
        if not n.needs_resolution:
            continue
        if debug:
            print(f"[ocr] hover {_node_tag(n)}: {'; '.join(n.resolution_reasons)}")
        before = n.name
        # mutates n, may set resolved_by='ocr'
        ocr.find_node_tooltip(n, frame, region, rows, hover_delay_s=hover_delay_s)
        if n.resolved_by != 'ocr' and weak_match_fallback:
            # ocr fell through: mark it so item rules trust the weak icon match rather than skip
            n.ocr_failed = True
        if debug:
            if n.resolved_by == 'ocr':
                print(f"[ocr]   read {before!r} -> {n.name!r}")
            elif n.ocr_failed:
                print(f"[ocr]   no tooltip read, falling back to icon match {n.name!r}")
            else:
                print(f"[ocr]   no tooltip read, left as {n.name!r} (item rules will skip it)")
    return nodes


def _break_tie(hits, nodes, entity_race):
    """pick one of several equally-ranked nodes.
    default is random (inner/cheaper nodes tend to be lower quality, so no order is imposed).
    entity_race instead takes the node NEAREST the entity, because everything it has eaten is gone for
    good and it keeps eating outward from there, so the nodes beside it are the ones we're about to
    lose. it only kicks in once the entity has actually shown up (the first buys of a web are always
    entity-free), and it never crosses a tier: this reorders WITHIN a tier, it doesn't outrank one.
    ties within a node radius go back to random so two equally-exposed nodes aren't ordered by
    subpixel noise."""
    if not entity_race:
        return random.choice(hits)
    eaten = [n for n in nodes if n.state == detect.STATE_ENTITY]
    if not eaten:
        return random.choice(hits)   # entity hasn't appeared yet, nothing to race
    dists = [min(math.hypot(h.x - e.x, h.y - e.y) for e in eaten) for h in hits]
    nearest = min(dists)
    exposed = [h for h, d in zip(hits, dists) if d <= nearest + h.r]
    return random.choice(exposed)


def refresh_states(nodes, frame):
    """re-read the still-available nodes' state off a fresh frame, in place. returns what CHANGED.

    the once-per-level scan works because node positions don't move within a level, and that cuts both
    ways: a state re-read needs no re-detect either. no find_circles, no glyph extraction, no matcher,
    no ocr hover, just a color read of each socket ring at coords we already have. cheap enough to run
    after every buy, which is the point: dbd auto-buys the whole cheapest path to whatever we click and
    the entity eats nodes mid-web, and a stale snapshot sees neither (8 of 22 targets on one spent
    fixture were already dead).

    STATE LATCHES, and that is what makes the entity read reliable. a node never un-buys and the entity
    never gives one back, so state only ever moves away from 'available' and a taken node is simply
    never re-read. it has to work this way: the entity's smoke pulses and half-renders, so any single
    frame misses ~12% of eaten nodes (they read fine a frame later). latching turns that per-frame read
    into a cumulative one, and since a refresh follows every buy, a miss self-corrects almost at once
    (88% per frame -> 100% over the 8-frame live capture). it also gets cheaper as a web fills.

    a node with no lattice geometry (no fit that scan) can't be re-read and stays available."""
    # the center is skipped, not just spared the work: its own red entity glow reads as 'bought'.
    live = [n for n in nodes
            if not n.is_center and not n.taken and n.ring_r and n.slot_xy]
    states = detect.read_node_states(
        frame, [(n.slot_xy[0], n.slot_xy[1], n.ring_r) for n in live])
    delta = {}
    for n, state in zip(live, states):
        if state != n.state:            # live is available-only, so this only ever latches a node taken
            delta[state] = delta.get(state, 0) + 1
            n.state = state
    return delta


def choose_next(nodes, config):
    """return the highest-priority Node to buy, or None for center auto-spend.
    walks the tiers high to low and stops at the first tier with any matching node:
      normal tier   every node matching any rule in the tier is a hit.
      ordered tier  prefer nodes matching the earliest (topmost) rule, i.e. the first rule with any
                    match wins (within-tier ordering).
    either way the winner among equally-ranked hits comes from _break_tie (random, or entity-race).
    taken nodes never match (Node.matches gates on state), so a bought or eaten node can't be picked.
    returns one node; it tags the choice with its pick provenance (pick_tier / pick_rank /
    pick_ordered) so the loop can log why it won.
    """
    entity_race = bool(config.get("entity_race", False))
    for ti, tier in enumerate(config.get("priorities", [])):
        rules = tier_rules(tier)
        hits = [n for n in nodes if any(n.matches(rule) for rule in rules)]
        if not hits:
            continue
        if tier_is_ordered(tier):
            for rank, rule in enumerate(rules, start=1):
                rule_hits = [n for n in hits if n.matches(rule)]
                if rule_hits:
                    choice = _break_tie(rule_hits, nodes, entity_race)
                    choice.pick_tier, choice.pick_rank, choice.pick_ordered = ti + 1, rank, True
                    return choice
        choice = _break_tie(hits, nodes, entity_race)
        choice.pick_tier, choice.pick_rank, choice.pick_ordered = ti + 1, None, False
        return choice
    return None


def _pick_note(node):
    """compact pick provenance for the buy/dry-run log: which tier the pick came from and, in an
    ordered tier, the within-tier rank of the matched rule. empty when unset (e.g. a sim node not
    routed through choose_next), so the log line is unchanged where there's nothing to add."""
    if node.pick_tier is None:
        return ""
    if node.pick_ordered:
        return f" [tier {node.pick_tier} ordered, rank #{node.pick_rank}]"
    return f" [tier {node.pick_tier} random]"


def _node_cost(node, cost_table):
    """approximate bloodpoint cost of buying one node, by its (disk-read) rarity."""
    return cost_table.get(node.rarity, DEFAULT_NODE_COST)


def estimate_web_cost(nodes, cost_table=RARITY_BP_COST):
    """rough total bloodpoint cost to fill the REMAINING web, summed over every node the center
    auto-spend would still have to buy. pooled-out nodes still cost bp, so they count; taken ones
    don't, since a bought node is already paid for and an entity-eaten one can't be bought at all.
    used by the bp-floor stop to tell an affordable level from the final partial one."""
    return sum(_node_cost(n, cost_table) for n in nodes if not n.is_center and not n.taken)


def _reached_level_goal(prestige, level, stop_prestige, stop_level):
    """has the run reached its (prestige, level) goal, so it should stop? both 0 = no goal.
    with a prestige target we stop once past it, or on it once the level side is also met (level
    defaulting to 1, i.e. the instant that prestige is reached). a level-only goal stops at that
    bloodweb level in the current prestige. a None read (ocr couldn't tell) never triggers a stop."""
    if not stop_prestige and not stop_level:
        return False
    if stop_prestige:
        if prestige is None:
            return False
        if prestige > stop_prestige:
            return True
        if prestige < stop_prestige:
            return False
        return level is not None and level >= (stop_level or 1)   # prestige matches, check the level
    return level is not None and level >= stop_level              # level-only goal


def do_prestige(center_xy, frame, region, hold_s=0.05, prestige_wait_s=PRESTIGE_WAIT_S,
                advance_s=ADVANCE_S, ok_tries=8, ok_wait_s=1.0, switch=None):
    """prestige the character: click the star at center_xy, then dismiss the REWARDS UNLOCKED screen.
    center_xy is the full-frame web center remembered from the filled level-50 web (the prestige star
    sits there; the empty prestige screen's own center read is unreliable, so we reuse the good one).
    clicks the star, waits prestige_wait_s for the animation before the rewards OK button appears,
    polls for that OK button (ocr) and clicks it, then waits advance_s (the bloodweb transition timer)
    for the fresh level-1 web and parks off-web. live only.
    switch (a Switch, optional) makes every wait wake early on a kill, stops the OK polling, and
    skips the park, so the panic key doesn't leave the run clicking or moving the mouse.
    """
    def _wait(s):
        # same duration either way; with a switch the sleep wakes early on a kill
        if switch is None:
            time.sleep(s)
        else:
            _interruptible_sleep(switch, s)

    sx, sy = capture.frame_to_screen(int(center_xy[0]), int(center_xy[1]), region)
    input_control.click_node(sx, sy, hold_s=hold_s)
    _wait(prestige_wait_s)   # animation plays out before the rewards OK button shows up
    for _ in range(ok_tries):
        if switch is not None and switch.killed:
            return   # kill landed while polling the rewards screen: no more grabs or clicks
        f, reg = capture.grab_with_region(region)
        ok = ocr.find_ok_button(f)
        if ok is not None:
            ox, oy = capture.frame_to_screen(int(ok[0]), int(ok[1]), reg)
            input_control.click_node(ox, oy, hold_s=hold_s)
            break
        _wait(ok_wait_s)   # rewards screen not up yet, wait and re-check
    _wait(advance_s)   # bloodweb transition timer: let the fresh level-1 web render
    if switch is not None and switch.killed:
        return   # kill landed during the transition: no park, don't touch the mouse again

    park_cursor(frame, region)   # so the next scan's grab holds no hovered tooltip


def do_auto_spend(nodes, frame, region, hold_s=0.05, advance_s=ADVANCE_S, switch=None):
    """no priority left in this web, hand it to dbd's center entity node to finish the level.
    clicking the center auto-completes the remaining buys and advances to the next web.
    targets the is_center node detect tagged by its red glow (find_center_node), falling back to
    the frame center if it wasn't found.
    waits advance_s for the fill + level transition to play out before returning, so the loop's
    next scan sees the fresh web (detecting a reward/prestige screen instead is a later concern).
    hold_s/advance_s are tuned live like the input_control timings; raise hold_s if dbd needs a
    real press-and-hold to auto-fill rather than a tap.
    switch (a Switch, optional) makes the transition wait wake early on a kill and skips the park
    after one, so the panic key doesn't leave the run moving the mouse.
    """
    center = next((n for n in nodes if n.is_center), None)
    if center is not None:
        fx, fy = center.x, center.y
    elif frame is not None:
        h, w = frame.shape[:2]
        fx, fy = w / 2, h / 2 # center glow not found, aim at the web center
    else:
        return # no frame to click into (sim path never reaches here)

    sx, sy = capture.frame_to_screen(fx, fy, region)
    input_control.click_node(sx, sy, hold_s=hold_s)
    if switch is None:
        time.sleep(advance_s)   # let the fill + level transition play out
    else:
        _interruptible_sleep(switch, advance_s)   # same wait, but a kill wakes it early
        if switch.killed:
            return   # kill landed during the transition: no park, don't touch the mouse again

    # park AFTER the transition, not before: dbd re-centers the cursor onto the center node while the
    # level loads, so a pre-wait park gets undone (leaving the auto-spend tooltip over the fresh web).
    # parking once settled makes it stick, and the loop's settle_s wait lets the tooltip fade.
    if frame is not None:
        park_cursor(frame, region)


SCAN_KEEP = 8   # debug scan pairs kept on disk before the oldest are pruned


def _save_scan_frames(frame, annotated):
    """persist each debug scan (raw grab + annotated detections) under the debug dir's scans/
    folder, so a live detection miss can be reproduced offline by rerunning find_circles on the
    exact raw frame instead of guessing from the log. keeps the newest SCAN_KEEP pairs."""
    import cv2   # spender itself is otherwise cv2-free, keep it a local dep of this debug helper
    d = paths.debug_dir() / "scans"
    d.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%H%M%S")
    cv2.imwrite(str(d / f"scan-{tag}-raw.png"), frame)
    cv2.imwrite(str(d / f"scan-{tag}-det.png"), annotated)
    for old in sorted(d.glob("scan-*-raw.png"))[:-SCAN_KEEP]:
        old.unlink(missing_ok=True)
        (d / old.name.replace("-raw", "-det")).unlink(missing_ok=True)


def _interruptible_sleep(switch, seconds):
    """sleep up to `seconds` but wake early if a kill lands, so the panic key stops the run
    promptly instead of only after the full post-buy wait."""
    end = time.time() + seconds
    while not switch.killed:
        remaining = end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(IDLE_POLL_S, remaining))


def _recently_bought(node, spent, tol=REPICK_TOL_PX):
    """has this node's spot already been clicked on the current web?
    proximity match so a near-duplicate detection of the same node still counts as one click;
    nodes sit well over tol apart, so two distinct nodes won't collide."""
    return any(abs(node.x - sx) <= tol and abs(node.y - sy) <= tol for sx, sy in spent)


class _FileTee:
    """wrap a text stream so everything written also lands in an on-disk log, line-buffered with a
    timestamp prefix per line. lets the loop's existing prints persist (and still show in the
    terminal or ui log pane) without changing any print call. flushes per line so the log survives
    a hard kill mid-scan."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
        self._buf = ""

    def write(self, s):
        self._stream.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            try:
                self._fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")
                self._fh.flush()
            except OSError:
                pass

    def flush(self):
        self._stream.flush()


def _open_run_log():
    """open (append) the shared debug log in the debug dir, or None if it can't be written.
    one file across runs so the whole history is in one place to grep for recurring misreads
    (dev: repo .tmp/spender-debug.log, frozen: %APPDATA%/dbdbp-pas/debug/spender-debug.log)."""
    try:
        d = paths.debug_dir()
        d.mkdir(parents=True, exist_ok=True)
        return open(d / "spender-debug.log", "a", encoding="utf-8")
    except OSError:
        return None


def run(source, config, switch, rows, click=True, debug=False, frame_sink=None, status_sink=None):
    """the decide loop: one scan per level, then click our priorities off that single snapshot
    before handing the rest to dbd's center auto-spend and advancing.
    dry-run (logs intent, sends no input) whenever click=False; main folds the config dry_run
    default plus the --dry-run / --live / --sim flags into that one flag.
    debug adds the per-scan ocr summary + per-node/per-buy provenance logging (see --debug).
    frame_sink, if given, is called each scan (only while debug and a live frame exist) with
    (annotated_bgr, raw_bgr) — the detect.draw_detections overlay plus the clean grab it was drawn
    from — so a caller like the ui's debug view can show what the loop is seeing (and save both).
    status_sink, if given, is called each live scan with {"prestige","level","bp"} (any may be None),
    so the debug view can show the ocr'd values to sanity-check the threshold/prestige reads.

    every live run also stops itself when the ocr'd bloodpoints can't cover the next pick (or, at
    the level boundary, the cheapest node left), instead of dead-clicking a web it can't afford.

    optional stops and prestige (all live-only, off unless configured):
      stop_bp_threshold  spend down toward a bp floor; on the final level we can't fully afford, buy
                         priorities up to the estimated budget then stop instead of auto-spending.
      auto_prestige      at bloodweb level 50, once the web is consumed the center is the prestige
                         star: hover it to confirm 'PRESTIGE', click it, dismiss the rewards screen,
                         and carry on. a mere detection miss can't trigger it (the hover must confirm).
      stop_prestige /    stop once the character reaches the target (prestige, bloodweb level).
      stop_level

    one scan per level (not per buy): dbd auto-buys the cheapest path to any node, so every node is
    clickable now and positions stay put until the level advances. we walk priorities high to low off
    the snapshot, clicking each and recording its spot in a per-web 'spent' list so the next pick
    skips it. far fewer detect() calls than re-capturing per buy, and avoids detecting mid-animation;
    the trade-off is a node the single scan misses won't be prioritized, but the center auto-spend at
    level end still buys it.

    runs until switch.killed; while not switch.running (idle or paused) it just polls. a stateful
    source may expose consume(node)/advance() to be told of buys and level changes (the sim uses
    them); live_source has neither and relies on the screen changing after the auto-spend."""
    dry_run = not click
    bp_floor = config.get("stop_bp_threshold", 0)  # 0 disables the bp stop
    auto_prestige = bool(config.get("auto_prestige", False))
    stop_prestige = int(config.get("stop_prestige", 0) or 0)
    stop_level = int(config.get("stop_level", 0) or 0)
    cost_table = config.get("rarity_bp_cost") or RARITY_BP_COST
    # timing knobs, all tunable from the settings ui to slow the loop down on a laggy machine.
    settle_s = config.get("settle_s", SETTLE_S)         # post-buy wait before the next pick
    entity_settle_s = config.get("entity_settle_s", ENTITY_SETTLE_S)  # extra wait for the entity smoke to render
    advance_s = config.get("advance_s", ADVANCE_S)      # wait for the level transition after auto-spend
    prestige_wait_s = config.get("prestige_wait_s", PRESTIGE_WAIT_S)  # post-prestige-click wait before the ok button
    hover_s = config.get("ocr_hover_s", ocr.HOVER_DELAY_S)  # tooltip fade-in wait before an ocr read
    consume = getattr(source, "consume", None)     # stateful-source hooks, absent on live_source
    advance = getattr(source, "advance", None)
    # push the tunable detection gates into their modules before the first scan: detect/node read
    # module globals, so a value tuned in the settings ui wouldn't reach them otherwise.
    detect.set_presence_thresh(config.get("presence_thresh", PRESENCE_THRESH_DEFAULT))
    set_rescue_gate(config.get("matcher_rescue_min", RESCUE_MIN_DEFAULT),
                    config.get("matcher_rescue_margin", RESCUE_MARGIN_DEFAULT))
    set_neardup_veto(config.get("matcher_neardup_veto", NEARDUP_VETO_DEFAULT))
    # log the ocr'd status each scan when debugging a threshold/prestige feature, to sanity-check it.
    report_status = debug and (bp_floor or stop_prestige or stop_level or auto_prestige)
    last_center = None   # full-frame (x,y) web center from the last filled scan; the prestige star sits here

    # mirror this run's debug prints to an on-disk log so misreads can be mined across sessions.
    # per-line flush keeps the log intact even if the panic key kills the process mid-scan.
    log_fh = _open_run_log() if debug else None
    saved_stdout = sys.stdout
    if log_fh is not None:
        sys.stdout = _FileTee(saved_stdout, log_fh)
        print(f"# --- spend run {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"({'dry-run' if dry_run else 'LIVE'}, matcher={config.get('matcher', 'ncc')}) ---")

    while not switch.killed:
        if not switch.running:
            time.sleep(IDLE_POLL_S)       # idle or paused: touch nothing, wait for start or kill
            continue

        # one capture + detect for the whole level.
        frame, region, nodes = source()

        # read the ui status (live only), only what a live feature actually needs so a run with no
        # thresholds does no extra ocr. report_status (debug + a feature on) also forces every read so
        # the debug readout can show all three to sanity-check them (feature: the debug bp/level view).
        level = prestige = bp = None
        if frame is not None:
            level = ocr.read_bloodweb_level(frame)   # always: cheap, and gates the level-50 prestige/stop
            if stop_prestige or report_status:
                prestige = ocr.read_prestige_level(frame)
            if bp_floor or report_status or not dry_run:
                # any live run reads bp too (not just the bp-floor stop): it feeds the
                # insufficient-bloodpoints stop below, so a run out of points parks itself
                # instead of re-clicking a web it can't afford forever.
                bp = ocr.read_bp(frame)
            if report_status:
                if status_sink is not None:
                    status_sink({"prestige": prestige, "level": level, "bp": bp})
                print(f"[status] prestige={prestige} level={level} bp={bp}")

        # level/prestige goal-stop: quit once the target (prestige, level) is reached.
        if _reached_level_goal(prestige, level, stop_prestige, stop_level):
            print(f"stop: reached goal (prestige {stop_prestige}, level {stop_level}) "
                  f"at prestige={prestige} level={level}")
            break

        # remember the web center from a filled scan; the prestige star later sits here and the empty
        # prestige screen's own center read is unreliable, so we reuse this good one to hover/click it.
        real_nodes = [n for n in nodes if not n.is_center]
        center_node = next((n for n in nodes if n.is_center), None)
        if real_nodes and center_node is not None:
            last_center = (center_node.x, center_node.y)

        # prestige-ready: at level 50 the filled web has been consumed and the center is now the
        # prestige star. an empty web at 50 is the trigger, but a detection miss also looks empty, so
        # always hover the remembered center and require its tooltip to say PRESTIGE before acting
        # (prestige spends 20k bp and resets the character, and a false stop is a nuisance too).
        if level == PRESTIGE_LEVEL and not real_nodes and frame is not None:
            confirmed = False
            if last_center is not None and not switch.killed:
                txt = ocr.read_center_hover_text(frame, region, last_center, hover_delay_s=hover_s)
                confirmed = "PRESTIGE" in txt
            if confirmed and not auto_prestige:
                print("stop: bloodweb level 50 reached (auto-prestige off)")
                break
            if confirmed:
                if dry_run:
                    print(f"[dry-run] prestige: click star @ {last_center}, dismiss rewards, continue")
                else:
                    print("[prestige] level 50 reached, prestiging")
                    do_prestige(last_center, frame, region, prestige_wait_s=prestige_wait_s,
                                advance_s=advance_s, switch=switch)
                if advance:
                    advance()
            elif debug:
                print("[prestige] level 50 empty web but hover didn't confirm PRESTIGE; waiting")
            _interruptible_sleep(switch, settle_s)
            continue     # rescan (prestiged onto a fresh web, or waiting to confirm)

        # nothing detected (live: anchors unread / bloodweb not visible). do NOT fall through: with
        # zero nodes choose_next returns None and the auto-spend fallback would blind-click the frame
        # center on whatever screen is up. wait and rescan instead.
        if not nodes:
            _interruptible_sleep(switch, settle_s)
            continue

        # bp floor: this level's spend budget. 0 disables. a budget already spent means stop now.
        budget = None
        if bp_floor and bp is not None:
            budget = bp - bp_floor
            if budget <= 0:
                print(f"stop: bloodpoints {bp} at or below floor {bp_floor}")
                break

        if debug:
            _log_scan_summary(nodes)
        nodes = resolve_uncertain(nodes, frame, region, rows, debug=debug, hover_delay_s=hover_s,
                                  weak_match_fallback=config.get("weak_match_fallback", True),
                                  switch=switch)
        if debug and frame is not None:
            annotated = detect.draw_detections(frame, nodes)
            if frame_sink is not None:
                frame_sink(annotated, frame)
            _save_scan_frames(frame, annotated)

        # on the final level the budget can't cover the whole web, so buy priorities up to the budget
        # and stop instead of auto-spending (which would spend the whole web and blow past the floor).
        final_level = budget is not None and budget < estimate_web_cost(nodes, cost_table)
        est_spent = 0

        # click every priority match on this snapshot, high to low, skipping spots already clicked.
        # bp_now tracks the freshest ocr bloodpoint read (re-read off every post-buy grab below),
        # for the insufficient-bloodpoints stop; None whenever the ocr couldn't read it.
        spent = []
        bp_now = bp
        out_of_bp = None   # the pick we couldn't afford, so the stop line can name it
        while switch.running and not switch.killed:
            candidates = [n for n in nodes if not _recently_bought(n, spent)]
            choice = choose_next(candidates, config)
            if choice is None:
                break                     # no priorities left on this web
            if switch.killed:             # panic key landed mid-pick, don't click
                break
            if final_level and est_spent + _node_cost(choice, cost_table) > budget:
                break                     # can't afford more priorities within the bp floor
            if not dry_run and bp_now is not None and bp_now < _node_cost(choice, cost_table):
                out_of_bp = choice        # can't afford the next pick: stop below, don't dead-click
                break
            est_spent += _node_cost(choice, cost_table)
            # margin + runner-up on the buy line are the followup lever;
            # a confident buy with a tiny margin vs a plausible runner-up flags a possible mispick.
            runner = f" vs {choice.runner_up!r}" if (debug and choice.runner_up) else ""
            if dry_run:
                print(f"[dry-run] buy {choice.name!r} "
                      f"({choice.effective_category}/{choice.rarity}) via {choice.resolved_by} "
                      f"s={choice.score:.2f} m={choice.margin:.3f}{runner} "
                      f"@ {choice.x},{choice.y}{_pick_note(choice)}")
            else:
                if debug:
                    print(f"[buy] {choice.name!r} ({choice.effective_category}/{choice.rarity}) "
                          f"via {choice.resolved_by} s={choice.score:.2f} m={choice.margin:.3f}"
                          f"{runner} @ {choice.x},{choice.y}{_pick_note(choice)}")
                sx, sy = capture.frame_to_screen(choice.x, choice.y, region)
                input_control.click_node(sx, sy)
                # park NOW, not just before the grab below: the click leaves the cursor sitting on the
                # node, and parking here lets dbd's tooltip fade DURING the settle wait we already pay
                # rather than adding a fade wait of its own to every single buy.
                if frame is not None:
                    park_cursor(frame, region)
            spent.append((choice.x, choice.y))   # remember the spot so we don't re-pick it
            if consume:
                consume(choice)                  # sim drops it from the web; live no-op
            # the wait covers the buy animation, the tooltip fade, and the entity's smoke animating in:
            # the next thing we do is read state off a fresh grab, so all three have to be done.
            # (dry-run clicks nothing, so none of them apply and it keeps the plain settle.)
            _interruptible_sleep(switch, settle_s if dry_run
                                 else max(settle_s, PARK_FADE_S) + entity_settle_s)

            # that click bought more than the one node we aimed at (dbd auto-buys the whole cheapest
            # path to it) and the entity may have eaten a few more, so re-read the web before the next
            # pick rather than trusting a snapshot that is now several buys out of date. cheap: no
            # re-detect, just a color read at coords we already have (see refresh_states).
            # the grab is clean because we parked above (see park_cursor: a tooltip over the web hides
            # the very rings this reads), and it is AFTER the settle, not before, because mid-animation
            # the ring is only half drawn.
            if not dry_run and frame is not None and not switch.killed:
                frame, region = capture.grab_with_region(region)
                new_bp = ocr.read_bp(frame)   # keep the insufficient-bp stop current as we spend
                if new_bp is not None:
                    bp_now = new_bp
                delta = refresh_states(nodes, frame)
                if delta and debug:
                    # the burst size is the interesting bit: the entity takes 1 node some buys and 6
                    # or 7 others, so log it to build a picture of what the risk actually looks like.
                    print("[state] " + ", ".join(f"+{v} {k}" for k, v in sorted(delta.items())))
                    if frame_sink is not None:   # repaint so the debug view shows what just died
                        frame_sink(detect.draw_detections(frame, nodes), frame)

        if switch.killed:                 # killed: bail before the auto-spend
            break
        if not switch.running:            # paused mid-web: idle without auto-spending, resume rescans
            continue

        # out of bloodpoints mid-web: log both sides of the comparison (the ocr read and what the
        # next pick would have cost) and stop, rather than dead-clicking nodes the game won't sell.
        if out_of_bp is not None:
            print(f"stop: insufficient bloodpoints, can no longer spend "
                  f"(ocr bp {bp_now} < ~{_node_cost(out_of_bp, cost_table)} for next pick "
                  f"{out_of_bp.name!r})")
            break

        # final bp-floor level: we bought the priorities we could afford, now stop rather than
        # auto-spending the rest of the web (which would spend past the floor). leaves ~budget unspent.
        if final_level:
            left = bp - est_spent if bp is not None else None
            print(f"stop: near bp floor {bp_floor}"
                  + (f" (est ~{left} bp left)" if left is not None else "")
                  + ", stopping before auto-spend")
            break

        # same stop at the level boundary: if we can't afford even the cheapest remaining node, the
        # center auto-spend can't buy anything either, and every rescan would land right back here.
        if not dry_run and bp_now is not None:
            cheapest = min((_node_cost(n, cost_table) for n in nodes
                            if not n.is_center and not n.taken), default=None)
            if cheapest is not None and bp_now < cheapest:
                print(f"stop: insufficient bloodpoints, can no longer spend "
                      f"(ocr bp {bp_now} < ~{cheapest} for the cheapest node left on the web)")
                break

        # priorities exhausted: hand the rest of the web to dbd's center auto-spend, then advance.
        if dry_run:
            print("[dry-run] auto-spend center, advance level")
        else:
            if debug:
                print("[auto-spend] no priorities left, clicking center to finish the web")
            do_auto_spend(nodes, frame, region, advance_s=advance_s, switch=switch)
        if advance:
            advance()                     # sim draws the next level; live no-op
        _interruptible_sleep(switch, settle_s)

    if log_fh is not None:                # loop exited (kill or bp-stop): restore stdout, close log
        sys.stdout = saved_stdout
        log_fh.close()
        print(f"[log] debug output appended to {log_fh.name}")


def main():
    ap = argparse.ArgumentParser(description="dbd bloodweb priority auto-spender")
    ap.add_argument("--sim", action="store_true",
                    help="use the offline simulator instead of live capture + detect")
    ap.add_argument("--dry-run", action="store_true",
                    help="log intended clicks, send no input")
    ap.add_argument("--live", action="store_true",
                    help="actually click in-game, overriding the config dry_run safety default")
    ap.add_argument("--config", default=None, help="path to priority.json (default: config/priority.json)")
    ap.add_argument("--matcher", default=None,
                    help="detect matcher: cnn | ncc | ncc_masked | phash (overrides config; default cnn)")
    ap.add_argument("--debug", action="store_true",
                    help="verbose loop logging: per-scan ocr summary, why each node routes to ocr, "
                         "the ocr read result, and each buy's provenance (match vs ocr)")
    ap.add_argument("--no-crop", action="store_true",
                    help="disable bloodweb auto-crop, detect on the full frame (debugging)")
    args = ap.parse_args()

    config = load_config(args.config)
    rows, _ = detect.load_index()

    # detect knobs come from config so the settings ui owns them; --matcher / --debug still override.
    matcher = args.matcher or config.get("matcher", "cnn")
    debug = args.debug or config.get("debug", False)
    thresh_method = config.get("thresh_method", "adaptive_gaussian")
    use_hough = config.get("node_finder", "contours") == "hough"

    # safety: stay in dry-run unless explicitly sent live. --sim and --dry-run are always dry,
    # --live forces real clicks, otherwise fall back to the config dry_run (defaults True).
    if args.sim or args.dry_run:
        dry_run = True
    elif args.live:
        dry_run = False
    else:
        dry_run = config.get("dry_run", True)

    # keys come from config when present so the settings ui can rebind them, else the defaults.
    switch = Switch(
        start_key=config.get("start_key", START_KEY),
        kill_key=config.get("kill_key", KILL_KEY),
    ).arm()
    mode = "DRY-RUN, no clicks" if dry_run else "LIVE, will click in-game"
    print(f"armed (idle, {mode}{', DEBUG' if debug else ''}): press {switch.start_key!r} "
          f"to start/pause, {switch.kill_key!r} to stop")

    if args.sim:
        # quick lil bloodweb simulator because node matching in detect.py is giving me a brain hemorrhage
        # and no, I didn't have to google how to spell hemorrhage (I did)
        source = sim_source(rows, seed=0, low_conf_frac=0.2, discrepancy_frac=0.1)
    else:
        ncc_templates = detect.load_ncc_templates(rows) if matcher.startswith("ncc") else None
        # narrow the match library to the priority list's icons/sources, snapshotted here (not
        # rebuilt mid-run). None = no narrowing (compare against the whole library).
        row_pool = build_pool_mask(
            rows, config.get("priorities", []),
            inferred=config.get("pool_inferred", True),
            exclusive=config.get("pool_exclusive", False),
        )
        if row_pool is not None:
            print(f"pool: matching against {sum(row_pool)}/{len(rows)} library icons "
                  f"({'priority-only' if config.get('pool_exclusive') else 'priority-inferred'})")
        source = live_source(
            rows, ncc_templates, matcher=matcher, thresh_method=thresh_method, use_hough=use_hough,
            auto_crop=config.get("auto_crop", True) and not args.no_crop,
            web_bbox=config.get("web_bbox"),
            crop_pad_frac=config.get("crop_pad_frac", ocr.CROP_PAD_FRAC),
            debug=debug, row_pool=row_pool,
        )
    run(source, config, switch, rows, click=not dry_run, debug=debug)


if __name__ == "__main__":
    main()
