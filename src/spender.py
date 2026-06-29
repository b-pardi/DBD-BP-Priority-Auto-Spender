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
import random
import ctypes
from ctypes import wintypes
from pathlib import Path

from . import capture, detect, input_control, ocr
from .node import Node
from .sim import sim_source

KILL_KEY = "f8"      # dedicated always-stop panic hotkey, only ever stops, never resumes
START_KEY = "f7"     # start/pause toggle, idle -> running -> paused -> running
SETTLE_S = 0.6       # post-buy wait before the rescan, tune live like input_control.hold_s
IDLE_POLL_S = 0.05   # how often the loop re-checks the switch while idle or paused
REPICK_TOL_PX = 20   # spot-match tolerance for the don't-repick guard, tune with node spacing

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "priority.json"
VALID_CATEGORIES = {"item", "addon", "offering", "perk", "power"}
VALID_RARITIES = {"common", "uncommon", "rare", "very rare", "ultra rare", "event"}
# win32 global-hotkey plumbing for the run control.
# RegisterHotKey reserves the key system-wide and posts WM_HOTKEY to our own message loop, so it
# fires no matter which window has focus, including once dbd is foreground and we start clicking.
# the old keyboard-lib low-level hook only landed while the terminal had focus (a fullscreen game
# can swallow hook keys). caveat: if dbd runs elevated (as admin), launch this script elevated too
# or windows blocks the keys from the lower-integrity process (UIPI).
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
    one class instead of two because the states are coupled: start toggles running<->paused,
    while kill force-idles from anywhere and latches, so the panic key only ever stops.
    arm() starts the listener and returns self, leaving the control idle; the loop polls."""

    def __init__(self, start_key=START_KEY, kill_key=KILL_KEY):
        self.start_key = start_key
        self.kill_key = kill_key
        self._running = threading.Event()   # set = running, clear = idle/paused
        self._killed = threading.Event()    # set = hard stop latched, terminal

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

    @property
    def running(self):
        """is the loop cleared to do work this iteration? false while idle, paused, or killed."""
        return self._running.is_set() and not self._killed.is_set()

    @property
    def killed(self):
        """has a hard stop been latched? the loop's outer exit condition."""
        return self._killed.is_set()


def load_config(path=None):
    """read and lightly validate the priority config.
    returns the parsed dict {dry_run, stop_bp_threshold, priorities: [rule, ...]}.
    raises ValueError on a malformed rule, so a bad config fails fast at startup instead
    of silently never matching mid-run."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    for i, rule in enumerate(cfg.get("priorities", [])):
        kind = rule.get("type")
        if kind == "item":
            if not rule.get("name"):
                raise ValueError(f"priority[{i}]: item rule needs a 'name'")
        elif kind == "category":
            if rule.get("category") not in VALID_CATEGORIES:
                raise ValueError(
                    f"priority[{i}]: category must be one of {sorted(VALID_CATEGORIES)}"
                )
        else:
            raise ValueError(f"priority[{i}]: type must be 'item' or 'category', got {kind!r}")
        rarity = rule.get("rarity")
        if rarity is not None and rarity not in VALID_RARITIES:
            raise ValueError(f"priority[{i}]: bad rarity {rarity!r}")

    cfg.setdefault("dry_run", True)
    cfg.setdefault("stop_bp_threshold", 0)
    return cfg


# node sources from live capture
def live_source(rows, ncc_templates, region=None, matcher="ncc"):
    """build a () -> (frame, region, nodes) source backed by the real screen + detect()."""
    def _source():
        frame, reg = capture.grab_with_region(region)
        dets = detect.detect(frame, rows=rows, ncc_templates=ncc_templates, matcher=matcher)
        nodes = [Node.from_detection(d) for d in dets]
        return frame, reg, nodes
    return _source


def resolve_uncertain(nodes, frame, region, rows):
    """settle identity for nodes flagged needs_resolution via an ocr hover scan.
    a node is trusted when its observed reads and matched icon agree and the match is confident.
    otherwise we hover it and read the name tooltip rather than guess.
    live only, so skip entirely when frame is None (the sim path, nodes stay as detected).
    """
    if frame is None:
        return nodes
    for n in nodes:
        if n.needs_resolution:
            ocr.find_node_tooltip(n, frame, region, rows) # mutates n, may set resolved_by='ocr'
    return nodes


def choose_next(nodes, config):
    """return the highest-priority Node to buy, or None for center auto-spend.
    walks the ordered rules high to low and returns a random node among those matching the first
    rule that hits (random tie-break by design, inner/cheaper nodes tend to be lower quality).
    returns one node, the loop rescans after each buy; the don't-repick guard lives in the loop
    so this stays a pure function of the current node list.
    """
    for rule in config.get("priorities", []):
        hits = [n for n in nodes if n.matches(rule)]
        if hits:
            return random.choice(hits)
    return None


def do_auto_spend(nodes, frame, region, hold_s=0.05, advance_s=3.0):
    """no priority left in this web, hand it to dbd's center entity node to finish the level.
    clicking the center auto-completes the remaining buys and advances to the next web.
    targets the is_center node detect tagged by its red glow (find_center_node), falling back to
    the frame center if it wasn't found.
    waits advance_s for the fill + level transition to play out before returning, so the loop's
    next scan sees the fresh web (detecting a reward/prestige screen instead is a later concern).
    hold_s/advance_s are tuned live like the input_control timings; raise hold_s if dbd needs a
    real press-and-hold to auto-fill rather than a tap.
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
    time.sleep(advance_s)


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


def run(source, config, switch, rows, click=True):
    """the decide loop: one scan per level, then click our priorities off that single snapshot
    before handing the rest to dbd's center auto-spend and advancing.
    dry-run (logs intent, sends no input) whenever click=False; main folds the config dry_run
    default plus the --dry-run / --live / --sim flags into that one flag.

    one scan per level (not per buy): dbd auto-buys the cheapest path to any node, so every node is
    clickable now and node positions stay put until the level advances. we walk our priorities high
    to low off the snapshot, clicking each and recording its spot in a per-web 'spent' list so the
    next pick skips it (a node dbd already consumed as part of a path just clicks a harmless no-op).
    that's far fewer detect() calls than re-capturing per buy and avoids detecting the web
    mid-animation; the trade-off is that a node the single scan misses won't be prioritized, but
    the center auto-spend at level end still buys it.

    runs until switch.killed; while not switch.running (idle or paused) it just polls. a stateful
    source may expose consume(node)/advance() to be told of buys and level changes (the sim uses
    them); live_source has neither and relies on the screen changing after the auto-spend."""
    dry_run = not click
    bp_floor = config.get("stop_bp_threshold", 0)  # 0 disables the bp stop
    consume = getattr(source, "consume", None)     # stateful-source hooks, absent on live_source
    advance = getattr(source, "advance", None)
    while not switch.killed:
        if not switch.running:
            time.sleep(IDLE_POLL_S)       # idle or paused: touch nothing, wait for start or kill
            continue

        # one capture + detect for the whole level.
        frame, region, nodes = source()

        # optional stop: quit once the live bloodpoint total falls to the configured floor.
        # checked per level here (per scan), so a stop can lag the floor by up to one level's spend.
        if bp_floor and frame is not None:
            bp = ocr.read_bp(frame)
            if bp is not None and bp <= bp_floor:
                print(f"stop: bloodpoints {bp} at or below floor {bp_floor}")
                break

        nodes = resolve_uncertain(nodes, frame, region, rows)

        # click every priority match on this snapshot, high to low, skipping spots already clicked.
        spent = []
        while switch.running and not switch.killed:
            candidates = [n for n in nodes if not _recently_bought(n, spent)]
            choice = choose_next(candidates, config)
            if choice is None:
                break                     # no priorities left on this web
            if switch.killed:             # panic key landed mid-pick, don't click
                break
            if dry_run:
                print(f"[dry-run] buy {choice.name!r} "
                      f"({choice.effective_category}/{choice.rarity}) @ {choice.x},{choice.y}")
            else:
                sx, sy = capture.frame_to_screen(choice.x, choice.y, region)
                input_control.click_node(sx, sy)
            spent.append((choice.x, choice.y))   # remember the spot so we don't re-pick it
            if consume:
                consume(choice)                  # sim drops it from the web; live no-op
            _interruptible_sleep(switch, SETTLE_S)  # let the buy animation play before the next

        if switch.killed:                 # killed: bail before the auto-spend
            break
        if not switch.running:            # paused mid-web: idle without auto-spending, resume rescans
            continue

        # priorities exhausted: hand the rest of the web to dbd's center auto-spend, then advance.
        if dry_run:
            print("[dry-run] auto-spend center, advance level")
        else:
            do_auto_spend(nodes, frame, region)
        if advance:
            advance()                     # sim draws the next level; live no-op
        _interruptible_sleep(switch, SETTLE_S)


def main():
    ap = argparse.ArgumentParser(description="dbd bloodweb priority auto-spender")
    ap.add_argument("--sim", action="store_true",
                    help="use the offline simulator instead of live capture + detect")
    ap.add_argument("--dry-run", action="store_true",
                    help="log intended clicks, send no input")
    ap.add_argument("--live", action="store_true",
                    help="actually click in-game, overriding the config dry_run safety default")
    ap.add_argument("--config", default=None, help="path to priority.json (default: config/priority.json)")
    ap.add_argument("--matcher", default="ncc", help="detect matcher: ncc | ncc_masked | phash")
    args = ap.parse_args()

    config = load_config(args.config)
    rows, _ = detect.load_index()

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
    print(f"armed (idle, {mode}): press {switch.start_key!r} to start/pause, "
          f"{switch.kill_key!r} to stop")

    if args.sim:
        # quick lil bloodweb simulator because node matching in detect.py is giving me a brain hemorrhage
        # and no, I didn't have to google how to spell hemorrhage (I did)
        source = sim_source(rows, seed=0, low_conf_frac=0.2, discrepancy_frac=0.1)
    else:
        ncc_templates = detect.load_ncc_templates(rows) if args.matcher.startswith("ncc") else None
        source = live_source(rows, ncc_templates, matcher=args.matcher)
    run(source, config, switch, rows, click=not dry_run)


if __name__ == "__main__":
    main()
