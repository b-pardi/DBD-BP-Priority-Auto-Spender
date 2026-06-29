"""main loop: capture -> detect -> resolve -> decide -> click -> rescan, with a start/kill switch.

re-scans between buys because the web changes as you spend (and as the entity eats nodes).
dry-run mode logs what it WOULD click and sends no input, for safe testing, and pairs with
--sim to exercise the whole decide step offline with no game open. run control = two global
hotkeys, a start/pause toggle (launches idle, no capture or input until pressed) and a dedicated
kill (other stop conditions, bp threshold etc., come later).

the loop is source-agnostic: a source is a callable returning (frame, region, nodes), where
frame/region are None in sim. live builds nodes from capture + detect, sim from the simulator,
and both feed the identical resolve + choose_next path.
"""



import argparse
import threading
import time
import json
import random
from pathlib import Path

import keyboard

from . import capture, detect, input_control, ocr
from .node import Node
from .sim import simulate_level

KILL_KEY = "f8"      # dedicated always-stop panic hotkey, only ever stops, never resumes
START_KEY = "f7"     # start/pause toggle, idle -> running -> paused -> running
SETTLE_S = 0.6       # post-buy wait before the rescan, tune live like input_control.hold_s
IDLE_POLL_S = 0.05   # how often the loop re-checks the switch while idle or paused

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "priority.json"
VALID_CATEGORIES = {"item", "addon", "offering", "perk", "power"}
VALID_RARITIES = {"common", "uncommon", "rare", "very rare", "ultra rare", "event"}
class Switch:
    """global-hotkey run control: a start/pause toggle plus a dedicated kill, both backed by
    threading.Events the loop polls cheaply while keyboard's listener thread flips them.

    three states the loop reads off .running / .killed:
      idle     launch default, armed but doing nothing (no capture, no input)
      running  start pressed, the loop does its work
      paused   start pressed again, the loop idles but stays ready to resume
    one class instead of two because the states are coupled: start toggles running<->paused,
    while kill force-idles from anywhere and latches, so the panic key only ever stops.
    arm() registers both hotkeys and returns self, leaving the control idle; the loop polls."""

    def __init__(self, start_key=START_KEY, kill_key=KILL_KEY):
        self.start_key = start_key
        self.kill_key = kill_key
        self._running = threading.Event()   # set = running, clear = idle/paused
        self._killed = threading.Event()    # set = hard stop latched, terminal

    def arm(self):
        keyboard.add_hotkey(self.start_key, self.toggle)
        keyboard.add_hotkey(self.kill_key, self.kill)
        return self

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


# node sources from simulation ('perfect detection')
def sim_source(rows, **kw):
    """build a () -> (frame, region, nodes) source backed by the offline simulator (no game).
    frame and region are None, which makes the loop skip the live-only ocr hover step."""
    def _source():
        return None, None, simulate_level(rows, **kw)
    return _source


def resolve_uncertain(nodes, frame, region, rows):
    """CRUX (pseudocode): settle identity for nodes flagged needs_resolution via ocr hover.

    a node is trusted when its observed reads (disk rarity, socket shape),
    and its matched icon agree and the match is confident.
    otherwise we hover it and read the name tooltip rather than guess.
    live-only, so skip entirely when frame is None (sim path).

        if frame is None:
            return nodes                       # sim: nodes are already as resolved as they get
        for n in nodes:
            if n.needs_resolution:
                ocr.resolve_by_hover(n, frame, region, rows)   # mutates n, sets resolved_by='ocr'
        return nodes
    """
    return nodes  # passthrough until ocr.resolve_by_hover is built


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


def run(source, config, switch, rows, click=True):
    """the decide loop. structural scaffolding; the per-iteration brain is choose_next +
    resolve_uncertain + do_auto_spend (the pseudocoded crux). click=False forces dry-run
    regardless of config (used by --dry-run and always by --sim).
    runs until switch.killed; while not switch.running (idle or paused) it just polls, so
    start/pause gates the work without tearing the thread down."""
    dry_run = config.get("dry_run", True) or not click
    while not switch.killed:
        if not switch.running:
            time.sleep(IDLE_POLL_S)      # idle or paused: touch nothing, wait for start or kill
            continue

        frame, region, nodes = source()
        nodes = resolve_uncertain(nodes, frame, region, rows)
        choice = choose_next(nodes, config)

        if switch.killed:                # panic key hit mid-iteration, bail before any input
            break

        if choice is None:
            # nothing we want left here, hand the rest of the web to dbd's auto-spend.
            if dry_run:
                print("[dry-run] auto-spend center, advance level")
            else:
                do_auto_spend(nodes, frame, region)
            time.sleep(SETTLE_S)
            continue

        if dry_run:
            print(f"[dry-run] buy {choice.name!r} "
                  f"({choice.effective_category}/{choice.rarity}) @ {choice.x},{choice.y}")
        else:
            sx, sy = capture.frame_to_screen(choice.x, choice.y, region)
            input_control.click_node(sx, sy)
        time.sleep(SETTLE_S)


def main():
    ap = argparse.ArgumentParser(description="dbd bloodweb priority auto-spender")
    ap.add_argument("--sim", action="store_true",
                    help="use the offline simulator instead of live capture + detect")
    ap.add_argument("--dry-run", action="store_true",
                    help="log intended clicks, send no input")
    ap.add_argument("--config", default=None, help="path to priority.json (default: config/priority.json)")
    ap.add_argument("--matcher", default="ncc", help="detect matcher: ncc | ncc_masked | phash")
    args = ap.parse_args()

    config = load_config(args.config)
    rows, _ = detect.load_index()

    # keys come from config when present so the settings ui can rebind them, else the defaults.
    switch = Switch(
        start_key=config.get("start_key", START_KEY),
        kill_key=config.get("kill_key", KILL_KEY),
    ).arm()
    print(f"armed (idle): press {switch.start_key!r} to start/pause, {switch.kill_key!r} to stop")

    if args.sim:
        # quick lil bloodweb simulator because node matching in detect.py is giving me a brain hemorrhage
        # and no, I didn't have to google how to spell hemorrhage (I did)
        source = sim_source(rows, seed=0, low_conf_frac=0.2, discrepancy_frac=0.1)
        run(source, config, switch, rows, click=False)
    else:
        ncc_templates = detect.load_ncc_templates(rows) if args.matcher.startswith("ncc") else None
        source = live_source(rows, ncc_templates, matcher=args.matcher)
        run(source, config, switch, rows, click=not args.dry_run)


if __name__ == "__main__":
    main()
