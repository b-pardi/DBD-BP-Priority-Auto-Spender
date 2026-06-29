"""offline bloodweb simulator for testing the priority loop without the game.

produces the same Node list the live path produces (detect -> Node.from_detection), so
spender's loop is source-agnostic: swap the live source for this and run the whole decide
step under dry-run with no game open. it pulls real rows from the icon index, so names,
categories, and rarities are realistic and exercise actual priority rules.

the bloodweb graph is intentionally NOT modeled: dbd auto-buys the cheapest path to any
clicked node, so detect() returns a flat node list and so do we (no edges, no reachability).
"""

import random

from .detect import load_index, NODE_SHAPE_DICT, RARITIES
from .node import Node

# invert NODE_SHAPE_DICT: category -> the socket shape it presents, for consistent fake nodes.
# categories with no node shape (e.g. power) are absent, so we can filter them out of the draw.
CATEGORY_SHAPE = {cat: shape for shape, cats in NODE_SHAPE_DICT.items() for cat in cats}


def simulate_level(
        rows=None,
        n=12,
        seed=None,
        force=(),
        low_conf_frac=0.0,
        discrepancy_frac=0.0,
        frame_w=3440,
        frame_h=1440,
    ):
    """build a fake bloodweb level as a list[Node].

    n: number of nodes on the web.
    seed: fixes the rng for reproducible runs.
    force: iterable of index KEYS to guarantee present, so a specific priority rule is sure
        to hit (e.g. force=("CommodiousToolbox",)).
    low_conf_frac: fraction of nodes given a weak match score, which routes them to the ocr
        fallback (needs_resolution via low confidence).
    discrepancy_frac: fraction whose matched rarity is perturbed to disagree with the observed
        disk, which also routes them to the ocr fallback (needs_resolution via misalignment).
    positions are random within the frame since the graph is not modeled (see module docstring).
    """
    rng = random.Random(seed)
    if rows is None:
        rows, _ = load_index()

    # seed the level with any forced keys, then fill with random real rows that map to a node.
    chosen = []
    for key in force:
        row = next((r for r in rows if r['key'] == key), None)
        if row is None:
            raise KeyError(f"simulate_level: forced key {key!r} not in index")
        chosen.append(row)
    pool = [r for r in rows if r['category'] in CATEGORY_SHAPE]  # drop powers etc.
    while len(chosen) < n:
        chosen.append(rng.choice(pool))

    nodes = []
    for row in chosen[:n]:
        # observed disk rarity: use the row's known rarity, else a random non-event tier.
        obs_rarity = row['rarity'] or rng.choice(RARITIES[:-1])
        shape = CATEGORY_SHAPE[row['category']]

        # baseline: a confident match whose attrs agree with the observed reads.
        score, margin = 0.7, 0.1
        match = dict(row)

        # inject weak matches to exercise the low-confidence ocr path.
        if rng.random() < low_conf_frac:
            score, margin = 0.2, 0.01

        # inject attr disagreement to exercise the discrepancy ocr path.
        if rng.random() < discrepancy_frac:
            others = [r for r in RARITIES if r != obs_rarity]
            match = dict(row, rarity=rng.choice(others))  # matched rarity now != disk -> misaligned

        nodes.append(Node(
            x=rng.randint(0, frame_w), y=rng.randint(0, frame_h), r=40,
            rarity=obs_rarity, socket_shape=shape,
            name=match['name'], match=match,
            score=score, margin=margin, matcher='ncc',
        ))
    return nodes


class SimSource:
    """stateful offline source: owns one fake web, shrinks it as nodes are bought, and draws a
    fresh level on advance, so spender's loop actually progresses under --sim instead of rescanning
    the same web forever (a constant seed rebuilding the identical web was the original repeat bug).
    callable like spender.live_source (returns (None, None, nodes) so the loop skips the live-only
    ocr + bp steps); the loop also calls consume(node)/advance() to report buys and level changes.
    the live source needs neither: the screen is its state.
    """

    def __init__(self, rows, seed=0, **kw):
        self.rows = rows
        self.kw = kw          # simulate_level knobs (n, low_conf_frac, discrepancy_frac, ...)
        self.seed = seed      # stepped per level so consecutive webs differ
        self.level = 0
        self.nodes = []
        self._draw()

    def _draw(self):
        # step the seed by the level so each web differs but the run stays reproducible.
        s = self.seed if self.seed is None else self.seed + self.level
        self.nodes = simulate_level(self.rows, seed=s, **self.kw)

    def __call__(self):
        # copy the list so the caller can't mutate our web; the Node objects stay shared so
        # consume() can match the bought one by identity.
        return None, None, list(self.nodes)

    def consume(self, node):
        # a node was bought: drop it from the current web (same objects we handed out).
        self.nodes = [n for n in self.nodes if n is not node]

    def advance(self):
        # priorities exhausted, dbd auto-spends the rest: model that as moving to the next level.
        self.level += 1
        self._draw()


def sim_source(rows, **kw):
    """build a stateful () -> (frame, region, nodes) source backed by the offline simulator."""
    return SimSource(rows, **kw)
