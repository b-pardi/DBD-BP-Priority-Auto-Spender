"""semantic node model for the priority engine.

detect() emits raw per-node dicts full of detection internals (the glyph image, the
phash/ncc score, socket geometry). the priority + spend algorithm only cares about
semantic attributes: where to click, what the node IS, and how much to trust that read.
Node is that semantic view, built once at the boundary via from_detection, so the rest
of the algorithm never touches detection dicts (and the offline simulator can build the
exact same objects without a frame).

Node also owns the observed-vs-matched reconciliation. rarity comes from the disk color
(authoritative live read) and category from the matched icon (the only thing that can
split item from addon, since both are square sockets). when those two sources disagree,
or the icon match is weak, the node is flagged needs_resolution so the loop can fall back
to an ocr tooltip-hover scan to settle identity (see ocr.find_node_tooltip). that fallback
replaces a pure confidence gate: both weak matches and observed/matched misalignment get
routed to ocr instead of being guessed at.
"""

from dataclasses import dataclass, field
import re

# socket geometry -> the game categories that shape can be (mirror of detect.NODE_SHAPE_DICT).
# square can't split item vs addon by geometry alone, so that needs the icon match or ocr.
SHAPE_CATEGORIES = {
    'square': ('item', 'addon'),
    'rhombus': ('perk',),
    'hexagon': ('offering',),
}

# confidence thresholds per matcher, score direction differs (see detect.MATCHERS).
# ncc is cosine (higher=better), phash is hamming distance (lower=better).
# placeholders from the matcher eval (ncc real conf ~0.5-0.84), tune against the fixtures.
NCC_CONF_MIN = 0.45
NCC_MARGIN_MIN = 0.03
PHASH_MAX_HAM = 10
PHASH_MARGIN_MIN = 2


def normalize_name(s):
    """fold a name to lowercase alphanumerics for tolerant matching.
    mirrors the scraper's name-key convention so a rule's "Commodious Toolbox" matches the
    index name regardless of spacing, case, or punctuation.
    returns '' for None or empty."""
    if not s:
        return ''
    return re.sub(r'[^a-z0-9]', '', s.lower())


@dataclass
class Node:
    # click target in detection-frame coords (map to screen via capture.frame_to_screen).
    x: int
    y: int
    r: int

    # observed reads, straight off the frame (rarity is the authoritative live read).
    rarity: str            # disk-color rarity, one of detect.RARITIES
    socket_shape: str      # 'square' | 'rhombus' | 'hexagon'

    is_center: bool = False # center auto-spend node, found by red glow; never a buy target

    # matched-icon reads, from the library row the matcher picked.
    name: str = None       # matched icon display name
    match: dict = None     # raw library row {key,name,category,rarity,...} or None
    score: float = 0.0
    margin: float = 0.0
    matcher: str = 'ncc'

    # provenance of the final identity, the loop sets this to 'ocr' after a hover scan.
    resolved_by: str = 'match'   # 'match' | 'ocr'

    # detection internals, kept optional for debug/ocr only, never used by the algorithm.
    glyph_bgr: object = field(default=None, repr=False)

    @classmethod
    def from_detection(cls, d):
        """adapt one detect() result dict into a Node.
        detect()'s 'cat' key is the socket SHAPE, not a game category, so it maps to
        socket_shape here (the real category is derived from the match)."""
        m = d.get('match') or None
        return cls(
            x=int(d['x']), y=int(d['y']), r=int(d['r']),
            rarity=d['rar'],
            socket_shape=d['cat'],
            is_center=(d.get('kind') == 'center'),
            name=(m.get('name') if m else None),
            match=m,
            score=float(d.get('score', 0.0)),
            margin=float(d.get('margin', 0.0)),
            matcher=d.get('matcher', 'ncc'),
            glyph_bgr=d.get('glyph_bgr'),
        )

    # identity reconciliation 
    @property
    def matched_category(self):
        """game category from the matched row (item/addon/perk/offering/power), or None."""
        return self.match.get('category') if self.match else None

    @property
    def matched_rarity(self):
        """wiki rarity from the matched row, often None (a soft cross-check only)."""
        return self.match.get('rarity') if self.match else None

    @property
    def shape_categories(self):
        """the categories this socket shape is allowed to be."""
        return SHAPE_CATEGORIES.get(self.socket_shape, ())

    @property
    def confident(self):
        """is the icon match strong enough to trust on its own?
        direction depends on the matcher (ncc higher=better, phash lower=better)."""
        if self.match is None:
            return False
        if self.matcher == 'phash':
            return self.score <= PHASH_MAX_HAM and self.margin >= PHASH_MARGIN_MIN
        return self.score >= NCC_CONF_MIN and self.margin >= NCC_MARGIN_MIN  # ncc / ncc_masked

    @property
    def category_agrees(self):
        """does the matched category fall within what the socket shape allows?
        always true under the current pooled detect(), but becomes a real discrepancy
        signal once detect() stops pooling and matches against the full library."""
        if self.matched_category is None:
            return False
        return self.matched_category in self.shape_categories

    @property
    def rarity_agrees(self):
        """does the matched wiki rarity line up with the observed disk rarity?
        wiki rarity is null for many rows, so treat unknown as 'no disagreement'."""
        if self.matched_rarity is None:
            return True
        return self.matched_rarity == self.rarity

    @property
    def needs_resolution(self):
        """should the loop fall back to an ocr hover scan to settle identity?
        trigger when the icon match is weak OR the observed and matched attrs disagree.
        this is the replacement for a confidence gate: both routes go to ocr, not a guess."""
        return (not self.confident) or (not self.category_agrees) or (not self.rarity_agrees)

    @property
    def effective_category(self):
        """best category estimate for rule matching.
        the matched row is the only source that splits item from addon, so trust it only
        when the match is trustworthy (confident or ocr-settled). otherwise fall back to
        the shape: perk/offering resolve from shape alone, a square stays ambiguous (None)."""
        if (self.confident or self.resolved_by == 'ocr') and self.matched_category:
            return self.matched_category
        cats = self.shape_categories
        return cats[0] if len(cats) == 1 else None

    @property
    def effective_rarity(self):
        """rarity for rule matching, always the observed disk read."""
        return self.rarity


    def matches(self, rule):
        """does this node satisfy one priority rule?
        rule is {"type":"item","name",...} or {"type":"category","category",...} with an optional
        "rarity" filter, compared against effective_rarity (the authoritative disk read).
        item rules demand trustworthy identity (a confident match or ocr-settled) then a name match.
        category rules lean on effective_category (shape-derived for perk/offering,
        matched-row for item/addon, None for an untrustworthy square).
        by here the loop has already ocr-resolved the needs_resolution nodes.
        """
        if self.is_center:
            return False # auto-spend node, the loop handles it, never a buy target

        rarity = rule.get("rarity")
        if rarity is not None and rarity != self.effective_rarity:
            return False

        if rule.get("type") == "item":
            if not (self.confident or self.resolved_by == "ocr"):
                return False # must BE this icon, so demand a trustworthy read first
            return normalize_name(self.name) == normalize_name(rule["name"])
        if rule.get("type") == "category":
            return self.effective_category == rule["category"]
        return False
