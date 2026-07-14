"""semantic node model for the priority engine.

detect() emits raw per-node dicts of detection internals (glyph image, phash/ncc score,
socket geometry); Node is the semantic view the spend algorithm wants (where to click, what
the node is, how much to trust that read), built once at the boundary via from_detection so
the rest of the algorithm never touches detection dicts (and the sim builds the same objects
without a frame).

Node also owns observed-vs-matched reconciliation. rarity comes from the disk color (the
authoritative live read), category from the matched icon (the only thing that splits item
from addon, both square sockets). when the two disagree or the match is weak, the node is
flagged needs_resolution so the loop falls back to an ocr tooltip-hover scan to settle
identity (see ocr.find_node_tooltip), instead of guessing behind a plain confidence gate.
"""

from dataclasses import dataclass, field
import re
import unicodedata

# socket geometry -> the game categories that shape can be (mirror of detect.NODE_SHAPE_DICT).
# square can't split item vs addon by geometry alone, so that needs the icon match or ocr.
SHAPE_CATEGORIES = {
    'square': ('item', 'addon'),
    'rhombus': ('perk',),
    'hexagon': ('offering',),
}

# confidence thresholds per matcher; score direction differs (see detect.MATCHERS).
# ncc/cnn are cosine (higher=better), phash is hamming distance (lower=better).
# ncc gate is a placeholder from the matcher eval (real conf ~0.5-0.84), tune vs fixtures.
NCC_CONF_MIN = 0.45
# cnn cosine gate calibrated on the real independent labels (eval_matchers cnneval + sweep): 0.65 is
# ~80% coverage at ~96% precision, so confident cnn matches skip ocr and the near-dup slippers
# (wornOutTools, skeletonKey) fall below it and route to ocr.
CNN_CONF_MIN = 0.65
PHASH_MAX_HAM = 10
# cnn margin RESCUE. gating ON the margin was dropped 2026-06-29 because it rejected correct
# near-ties; it earns its keep the other way round. real top5 is 98.6% vs top1 83.1%, i.e. the model's
# misses ARE near-ties, so a decisive runner-up gap is strong evidence even at a middling cosine.
# purely additive: it can only pull nodes OFF the ocr path, never onto it.
# the cap is because detect scores pool-excluded rows -2.0, so a pool narrowed to ONE candidate
# reports a nonsense margin (score + 2.0) with no real runner-up behind it; real margins sit far below.
# NOT CALIBRATED -- re-sweep against the real labels (eval_matchers cnneval) after the retrain.
CNN_RESCUE_MIN = 0.55
CNN_RESCUE_MARGIN = 0.15
CNN_MARGIN_CAP = 1.0
# ncc/phash margin floors: logged only, never gated (see runner_up).
NCC_MARGIN_MIN = 0.03
PHASH_MARGIN_MIN = 2


def tier_rules(tier):
    """the rule dicts of one priority tier, tolerating either config shape.
    canonically a tier is {"rules": [...], "ordered": bool}, but an old/migrated/hand-edited file may
    store a bare list of rules; both read the same here so the engine needn't care which it got."""
    if isinstance(tier, dict):
        return tier.get("rules") or []
    if isinstance(tier, list):
        return tier
    return []


def tier_is_ordered(tier):
    """is this tier in within-tier ordered mode (prefer the earliest matching rule) rather than the
    default random pick? a bare-list tier is always unordered (the pre-feature behavior)."""
    return bool(tier.get("ordered")) if isinstance(tier, dict) else False


def normalize_name(s):
    """fold a name to lowercase alphanumerics for tolerant matching.
    mirrors the scraper's name-key convention so a rule's "Commodious Toolbox" matches the
    index name regardless of spacing, case, or punctuation.
    accents fold to their base letter first: stripping non-ascii outright would drop the letter
    entirely, so 'Déjà Vu' folded to 'djvu' and 'Zōri' to 'zri' and neither could ever match its own
    wiki page (which is why those rows carried no rarity or description).
    returns '' for None or empty."""
    if not s:
        return ''
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]', '', s.lower())


def row_names(row):
    """every name a library row answers to, folded: its canonical name plus its aliases.

    a row's aliases are the wiki's redirects to its article plus whatever we called it before that
    article was resolved, so an old config rule ("Decisive Strike"), the community shorthand ("DS")
    and the current name ("Will to Live") all land on the same row. an index predating the alias
    scrape has no aliases field and simply answers to its one name."""
    if not row:
        return set()
    names = [row.get('name')] + list(row.get('aliases') or [])
    return {n for n in (normalize_name(x) for x in names) if n}


def name_matches(query, row):
    """does `query` name this row, by its canonical name or any alias?
    the one place a name is resolved to a row, so a priority rule, the ui search box and an ocr
    tooltip read all agree on what counts as a hit."""
    q = normalize_name(query)
    return bool(q) and q in row_names(row)


def _phash_hamming(a, b):
    """hamming distance (0..64) between two 16-hex-char phash strings, via int popcount.
    kept stdlib-only so the ui can dedup the index without pulling in imagehash."""
    return bin(int(a, 16) ^ int(b, 16)).count('1')


def _name_wordset(name):
    """the set of lowercase alphanumeric word tokens in a name.
    used to spot swapped-word filename aliases of the same sprite, where the wiki uploads one
    glyph under both orders ('Sport Flashlight' vs 'Flashlight Sport') so the two share a word
    set but differ in order."""
    return frozenset(re.findall(r'[a-z0-9]+', (name or '').lower()))


def _row_informativeness(row):
    """rank an index row by how much metadata it carries, so dedup keeps the richest copy.
    the alias uploads come in with rarity null and an empty desc, because their swapped name
    misses the wiki rarity-category and page-title lookups, so the canonical row outranks them."""
    return (row.get('rarity') is not None, bool(row.get('desc')))


def dedup_index_rows(rows, near_ham=10):
    """drop duplicate index rows, keeping the most informative copy of each glyph.

    two passes, both scoped per category so a chance cross-category hash collision isn't merged:
      1. exact phash duplicates (the wiki serves a byte-identical sprite under several filenames).
      2. near-dup swapped-word aliases: same category, same name word-set, phashes within near_ham
         bits. the wiki uploads one sprite under both word orders (sportFlashlight vs flashlightSport),
         which pass 1 misses since the two uploads aren't byte-identical (hamming ~2-6). near_ham=10
         sits above that real-dup spread and below distinct same-word-set sprites (~28), so only true
         aliases merge.

    keeps the richest row per group (rarity known, then desc present), i.e. the canonical name; the
    bare alias (null rarity, empty desc) is dropped. order preserved."""
    drop = set()
    # pass 1: exact phash, any name
    by_hash = {}
    for r in rows:
        key = (r.get('category'), r.get('phash'))
        keep = by_hash.get(key)
        if keep is None:
            by_hash[key] = r
        else:
            lose = r if _row_informativeness(r) <= _row_informativeness(keep) else keep
            by_hash[key] = keep if lose is r else r
            drop.add(id(lose))
    # pass 2: near phash, only within same category + same word-set (a near-zero-risk signal)
    by_wordset = {}
    for r in rows:
        if id(r) in drop:
            continue
        by_wordset.setdefault((r.get('category'), _name_wordset(r.get('name'))), []).append(r)
    for (cat, ws), group in by_wordset.items():
        if len(group) < 2 or not ws:
            continue
        group.sort(key=_row_informativeness, reverse=True)
        anchor = group[0]
        for other in group[1:]:
            if _phash_hamming(anchor['phash'], other['phash']) <= near_ham:
                drop.add(id(other))
    return [r for r in rows if id(r) not in drop]


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
    pooled_out: bool = False # outside the priority match pool: read as unknown, never ocr'd or bought

    # live node state off the socket ring (detect.read_node_state), refreshed after every buy since
    # dbd auto-buys the whole path to whatever we click and the entity eats nodes mid-web.
    state: str = 'available'   # 'available' | 'bought' | 'entity'
    slot: int = None           # lattice slot index (None without a lattice fit)
    # the geometry a state re-read samples: the LATTICE slot center + socket radius, not (x, y, r).
    # x/y is isolate's plate centroid, a better click target but it shifts with the plate, and off it
    # the ring read lands a few px out and misses a marginal entity node. None without a lattice fit.
    slot_xy: tuple = None
    ring_r: float = None

    # matched-icon reads, from the library row the matcher picked.
    name: str = None       # matched icon display name
    match: dict = None     # raw library row {key,name,category,rarity,...} or None
    matched_name: str = None  # matcher's own guess, frozen at detection (name gets ocr-overwritten on resolution; this stays the cnn/ncc read for debug)
    score: float = 0.0
    margin: float = 0.0
    matcher: str = 'ncc'
    runner_up: str = None   # 2nd-best match name; debug-only diagnostic for near-tie margins

    # provenance of the final identity, the loop sets this to 'ocr' after a hover scan.
    resolved_by: str = 'match'   # 'match' | 'ocr'
    ocr_failed: bool = False      # loop set: hover attempted but read no tooltip, so item rules fall back to the weak icon match instead of skipping

    # pick provenance, set by spender.choose_next on the returned node, never by detection;
    # lets the loop log which tier won and, in an ordered tier, the within-tier rank of the rule.
    pick_tier: int = None        # 1-based index of the tier this node was picked from
    pick_rank: int = None        # 1-based within-tier rank of the rule it matched (ordered tiers)
    pick_ordered: bool = False   # was that tier in ordered mode

    # detection internals, kept optional for debug/ocr only, never used by the algorithm.
    glyph_bgr: object = field(default=None, repr=False)

    @classmethod
    def from_detection(cls, d):
        """adapt one detect() result dict into a Node.
        detect()'s 'cat' key is the socket SHAPE not a game category, so it maps to socket_shape here
        (the real category is derived from the match)."""
        m = d.get('match') or None
        return cls(
            x=int(d['x']), y=int(d['y']), r=int(d['r']),
            rarity=d['rar'],
            socket_shape=d['cat'],
            is_center=(d.get('kind') == 'center'),
            pooled_out=bool(d.get('pooled_out', False)),
            state=d.get('state', 'available'),
            slot=d.get('slot'),
            slot_xy=d.get('slot_xy'),
            ring_r=d.get('ring_r'),
            name=(m.get('name') if m else None),
            match=m,
            matched_name=(m.get('name') if m else None),
            score=float(d.get('score', 0.0)),
            margin=float(d.get('margin', 0.0)),
            matcher=d.get('matcher', 'ncc'),
            runner_up=d.get('runner_up'),
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
    def taken(self):
        """already bought (by us, or by dbd auto-pathing through it) or eaten by the entity.
        a taken node is never a buy target and is never identified, so it carries no glyph or match."""
        return self.state != 'available'

    @property
    def cnn_rescued(self):
        """a mid-score cnn match its runner-up gap vouches for (see CNN_RESCUE_MIN)."""
        return (self.matcher == 'cnn' and self.score >= CNN_RESCUE_MIN
                and CNN_RESCUE_MARGIN <= self.margin <= CNN_MARGIN_CAP)

    @property
    def confident(self):
        """is the icon match strong enough to trust on its own?
        direction depends on the matcher (ncc higher=better, phash lower=better).
        cnn takes a 2-d gate: a high score, OR a mid score whose runner-up is far behind
        (see cnn_rescued). ncc/phash stay score-only."""
        if self.match is None:
            return False
        if self.matcher == 'phash':
            return self.score <= PHASH_MAX_HAM
        if self.matcher == 'cnn':
            return self.score >= CNN_CONF_MIN or self.cnn_rescued
        return self.score >= NCC_CONF_MIN  # ncc / ncc_masked

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
    def resolution_reasons(self):
        """the specific reasons this node would route to ocr, empty list if it's trusted.
        mirrors needs_resolution but spells out which check failed and with what values, so a debug
        log can say WHY ocr fired (weak score, attr disagreement) not just that it did.
        a pooled-out node (outside the match pool) is deliberately unidentified, so it carries no
        reasons: it reads as unknown and is skipped, never routed to ocr. same for a taken node,
        which can never be bought, so hovering it would burn a second of ocr for nothing.
        """
        if self.pooled_out or self.taken:
            return []
        reasons = []
        if not self.confident:
            if self.match is None:
                reasons.append("no icon match")
            elif self.matcher == 'phash':
                reasons.append(f"weak match (dist {self.score:.0f}>{PHASH_MAX_HAM})")
            else:  # ncc / ncc_masked / cnn, higher cosine = better
                floor = CNN_CONF_MIN if self.matcher == 'cnn' else NCC_CONF_MIN
                # log the margin too, so a debug line says whether the rescue was even in play
                gap = f", margin {self.margin:.2f}" if self.matcher == 'cnn' else ""
                reasons.append(f"weak match (score {self.score:.2f}<{floor}{gap})")
        if self.match is not None:  # disagreement only means something against an actual match
            if not self.category_agrees:
                reasons.append(f"category {self.matched_category!r} not valid for {self.socket_shape}")
            if not self.rarity_agrees:
                reasons.append(f"rarity disagree (icon {self.matched_rarity!r} vs disk {self.rarity!r})")
        return reasons

    @property
    def needs_resolution(self):
        """should the loop fall back to an ocr hover scan to settle identity?
        true when the match is weak OR the observed and matched attrs disagree (see
        resolution_reasons for the itemized why). replaces a plain confidence gate: both routes go
        to ocr, not a guess."""
        return bool(self.resolution_reasons)

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
        if self.pooled_out:
            return False # outside the priority pool, left unidentified on purpose, never a target
        if self.taken:
            return False # already bought (ours or dbd's auto-path) or entity-eaten: clicking it does nothing

        rarity = rule.get("rarity")
        if rarity is not None and rarity != self.effective_rarity:
            return False

        if rule.get("type") == "item":
            if not (self.confident or self.resolved_by == "ocr"):
                # item rules want a trustworthy read, but if we already hovered and ocr read nothing
                # don't toss a correct-but-weak icon match: fall back to it (ocr_failed set post-hover).
                if not (self.ocr_failed and self.match is not None):
                    return False
            # the matched row's aliases count too, so a rule written against a perk's old name
            # ("Decisive Strike") still fires on the row the wiki now calls "Will to Live". fall back
            # to the bare name for a node with no matched row (the sim builds those).
            return (name_matches(rule["name"], self.match)
                    or normalize_name(self.name) == normalize_name(rule["name"]))
        if rule.get("type") == "category":
            return self.effective_category == rule["category"]
        return False


def _rule_row_indices(rows, rule):
    """indices of the library rows a single priority rule names.
    an item rule -> every row the name reaches, by its own name or an alias (rarity variants of the
    same icon, an old licensed name, and the two killers who genuinely share an add-on name: a bare
    "Mirror Shards" rule keeps both chucky's and jason's, since only one of them can be on the web in
    front of you); a category rule -> every row in that category. seeds both pool modes."""
    if rule.get("type") == "item":
        target = normalize_name(rule.get("name"))
        return [i for i, r in enumerate(rows) if target and target in row_names(r)]
    if rule.get("type") == "category":
        cat = rule.get("category")
        return [i for i, r in enumerate(rows) if r.get("category") == cat]
    return []


def build_pool_mask(rows, priority_tiers, inferred=True, exclusive=False):
    """which library rows the icon matcher may compare an on-screen glyph against, as a bool list
    (len == len(rows)), or None for no restriction (compare against all, the old behavior).

    two optional narrowings driven by the priority list, so we skip matching glyphs we'd never buy
    (a survivor run needn't score every killer's add-ons, and vice versa):
      exclusive: only the rows the priority list literally names (item rules pin their icon, category
                 rules keep their whole category); the strictest, a subset of inferred.
      inferred:  the union of each priority item's full bloodweb 'source' (a survivor item pulls in
                 the whole survivor side; a killer add-on pulls in that killer's add-ons plus shared
                 killer perks/offerings), so you drop other killers' add-ons but keep any real web.
    exclusive wins when both are on (narrower). rows whose side/owner is unknown are kept (safe
    superset: a little wasted compute beats dropping a node you might want). returns None when neither
    narrowing is active, or the inferred pass finds no side signal (e.g. an offering-only list).

    rows must be the same list/order detect matches against, so the mask lines up index-for-index.
    """
    if not (inferred or exclusive):
        return None
    n = len(rows)
    rules = [rule for tier in priority_tiers for rule in tier_rules(tier)]

    if exclusive:
        mask = [False] * n
        for rule in rules:
            for i in _rule_row_indices(rows, rule):
                mask[i] = True
        return mask

    # inferred: resolve the priority items to their source webs, then union those webs back in.
    survivor_active = False           # any survivor-side item/perk in the list -> keep survivor side
    killer_any = False                # any killer-side item in the list -> keep shared killer perks
    killer_owners = set()             # specific killer powers named, to keep just their add-ons
    for rule in rules:
        if rule.get("type") != "item":
            continue                  # category rules can't name a side; handled by cat union below
        for i in _rule_row_indices(rows, rule):
            r = rows[i]
            if r.get("side") == "survivor":
                survivor_active = True
            elif r.get("side") == "killer":
                killer_any = True
                if r.get("category") == "addon" and r.get("owner"):
                    killer_owners.add(r["owner"])
    cat_rule_cats = {rule.get("category") for rule in rules if rule.get("type") == "category"}

    if not (survivor_active or killer_any or killer_owners or cat_rule_cats):
        return None                   # no usable signal -> don't restrict (safer than blanking)

    mask = [False] * n
    for i, r in enumerate(rows):
        cat, side = r.get("category"), r.get("side")
        if cat in cat_rule_cats:
            keep = True               # a category rule keeps its whole category
        elif side is None:
            keep = True               # shared/unknown (offerings, undated add-ons): safe superset
        elif side == "survivor":
            keep = survivor_active
        elif cat == "addon":          # killer add-on: only the specific killers named
            keep = r.get("owner") in killer_owners
        elif cat == "power":
            keep = False              # powers are never bloodweb nodes
        else:                         # killer perk (shared across every killer web)
            keep = killer_any
        mask[i] = keep
    return mask
