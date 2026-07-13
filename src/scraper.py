"""scrape dbd icon assets + metadata from deadbydaylight.wiki.gg into a local library.

the whole detector leans on this: we identify on-screen bloodweb icons by matching them against
these exact sprites. we pull every category (perks, items, addons, offerings, powers), even ones
we'd never buy, so unwanted nodes still get identified instead of causing false matches.

source is the mediawiki api (action=query&list=allimages) filtered by the wiki's icon filename
prefixes (cargo tables aren't exposed here, but the prefixes reliably enumerate each category).
rarity isn't in the file metadata, so it's pulled in the same pass from the wiki's rarity categories
(item/addon/offering only; perks/powers have none and stay null). it's a soft cross-check; detection
reads rarity live from the on-screen disk color.

A FILENAME IS NOT AN IDENTITY, and most of what follows is that lesson. it is prefix-unstable (a new
chapter's sprites land as raw T_UI_ uploads and gain a curated Icon* twin weeks later), case-unstable
(IconAddon_VCR vs IconAddon_vcr), non-unique (chucky and jason both have a "Mirror Shards"), and
often just wrong (the offering the game calls "Toothy Torte" ships as IconsFavors_10thAnniversary; a
perk that lost its licence keeps its old filename, so IconPerks_decisiveStrike is now "Will to
Live"). so a prefix is only ever a place to LOOK; what a sprite *is* comes from the wiki article it
belongs to (resolve_articles), which is unique, stable, and the name the game's own tooltip prints.

one pass does it all: enumerate (twins collapsed) -> resolve each sprite to its article -> key it
(collision-safe) -> download (or reuse cached) -> normalize+phash -> annotate rarity -> aliases ->
dedup -> write. just `python -m src.scraper`, or `--dry-run` to see what would change first.

writes:
  data/icons/<category>/<key>.png   the raw sprite
  data/icons_index.json             one row per icon: key, name, aliases, category, rarity,
                                    obtainable, role, owner, side, desc, file, phash, url

`aliases` are the other names a row answers to: the wiki's redirects to its article (old licensed
names and the shorthand people type -- "Dying Light", "STBFL", "BBQ") plus the name its filename
would have given it. they are matched, not just searched (node.row_names), which is what lets us
correct a name without stranding the priority rule someone already wrote against the old one.

obtainable ('normal'|'event'|'unavailable') + desc (the wiki lead-sentence tooltip) are pulled in the
same pass: obtainability from the wiki's retired/event categories (+ powers by category), desc from
TextExtracts. both let the ui hide unbuyable glyphs and show a hover description; detection ignores them.

owner + side are filled last (they read the desc): an add-on's `owner` is the item type or killer
power it belongs to, parsed from its lead sentence ('... Add-on for Med-Kits.'), and `side` is
'survivor' | 'killer' | None for which bloodweb it shows up in. together they let the spender narrow
the match pool to the priority list's sources (see node.build_pool_mask); the survivor item types
that split a survivor add-on from a killer one are scraped from the Survivors article, not hardcoded,
so a new item type is picked up automatically. detection ignores both.

the phash is precomputed here (via normalize_sprite, kept in sync with detect.normalize_glyph) so
nearest-neighbor lookup at detect time is a cheap hamming distance over ~2k templates instead of full
template matching per candidate. if framing changes on either side, re-run the scrape to recompute
hashes from the cached pngs, no re-download needed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import imagehash
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from . import paths
from .node import dedup_index_rows, normalize_name

API = "https://deadbydaylight.wiki.gg/api.php"

# wiki icon filename prefix -> our category name; the prefixes namespace the sprite files, so they
# double as a clean way to enumerate each category. the wiki also files some newer/special assets
# under an inconsistent 'Icons' (extra s) prefix, so we enumerate those too (gold event offerings,
# plus a batch of newer perks) or we'd silently miss them.
#
# a prefix is only ever a place to LOOK. it is not an identity: see resolve_articles, which decides
# what a sprite actually is, so a file moving between prefixes changes nothing about its row.
PREFIXES = {
    "IconPerks_": "perk",
    "IconItems_": "item",
    "IconAddon_": "addon",
    "IconFavors_": "offering",
    "IconPowers_": "power",
    "IconsFavors_": "offering",   # Icons-variant prefix: special/event offerings (10thAnniversary, ...)
    "IconsPerks_": "perk",        # Icons-variant prefix: newer perks missing from IconPerks_
    # raw unreal texture uploads. a new chapter lands on the wiki under these FIRST, and only when an
    # editor gets round to it does the same sprite gain a curated Icon* copy under the same stem
    # (T_UI_iconAddon_AmonsNecktie -> IconAddon_AmonsNecktie). without them a brand-new killer is
    # invisible to prefix enumeration: jason (K43) shipped 20 add-ons and 3 perks and we had none of
    # them, because aiprefix matches the START of a filename and 'IconAddon_' never matches
    # 'T_UI_iconAddon_...'. the stem survives the move, so enumerate_icons can collapse the two.
    "T_UI_iconAddon_": "addon",
    "T_UI_iconItems_": "item",
    "T_UI_iconPerks_": "perk",
    "T_UI_iconsPerks_": "perk",
    "T_UI_iconPowers_": "power",
    "T_UI_iconFavors_": "offering",
}

# the not-yet-curated upload of a sprite. when both exist we take the curated twin (see enumerate_icons)
RAW_PREFIX = "T_UI_"

# extra names a row should answer to, for the rare thing the wiki has no redirect for. keyed by the
# row's folded name (see _norm). aliases are SCRAPED (fetch_redirects) precisely so this doesn't
# become another hand-maintained list that goes stale, so this stays an escape hatch, not a registry.
EXTRA_ALIASES = {
    # "toothytorte": ["birthday cake"],
}

# prefixes whose icons are the gold "event" disk tier; they carry no wiki rarity category, so tag
# them 'event' directly (lets the soft cross-check + priority rules target them).
EVENT_PREFIXES = {"IconsFavors_"}

# event icons pinned by hand, by exact File: name -> (key, category). the T_UI_ prefixes now enumerate
# these too, so this is no longer what FINDS them (that hole is what let jason's whole kit go missing:
# the same raw-upload convention, but nobody hand-listed a killer). what it still does is pin their
# 'event' rarity, which no wiki category carries, and pin a stable key that existing configs name.
# enumerate_icons skips anything listed here so they aren't scraped twice. their display name comes
# from the article like everyone else's, so they read as 'Banquet Toolbox', not as a texture stem.
EXTRA_EVENT_ICONS = {
    "T_UI_iconItems_toolbox_anniversary2026.png":     ("banquetToolbox",   "item"),
    "T_UI_iconItems_flashlight_anniversary2026.png":  ("banquetFlashlight", "item"),
    "T_UI_iconItems_medkit_anniversary2026.png":      ("banquetMedKit",    "item"),
    "T_UI_iconItems_poison_anniversary2026.png":      ("banquetPoison",    "item"),
}

# first-run scrape writes here: cache_dir = repo data/ in dev, %APPDATA%/dbdbp-pas/cache when frozen
DEFAULT_OUT = paths.cache_dir() / "icons"
DEFAULT_INDEX = paths.cache_dir() / "icons_index.json"

# rarity comes from wiki categories: pages are tagged "<rarity> Items/Add-ons/Offerings".
# only reliable source, since allimages carries no rarity and there are no cargo tables.
# ~100 reworked/top-tier icons sit in a "Visceral" bucket with no rarity tag, so they stay
# null. fine, since rarity is just a soft cross-check; the disk color is the live read.
RARITY_WORDS = ["Common", "Uncommon", "Rare", "Very Rare", "Ultra Rare"]
RARITY_TYPES = {"item": "Items", "addon": "Add-ons", "offering": "Offerings"}

# obtainability: which scraped glyphs can still turn up in a current bloodweb. three states:
# 'unavailable' (never/no longer), 'event' (only while its event runs), 'normal' (always).
# the wiki marks this through categories, not one flag, so we read two sets:
#   retired -> gone for good; only offerings have a retired category (base content gets reworked not
#              removed, so there's no Retired Items/Add-ons worth pulling).
#   event   -> event-only skins/rewards (masquerade med-kit, ghastly gateau, ...).
# powers need no wiki call: the only non-bloodweb category, flagged 'unavailable' by category at row
# build. retired wins over event.
RETIRED_CATEGORIES = ["Retired Offerings"]
EVENT_CATEGORIES = ["Event Items", "Event Add-ons", "Event Offerings"]

# role: which side plays a glyph, for the ui's killer/survivor filter. only perks carry a clean wiki
# role category; items are always survivor and powers always killer (by category). add-ons and
# offerings have no role category (most offerings suit either side), so they stay null = shown for
# both roles. like rarity/obtainability, a soft ui hint; detection ignores it.
PERK_ROLE_CATEGORIES = {
    "survivor": ["Survivor Perks", "Unique Survivor Perks"],
    "killer":   ["Killer Perks", "Unique Killer Perks"],
}

# wiki.gg etiquette: identify the scraper with a contact
USER_AGENT = "dbd-bloodweb-autospender/0.1 (contact: brandonpardi24@gmail.com)"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    # retry transient errors so a long scrape (and the eventual public exe) doesn't die
    # on a network blip; backoff spaces the retries out
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504)
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def list_icons(session, prefix, limit=None):
    """yield {name, url} for every file under a wiki icon prefix, paging the api.

    allimages caps at 500 results per call, so we follow the aicontinue token until the
    prefix is exhausted (or until `limit` files, for quick test runs).
    """
    out = []
    aicontinue = None
    while True:
        params = {
            "action": "query",
            "list": "allimages",
            "aiprefix": prefix,
            "ailimit": "500",
            "format": "json",
        }
        if aicontinue:
            params["aicontinue"] = aicontinue
        r = session.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for img in data.get("query", {}).get("allimages", []):
            out.append({"name": img["name"], "url": img["url"]})
            if limit and len(out) >= limit:
                return out
        cont = data.get("continue", {}).get("aicontinue")
        if not cont:
            return out
        aicontinue = cont


def prettify(key: str) -> str:
    """turn a camelcase filename stem into a human-ish display name.

    display only; `key` stays the canonical id. e.g. 'EyesOfBelmont' -> 'Eyes Of Belmont'.
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key)        # camel boundary
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)        # acronym -> word boundary
    return s.replace("_", " ").strip()


# normalized glyph canvas size. MUST match detect.GLYPH_SIZE: an on-screen glyph and a library
# sprite have to be framed the same way or their phashes don't compare. detect's side strips the
# rarity disk; the sprite has none, so here it's just crop+pad+resize.
GLYPH_SIZE = 128


def normalize_sprite(img: Image.Image) -> Image.Image:
    """frame a library sprite the way detect frames an on-screen glyph, so their phashes are
    comparable: tight-crop to the glyph, square-pad preserving aspect, resize to a fixed size,
    on black. phash squishes whatever it gets to 32x32, so what makes two hashes comparable is
    identical framing (aspect + centering), not the source size.

    kept in sync with detect.normalize_glyph by hand on purpose (the two sides do different cleanup,
    detect removes the disk and handles off-center crops, but must agree on this final framing).
    returns an 'L' (grayscale) image ready to phash.
    """
    bbox = img.getbbox()                       # glyph extent (alpha-aware); None if fully blank
    glyph = img.crop(bbox) if bbox else img
    gw, gh = glyph.size
    side = max(gw, gh)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))   # black, opaque
    canvas.alpha_composite(glyph, ((side - gw) // 2, (side - gh) // 2))   # center, transparent->black
    canvas = canvas.resize((GLYPH_SIZE, GLYPH_SIZE), Image.LANCZOS)
    return canvas.convert("L")


def icon_phash(img: Image.Image) -> str:
    """perceptual hash of an icon sprite, framed by normalize_sprite to match detect's query
    glyphs. 8x8 dct hash -> 64-bit value (str() gives 16 hex chars). if the framing on either
    side changes, re-run the scrape to recompute these from the cached pngs (no re-download).
    """
    return str(imagehash.phash(normalize_sprite(img)))


def download_icon(session, url, dest: Path, attempts=3):
    """download one icon to dest as rgba png and return the PIL image for hashing.

    the wiki intermittently serves a transient corrupt body (valid png header, garbled
    rest) that pillow can't decode, so we re-fetch a few times before giving up; a fresh
    fetch usually comes back clean. returns None only if every attempt fails, so one
    persistently-bad asset doesn't take down the whole scrape.
    """
    for attempt in range(attempts):
        r = session.get(url, timeout=30)
        r.raise_for_status()
        try:
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")  # convert also forces the decode
        except Exception:
            time.sleep(0.3 * (attempt + 1))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return img
    return None


def resolve_image_urls(session, filenames):
    """canonical download url for each bare image filename, via the imageinfo api. used for the
    EXTRA_EVENT_ICONS, which live under raw T_UI_ texture names we don't enumerate by prefix, so
    we look them up by exact File: title instead of paging allimages. returns {filename: url}."""
    titles = "|".join(f"File:{f}" for f in filenames)
    data = session.get(API, params={
        "action": "query", "format": "json",
        "prop": "imageinfo", "iiprop": "url", "titles": titles,
    }, timeout=30).json()
    out = {}
    for pg in data.get("query", {}).get("pages", {}).values():
        info = pg.get("imageinfo")
        if info:  # mediawiki normalizes underscores to spaces in title; put them back to match
            out[pg["title"].split(":", 1)[1].replace(" ", "_")] = info[0]["url"]
    return out


def _index_one(session, url, key, name, category, rarity, obtainable, role,
               out_dir, index_path, force, delay, skipped):
    """download (or reuse the cached) icon and build its index row, or None if it won't decode.
    shared by the prefix-enumerated icons and the explicit EXTRA_EVENT_ICONS so both sides frame
    the row + phash identically. appends to skipped on an undecodable asset. `desc` is left empty
    here and filled in one batched pass (fill_descriptions) once every row's name is known."""
    dest = out_dir / category / f"{key}.png"
    if dest.exists() and not force:
        img = Image.open(dest).convert("RGBA")
    else:
        img = download_icon(session, url, dest)
        if img is None:                      # undecodable asset, log it and move on
            skipped.append(key)
            return None
        time.sleep(delay)                    # be polite to the wiki between downloads
    return {
        "key": key,
        "name": name,
        "category": category,
        "rarity": rarity,
        "obtainable": obtainable,            # 'normal' | 'event' | 'unavailable' (see _obtainable)
        "role": role,                        # 'killer' | 'survivor' | None (see _role)
        "owner": None,                       # add-on's item type / killer power, filled by fill_sources
        "side": None,                        # 'survivor' | 'killer' | None, filled by fill_sources
        "desc": "",                          # wiki lead sentence, filled by fill_descriptions
        "aliases": [],                       # other names it answers to, filled by fill_aliases
        # relative to the index file so the library stays portable; detect resolves it
        # against index_path.parent
        "file": os.path.relpath(dest, index_path.parent).replace("\\", "/"),
        "phash": icon_phash(img),
        "url": url,
    }


def scrape(categories, out_dir: Path, index_path: Path, limit=None, force=False, delay=0.1,
           dedup=True, progress=None):
    # progress is an optional callback(stage, current, total), current/total None for phases with no
    # countable work, so the ui can show a real bar instead of a spinner. cli passes nothing.
    def _p(stage, cur=None, tot=None):
        if progress:
            progress(stage, cur, tot)

    session = _session()
    prefixes = {p: c for p, c in PREFIXES.items() if c in categories}
    # rarity in the same pass: pull the wiki's rarity categories once up front, then look each
    # icon up as we build its row, so one scrape yields a fully-populated index (no second pass).
    # only item/addon/offering have rarity categories, so skip the fetch unless one's selected.
    _p("fetching rarity data")
    rmap = fetch_rarity_map(session) if any(c in RARITY_TYPES for c in categories) else {}
    # obtainability (retired/event sets) in the same up-front pass as rarity, so one scrape yields a
    # fully-annotated index. cheap (a few category calls); always pulled since any category can hold
    # event/retired glyphs and powers are flagged by category regardless.
    _p("fetching obtainability data")
    obmap = fetch_obtainability(session)
    # role (killer/survivor) for the ui filter, same up-front-map pattern: only perks need the wiki
    # lookup, so skip the category calls unless perks are selected (items/powers go by category).
    _p("fetching perk roles")
    rolemap = fetch_perk_roles(session) if "perk" in categories else {}
    # which articles are real buyable things, so an icon's fileusage can be filtered to candidates.
    # reads the same (memoised) category listings as the three maps above, so it costs nothing.
    _p("listing articles")
    things = fetch_thing_titles(session)

    # what each sprite IS, before we touch the network for images: the filename can't be trusted for
    # identity (see resolve_articles), so the article decides both the row's name and, where two
    # filenames collide, its key.
    _p("listing icons")
    files = enumerate_icons(session, prefixes, limit=limit)
    articles, shared = resolve_articles(session, files, things, progress=progress)
    keyed = assign_keys(files, articles)

    index, skipped = [], []
    aka = {}     # key -> (name the filename would have given it, article or None, articles sharing it)
    for i, (fname, url, key, stem, prefix, category) in enumerate(
            tqdm(keyed, desc="icons", unit="icon")):
        article = articles.get(fname)
        old = prettify(stem)
        name = article or old            # the article is the in-game name; the stem is a fallback
        aka[key] = (old, article, shared.get(fname, ()))
        # event prefixes -> 'event' (gold tier, no wiki rarity category); else the wiki
        # rarity, or null for perks/powers and ~visceral top-tier icons
        rarity = "event" if prefix in EVENT_PREFIXES else rmap.get(category, {}).get(_norm(name))
        obtainable = _obtainable(category, name, rarity, obmap)
        role = _role(category, name, rolemap)
        row = _index_one(session, url, key, name, category, rarity, obtainable, role,
                         out_dir, index_path, force, delay, skipped)
        if row:
            index.append(row)
        _p("downloading icons", i + 1, len(keyed))

    # explicit event icons the prefixes miss (raw T_UI_ textures); look them up by exact name
    # and tag 'event'. skip ones whose category wasn't requested so --categories stays honest.
    # their key is pinned here (configs reference it), but the name still comes from the article, so
    # they read as the game names them ('Banquet Toolbox') rather than as their texture stem.
    extras = {f: kc for f, kc in EXTRA_EVENT_ICONS.items() if kc[1] in categories}
    if extras:
        _p("downloading event icons")
        extra_urls = resolve_image_urls(session, list(extras))
        extra_files = [(f, extra_urls[f], k, "", c, (f,))
                       for f, (k, c) in extras.items() if f in extra_urls]
        extra_articles, extra_shared = resolve_articles(session, extra_files, things)
        for fname, (key, category) in extras.items():
            url = extra_urls.get(fname)
            if url is None:                  # not found on the wiki (renamed/removed), log + skip
                skipped.append(fname)
                continue
            article = extra_articles.get(fname)
            old = prettify(key)
            name = article or old
            aka[key] = (old, article, extra_shared.get(fname, ()))
            obtainable = _obtainable(category, name, "event", obmap)
            role = _role(category, name, rolemap)
            row = _index_one(session, url, key, name, category, "event", obtainable, role,
                             out_dir, index_path, force, delay, skipped)
            if row:
                index.append(row)

    # fill the wiki lead-sentence tooltips in one batched pass now that every row's name is known.
    # rows now carry their real article title, so this hits for the ~33 that never matched a page
    # before (and they pick up their rarity and role above for the same reason).
    fill_descriptions(session, index, progress=progress)

    # owner + side, last: owner is parsed from the add-on lead sentence (so it needs desc filled
    # first), and the survivor item types that classify an add-on's side are scraped live here.
    _p("fetching item sources")
    fill_sources(index, fetch_survivor_item_types(session))

    # the other names each row answers to (wiki redirects + the name its filename would have given
    # it), so a rename never strands an existing priority rule.
    _p("fetching aliases")
    redirects = fetch_redirects(session, [a for _old, a, _s in aka.values() if a], progress=progress)
    fill_aliases(index, aka, redirects)

    _p("writing index")
    if dedup:
        index = dedup_index_rows(index)
    clashes = check_key_collisions(index)
    if clashes:                          # assign_keys is meant to make this impossible; be loud
        raise RuntimeError(
            "two rows would share one sprite file: "
            + "; ".join(f"{cat}/{folded} <- {keys}" for cat, folded, keys in clashes))
    index.sort(key=lambda row: (row["category"], row["key"]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index, skipped


def _norm(name):
    """squash a name to lowercase alphanumerics so the wiki page 'Amanda's Letter' and our
    prettified key 'Amandas Letter' compare equal. the engine's fold (node.normalize_name), shared
    on purpose so the library, the priority rules and an ocr tooltip all agree what a name is."""
    return normalize_name(name)


def _deaccent(s):
    """'Zōri' -> 'Zori', so a wiki title can seed an ascii key."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c))


_CATEGORY_CACHE = {}


def category_members(session, cat_title):
    """all page titles in a wiki category, paged via list=categorymembers (500/call).
    memoised for the run: the rarity, event, role and 'is this a real thing' maps all read the same
    category listings, and there's no sense paging each one twice per scrape."""
    if cat_title in _CATEGORY_CACHE:
        return _CATEGORY_CACHE[cat_title]
    out, cont = [], {}
    while True:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Category:{cat_title}", "cmtype": "page", "cmlimit": "500"
        }
        params.update(cont)
        data = session.get(API, params=params, timeout=30).json()
        out += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data:
            _CATEGORY_CACHE[cat_title] = out
            return out
        cont = data["continue"]


def fetch_rarity_map(session):
    """build {category: {normalized_name: rarity}} from the wiki's rarity categories."""
    rmap = {cat: {} for cat in RARITY_TYPES}
    for our_cat, type_word in RARITY_TYPES.items():
        for rar in RARITY_WORDS:
            for title in category_members(session, f"{rar} {type_word}"):
                rmap[our_cat][_norm(title)] = rar.lower()
    return rmap


def fetch_obtainability(session):
    """build {normalized_name: 'event'|'unavailable'} from the wiki's retired/event categories.
    retired (gone for good) is applied second so it wins over event (event-only) when a name is in
    both. powers aren't here; they're flagged 'unavailable' by category as each row is built."""
    state = {}
    for cat in EVENT_CATEGORIES:
        for title in category_members(session, cat):
            state[_norm(title)] = "event"
    for cat in RETIRED_CATEGORIES:
        for title in category_members(session, cat):
            state[_norm(title)] = "unavailable"
    return state


def _obtainable(category, name, rarity, obmap):
    """3-state obtainability for one row. powers are never bloodweb nodes; otherwise take the wiki
    retired/event tag, falling back to the 'event' rarity band (the gold IconsFavors_/banquet tier)
    so event glyphs the categories miss still read 'event'. default 'normal'."""
    if category == "power":
        return "unavailable"
    return obmap.get(_norm(name)) or ("event" if rarity == "event" else "normal")


def fetch_perk_roles(session):
    """build {normalized_perk_name: 'killer'|'survivor'} from the wiki's perk role categories."""
    roles = {}
    for role, cats in PERK_ROLE_CATEGORIES.items():
        for cat in cats:
            for title in category_members(session, cat):
                roles[_norm(title)] = role
    return roles


def _role(category, name, rolemap):
    """role for one row: items are survivor, powers killer, perks from the wiki role map, and
    everything else (add-ons, offerings) null so the ui shows it under both role filters."""
    if category == "item":
        return "survivor"
    if category == "power":
        return "killer"
    if category == "perk":
        return rolemap.get(_norm(name))
    return None


# the article categories that mean "this page IS a thing you can buy", as opposed to a hub page that
# merely shows the icon (Perks, Achievements, the killer's own page). used to filter an icon's
# fileusage down to real candidates. maps a category title word to our category name.
_THING_CATEGORY_WORDS = [("Add-on", "addon"), ("Item", "item"), ("Offering", "offering")]


def fetch_thing_titles(session):
    """{article title: our category} for every wiki page that is an actual buyable thing.
    built from the category listings the scrape already pulls (rarity, event, retired, perk role), so
    it costs nothing extra (category_members is memoised)."""
    things = {}
    for our_cat, type_word in RARITY_TYPES.items():
        for rar in RARITY_WORDS:
            for title in category_members(session, f"{rar} {type_word}"):
                things[title] = our_cat
    for cat in EVENT_CATEGORIES + RETIRED_CATEGORIES:
        our = next((c for word, c in _THING_CATEGORY_WORDS if word in cat), None)
        for title in category_members(session, cat):
            things.setdefault(title, our)
    for role, cats in PERK_ROLE_CATEGORIES.items():
        for cat in cats:
            for title in category_members(session, cat):
                things[title] = "perk"
    return things


def enumerate_icons(session, prefixes, limit=None):
    """every icon worth scraping, as [(name, url, stem, prefix, category, twins)].

    the wiki carries a new chapter's sprite twice for a while: the raw T_UI_ upload, and (once an
    editor gets to it) a curated Icon* copy of the same image under the same stem. 38 of those exist
    today, all from last chapter. that's one item found twice, so we download one -- the curated
    name, since that's where the wiki settles -- and because the stem survives the move, the row's
    key doesn't shift when the curated copy lands. that swap being a no-op is the whole point.

    `twins` keeps BOTH filenames, and that matters more than it looks: the article links whichever
    upload the editor happened to use, and it is usually the raw one. Kaneki's Satchel cites
    T_UI_iconAddon_satchel; the curated IconAddon_satchel is embedded on no page at all. so resolving
    a sprite to its article has to look under every name that sprite has, or the file we chose to
    download resolves to nothing and silently keeps its filename-derived name.
    """
    found = {}
    for prefix, category in prefixes.items():
        for entry in list_icons(session, prefix, limit=limit):
            if entry["name"] in EXTRA_EVENT_ICONS:
                continue                       # handled explicitly below, with a pinned key + tag
            stem = Path(entry["name"][len(prefix):]).stem
            found.setdefault((category, stem), []).append((entry["name"], entry["url"], prefix))
    out = []
    for (category, stem), uploads in found.items():
        # curated copy wins the download; a raw T_UI_ name only stands in until one exists. sorted so
        # the choice never depends on which prefix happened to be enumerated first.
        name, url, prefix = min(uploads, key=lambda u: (u[2].startswith(RAW_PREFIX), u[0]))
        out.append((name, url, stem, prefix, category, tuple(u[0] for u in uploads)))
    return out


def fetch_file_usage(session, filenames, progress=None):
    """{filename: [ns0 article titles that embed it]}, via prop=fileusage, batched 50 at a time.
    the wiki renders item infoboxes from a lua module, so a page's icon isn't in its wikitext at all;
    going backwards from the sprite is the only cheap way to find the article it belongs to."""
    out = {}
    for start in range(0, len(filenames), 50):
        batch = filenames[start:start + 50]
        cont = {}
        while True:
            params = {
                "action": "query", "format": "json", "prop": "fileusage",
                "fuprop": "title", "funamespace": "0", "fulimit": "500",
                "titles": "|".join(f"File:{f}" for f in batch),
            }
            params.update(cont)
            data = session.get(API, params=params, timeout=60).json()
            # mediawiki normalizes titles (underscores -> spaces); map back to our exact filename
            renamed = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
            for pg in data.get("query", {}).get("pages", {}).values():
                title = renamed.get(pg["title"], pg["title"])
                fname = title.split(":", 1)[1].replace(" ", "_")
                out.setdefault(fname, []).extend(
                    u["title"] for u in pg.get("fileusage", []) or [])
            if "continue" not in data:
                break
            cont = data["continue"]
        if progress:
            progress("resolving names", min(start + 50, len(filenames)), len(filenames))
    return out


# which icon filenames may belong to a row of each category, for the own-icon check below
CATEGORY_FILE_RE = {
    "addon":    r"Icons?Addon_",
    "perk":     r"Icons?Perks_",
    "item":     r"Icons?Items_",
    "offering": r"Icons?Favors_",
    "power":    r"Icons?Powers_",
}

_ARTICLE_ICON_CACHE = {}


def _icon_stem(fname):
    """the stem of an icon filename, whichever prefix it arrived under.

    a sprite's curated Icon* copy and its raw T_UI_ upload share a stem -- that is what makes them
    twins -- so comparing stems is how we confirm an article's icon is OUR file even while the
    article still renders the other twin. comparing the raw filenames instead silently loses the
    rename for every chapter mid-transition (kaneki's satchel, torture apparatus).
    case-sensitive on purpose: mirrorShards and MirrorShards are two different add-ons.
    """
    for prefix in sorted(PREFIXES, key=len, reverse=True):
        if fname.startswith(prefix):
            return Path(fname[len(prefix):]).stem
    return Path(fname).stem


def article_icon(session, title, category):
    """the icon file an article renders as ITS OWN, or None if it doesn't show exactly one.

    this is the check that makes renaming safe, and it is worth the api call. fileusage only says a
    sprite appears somewhere on a page, which is far too loose to identify anything: the Stake Out
    page cites Hyperfocus, the Gnarled Compass page cites the Maps item it's an add-on for. trusting
    that alone would rename 40 healthy rows onto other things' icons.

    an article's own icon is the one it renders full-size in its lead section (a merely *referenced*
    icon is served as a /thumb/), with a filename prefix matching the row's category. exactly one
    such icon means the article is about that sprite; anything else, we don't rename.
    """
    hit = (title, category)
    if hit in _ARTICLE_ICON_CACHE:
        return _ARTICLE_ICON_CACHE[hit]
    pat = re.compile(
        r"/images/(thumb/)?((?:T_UI_)?" + CATEGORY_FILE_RE[category] + r"[^\"?/]+\.png)", re.I)
    data = session.get(API, params={
        "action": "parse", "page": title, "prop": "text", "section": "0", "format": "json",
    }, timeout=30).json()
    html = data.get("parse", {}).get("text", {}).get("*", "")
    full = []
    for tag in re.findall(r"<img[^>]+>", html):
        m = pat.search(tag)
        if m and not m.group(1):     # full-size render = the infobox sprite, not an inline citation
            full.append(m.group(2))
    uniq = list(dict.fromkeys(full))
    icon = uniq[0] if len(uniq) == 1 else None
    _ARTICLE_ICON_CACHE[hit] = icon
    return icon


def resolve_articles(session, files, things, progress=None):
    """what each sprite actually IS: ({filename: article title}, {filename: [articles sharing it]}).

    a filename is not an identity. it is case-unstable (IconAddon_VCR vs IconAddon_vcr), prefix-
    unstable (a new chapter's T_UI_ upload gains a curated twin later), and routinely just wrong: the
    offering the game calls "Toothy Torte" ships as IconsFavors_10thAnniversary, and a perk that lost
    its licence keeps the old filename (IconPerks_decisiveStrike is now "Will to Live"). the article
    is none of those things -- it is unique, it is stable, and it is what the game's tooltip says.

    cheap first, careful only where it matters:
      1. fileusage -> the articles embedding the sprite, kept only where the article is a real thing
         (`things`) rather than a hub page.
      2. take a candidate outright when its title folds to the filename stem: the name already agrees
         with the file, so there is nothing to check. that's ~95% of rows, at no extra cost.
      3. otherwise the file is claiming a name it doesn't look like -- exactly the case that must not
         be guessed -- so confirm it against the article's own icon (article_icon) and drop it if
         they disagree. a file we can't resolve keeps its filename-derived name, as it does today, so
         nothing regresses.

    a sprite can honestly belong to two articles: hillbilly's and bubba's "Begrimed Chains" are
    different add-ons drawn with one icon, and the wiki gives each a page. one sprite can only be one
    row, so it takes the first title (candidates are sorted, so the pick is stable) and answers to
    the other as an alias.
    """
    # every name the sprite goes by, not just the one we chose to download (see enumerate_icons)
    usage = fetch_file_usage(session, sorted({t for f in files for t in f[5]}), progress=progress)
    out, shared = {}, {}
    for i, (fname, _url, stem, _prefix, category, twins) in enumerate(files):
        pages = {p for twin in twins for p in usage.get(twin, [])}
        cands = sorted(p for p in pages if things.get(p) == category)
        picks = [p for p in cands if _norm(p) == _norm(stem)]
        if not picks and not cands:
            # a brand-new chapter's pages aren't in the rarity/role categories yet (jason's perks
            # aren't), so `things` can't nominate a candidate and we fall back to the pages
            # themselves: an exact title match, else a title that merely CONTAINS the stem -- the
            # game prefixes its hexes, so ScaredToDeath's page is "Hex: Scared to Death". a loose
            # match like that can't be trusted on its own, so it still has to be confirmed on the
            # article's own icon below.
            picks = sorted(p for p in pages if _norm(p) == _norm(stem))
            if not picks and _norm(stem):
                cands = sorted(p for p in pages if _norm(stem) in _norm(p))[:8]

        if not picks:
            picks = [p for p in cands
                     if _icon_stem(article_icon(session, p, category) or "") == stem]
        if picks:
            out[fname] = picks[0]
            if len(picks) > 1:
                shared[fname] = picks[1:]
        if progress:
            progress("confirming names", i + 1, len(files))
    return out, shared


def fetch_redirects(session, titles, progress=None):
    """{article title: [titles that redirect to it]} -- the wiki's own record of a thing's other names.

    this is where aliases come from, and why they're scraped rather than listed by hand: the
    redirects to "Keep Them Waiting" are its pre-licence name ("Save the Best for Last") and what
    players actually type ("STBFL"). batched 50 at a time.
    """
    out = {}
    titles = sorted(set(titles))
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        cont = {}
        while True:
            params = {"action": "query", "format": "json", "prop": "redirects",
                      "rdnamespace": "0", "rdlimit": "500", "titles": "|".join(batch)}
            params.update(cont)
            data = session.get(API, params=params, timeout=60).json()
            for pg in data.get("query", {}).get("pages", {}).values():
                out.setdefault(pg["title"], []).extend(
                    r["title"] for r in pg.get("redirects", []) or [])
            if "continue" not in data:
                break
            cont = data["continue"]
        if progress:
            progress("fetching aliases", min(start + 50, len(titles)), len(titles))
    return out


def _key_from_title(title):
    """a stable index key from a wiki article title: "Mirror Shards (Playtime's Over)" ->
    'mirrorShardsPlaytimesOver'. only used to break a filename collision (see assign_keys), i.e.
    where the filename has already proven it can't identify anything and the article can."""
    words = re.findall(r"[A-Za-z0-9]+", re.sub(r"['’]", "", _deaccent(title)))
    if not words:
        return "unnamed"
    head, *rest = words
    return head.lower() + "".join(w[:1].upper() + w[1:] for w in rest)


def assign_keys(files, articles):
    """key each enumerated icon, safely. returns [(name, url, key, stem, prefix, category)].

    the key stays the filename stem, as it always was -- except where two DIFFERENT files in one
    category have stems differing only by case. that happens: IconAddon_VCR vs IconAddon_vcr,
    IconsPerks_Deadline vs IconPerks_deadline, and now chucky's IconAddon_mirrorShards vs jason's
    T_UI_iconAddon_MirrorShards. it is not a T_UI artifact -- 7 of the 8 collisions today are between
    two curated files, so this has been latent for a long time.

    it matters because the sprite lands at data/icons/<category>/<key>.png and windows matches that
    path case-insensitively: the two rows are then ONE file. the second download either clobbers the
    first or is skipped as already-cached, both rows end up hashing identically, and dedup merges
    them -- silently deleting an add-on. that is exactly what would have happened to jason's Mirror
    Shards, which is a different sprite from chucky's (hamming 24).

    so on a collision, identity falls back to the one thing that CAN tell them apart: the article.
    two files with no article to separate them are the same sprite uploaded twice (the wiki does
    this), and get a deterministic suffix so neither clobbers the other; phash dedup then drops the
    duplicate at the end. sorting by stem keeps the outcome independent of enumeration order, so a
    file arriving under a different prefix later doesn't reshuffle anyone's key.
    """
    groups = {}
    for f in files:
        groups.setdefault((f[4], f[2].casefold()), []).append(f)
    out = []
    for group in groups.values():
        if len(group) == 1:
            name, url, stem, prefix, category, _twins = group[0]
            out.append((name, url, stem, stem, prefix, category))
            continue
        taken = {}
        for name, url, stem, prefix, category, _twins in sorted(group, key=lambda f: f[2]):
            title = articles.get(name)
            key = _key_from_title(title) if title else stem
            n = taken.get(key.casefold(), 0)
            taken[key.casefold()] = n + 1
            if n:                    # same article (or no article) as an earlier member: one sprite
                key = f"{key}_{n}"   # uploaded twice. keep both files; dedup drops the copy.
            out.append((name, url, key, stem, prefix, category))
    return out


def check_key_collisions(rows):
    """rows that would share one sprite file, as [(category, folded key, [keys])]. empty is the
    invariant assign_keys exists to hold; we assert it every scrape so this class of bug can never
    go quiet again (it has been silently clobbering VCR/vcr and friends for a long time)."""
    seen = {}
    for r in rows:
        seen.setdefault((r["category"], r["key"].casefold()), []).append(r["key"])
    return [(cat, folded, keys) for (cat, folded), keys in seen.items() if len(keys) > 1]


def fill_aliases(rows, aka, redirects):
    """populate each row's `aliases` in place: every other name it should answer to.

    four sources: the wiki's redirects to its article (old licensed names and the shorthand people
    actually type -- "Dying Light", "DS", "BBQ"), any other article drawn with the same sprite
    (bubba's vs hillbilly's "Begrimed Chains"), the name we'd have given it from its filename before
    the article was resolved (so a priority list saved against "10th Anniversary" still finds Toothy
    Torte), and EXTRA_ALIASES for the rare thing the wiki has no redirect for. an alias that folds to
    the row's own name is dropped as noise.

    aliases are matched, not merely searched: a rule, the ui search box and an ocr tooltip all read
    them through node.row_names. that's what keeps existing configs working across a rename.
    """
    for r in rows:
        old, article, shared = aka.get(r["key"], (None, None, ()))
        names = list(shared or [])
        names += list(redirects.get(article) or []) if article else []
        if old:
            names.append(old)
        names += EXTRA_ALIASES.get(_norm(r["name"]), [])
        seen, out = {_norm(r["name"])}, []
        for n in names:
            folded = _norm(n)
            if folded and folded not in seen:
                seen.add(folded)
                out.append(n)
        r["aliases"] = out


# the wiki article whose item table enumerates every survivor item TYPE (Flashlights, Med-Kits,
# ...). we read it to tell a survivor add-on from a killer one without a hardcoded list, so a new
# item type is picked up automatically on the next scrape (see fetch_survivor_item_types).
SURVIVOR_ITEMS_PAGE = "Survivors"


def fetch_survivor_item_types(session):
    """set of normalized survivor item-type names (flashlights, medkits, toolboxes, ...), read from
    the Survivors article's 'List of Survivor Items' table.
    an add-on is survivor-side when its owner (see _owner) is in this set, else a killer's. scraped
    rather than hardcoded so a new item type needs no code change. returns an empty set if the
    section/table isn't found, leaving add-on sides null (a safe non-narrowing default) rather than
    failing the scrape."""
    secs = session.get(API, params={
        "action": "parse", "page": SURVIVOR_ITEMS_PAGE, "prop": "sections", "format": "json",
    }, timeout=30).json().get("parse", {}).get("sections", [])
    idx = next((s["index"] for s in secs if "survivor item" in s["line"].lower()), None)
    if idx is None:
        return set()
    wt = session.get(API, params={
        "action": "parse", "page": SURVIVOR_ITEMS_PAGE, "section": idx,
        "prop": "wikitext", "format": "json",
    }, timeout=30).json().get("parse", {}).get("wikitext", {}).get("*", "")
    # each item type is a table header cell shaped '! [[Flashlights]] [[File:IconItems ...]]'
    return {_norm(t) for t in re.findall(r"!\s*\[\[([^\]|]+?)\]\]\s*\[\[File:", wt)}


def _owner(category, desc):
    """the item type or killer power an add-on belongs to, normalized, parsed from its wiki lead
    sentence ('Spring Clamp is an Uncommon Add-on for Toolboxes.' -> 'toolboxes'). only add-ons
    carry an owner; everything else (and the ~5% of add-ons with no wiki page/desc) is None."""
    if category != "addon":
        return None
    m = re.search(r"[Aa]dd-?on for ([^.]+)", desc or "")
    return _norm(m.group(1)) if m else None


def _side(category, owner, role, surv_types):
    """which bloodweb side a row appears in, for the priority-inferred match pool:
    'survivor' | 'killer' | None. items are survivor, powers killer, perks take their role, and an
    add-on is survivor when its owner is a survivor item type else killer (None when the owner didn't
    parse). offerings have no wiki side so stay None = shared, kept in the pool for either side."""
    if category == "item":
        return "survivor"
    if category == "power":
        return "killer"
    if category == "perk":
        return role
    if category == "addon":
        if owner is None:
            return None
        return "survivor" if owner in surv_types else "killer"
    return None


def fill_sources(rows, surv_types):
    """populate each row's `owner` + `side` in place, run after fill_descriptions since owner comes
    from the add-on lead sentence. a metadata-only pass (no downloads). these drive spender's
    optional comparison-pool narrowing (node.build_pool_mask); detection itself ignores them."""
    for r in rows:
        r["owner"] = _owner(r["category"], r.get("desc"))
        r["side"] = _side(r["category"], r["owner"], r.get("role"), surv_types)


def fetch_page_titles(session):
    """every real (non-redirect) main-namespace article title, paged via list=allpages (500/call).
    used to resolve our prettified row names to the exact wiki page title for the description lookup:
    our names drop apostrophes/hyphens ('Amons Necktie') so a direct title query would miss the real
    page ('Amon's Necktie'); matching by _norm against these real titles fixes that."""
    out, cont = [], {}
    while True:
        params = {
            "action": "query", "format": "json", "list": "allpages",
            "apnamespace": "0", "apfilterredir": "nonredirects", "aplimit": "500",
        }
        params.update(cont)
        data = session.get(API, params=params, timeout=30).json()
        out += [pg["title"] for pg in data.get("query", {}).get("allpages", [])]
        if "continue" not in data:
            return out
        cont = data["continue"]


# a disambiguated page ("Mirror Shards (Omnipresent Evil)") leads with a hatnote instead of its
# description, and _owner parses no owner out of that -> no side, and the ui role filter hides the row.
_HATNOTE_RE = re.compile(r"(?i)^\s*(?:For the\b|Not to be confused\b).*?\bclick here\.\s*")


def _extracts(session, titles, sentences):
    """{normalized title: intro extract} from TextExtracts. batch of <=20 (exlimit's cap; a bigger
    one silently returns empty for the overflow)."""
    data = session.get(API, params={
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "exintro": 1, "exsentences": sentences, "exlimit": "20",
        "redirects": 1, "titles": "|".join(titles),
    }, timeout=30).json()
    out = {}
    for pg in data.get("query", {}).get("pages", {}).values():
        ex = (pg.get("extract") or "").strip().replace("\n", " ")
        if ex:
            out[_norm(pg["title"])] = ex
    return out


def fetch_descriptions(session, titles, progress=None):
    """{normalized_name: lead_sentence} for the given wiki page titles, via the TextExtracts api.
    the lead is the wiki's one-line classifier (e.g. 'Spring Clamp is an Uncommon Add-on for
    Toolboxes.'), shown as a hover tooltip in the ui, and parsed for an add-on's owner (see _owner).
    the real effect text isn't reachable from the api (it's lua-rendered into the page html), so the
    lead sentence is all we keep.
    exsentences=1 leaves the sentence split to the wiki on purpose: splitting on '.' ourselves eats
    'S.T.A.R.S. Badge' and 'Rules Set No.2' at the wrong period, which loses their owner. a
    disambiguated page's one sentence is its hatnote, so re-ask just those for more and strip it."""
    out = {}
    nbatch = (len(titles) + 19) // 20
    for bi, i in enumerate(range(0, len(titles), 20)):
        out.update(_extracts(session, titles[i:i + 20], 1))
        if progress:
            progress("fetching descriptions", bi + 1, nbatch)

    hatnoted = [t for t in titles if _HATNOTE_RE.match(out.get(_norm(t), ""))]
    for i in range(0, len(hatnoted), 20):
        for key, ex in _extracts(session, hatnoted[i:i + 20], 3).items():
            out[key] = _HATNOTE_RE.sub("", ex, count=1).strip() or out[key]
    return out


def fill_descriptions(session, rows, progress=None):
    """populate each row's `desc` in place by matching its name to a real wiki title (via
    fetch_page_titles) and pulling that page's lead sentence. one batched pass over the whole index
    after the rows are built; rows with no matching article (powers' internal icon names, a few
    event items) keep an empty desc."""
    if progress:
        progress("fetching page titles", None, None)
    norm2title = {}
    for title in fetch_page_titles(session):
        norm2title.setdefault(_norm(title), title)
    needed = sorted({norm2title[_norm(r["name"])] for r in rows if _norm(r["name"]) in norm2title})
    descmap = fetch_descriptions(session, needed, progress=progress)
    for r in rows:
        r["desc"] = descmap.get(_norm(r["name"]), "")


def dry_run(categories, limit=None):
    """report what a scrape would change, without downloading or writing anything.

    the metadata passes only (no images), so it's quick. renames are the part worth eyeballing: an
    icon is only renamed when the article self-renders that exact sprite, but this is where you'd
    catch it if the wiki ever restructures and that check starts saying yes to the wrong thing.
    """
    session = _session()
    prefixes = {p: c for p, c in PREFIXES.items() if c in categories}
    things = fetch_thing_titles(session)
    files = enumerate_icons(session, prefixes, limit=limit)
    articles, shared = resolve_articles(session, files, things)
    keyed = assign_keys(files, articles)
    redirects = fetch_redirects(session, list(articles.values()))

    renames, rekeys, unresolved = [], [], 0
    for fname, _url, key, stem, _prefix, _cat in keyed:
        article = articles.get(fname)
        if not article:
            unresolved += 1
        elif _norm(article) != _norm(prettify(stem)):
            renames.append((prettify(stem), article, fname))
        if key != stem:
            rekeys.append((stem, key, fname))

    print(f"\n{len(files)} icons ({len(articles)} resolved to an article, {unresolved} not; "
          f"an unresolved icon keeps its filename-derived name, as today)")

    print(f"\nRENAMED to the name the game uses ({len(renames)}):")
    for old, new, fname in sorted(renames, key=lambda r: r[1]):
        print(f"  {old:34} -> {new:36} [{fname}]")

    print(f"\nRE-KEYED to break a colliding filename ({len(rekeys)}):")
    for stem, key, fname in sorted(rekeys):
        print(f"  {stem:34} -> {key:36} [{fname}]")

    aliased = {a: r for a, r in redirects.items() if r}
    print(f"\nALIASES: {sum(len(v) for v in aliased.values())} wiki redirects over {len(aliased)} "
          f"articles, plus each renamed row's old name.")
    if shared:
        print(f"  ({len(shared)} sprite(s) shared by two articles, kept as aliases: "
              f"{list(shared.values())[:2]})")
    print("  the ones this was all for:")
    for old, new, _f in sorted(renames, key=lambda r: r[1]):
        rd = redirects.get(new) or []
        if rd:
            print(f"    {new:34} <- {rd + [old]}")


def main():
    ap = argparse.ArgumentParser(description="scrape dbd icons + metadata into a local library")
    ap.add_argument(
        "--categories", nargs="+", default=sorted(set(PREFIXES.values())),
        choices=sorted(set(PREFIXES.values())), help="which categories to pull"
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="max icons per category (for quick test runs)"
    )
    ap.add_argument("--force", action="store_true", help="re-download icons that already exist")
    ap.add_argument("--delay", type=float, default=0.1, help="seconds to wait between downloads")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--no-dedup", action="store_true", help="skip phash dedup during a scrape")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="resolve names/keys and report what would change, without downloading or writing"
    )
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.categories, limit=args.limit)
        return

    # one pass: download (or reuse cached) -> normalize+phash -> annotate rarity -> dedup -> write
    index, skipped = scrape(
        args.categories, args.out, args.index,
        limit=args.limit, force=args.force, delay=args.delay, dedup=not args.no_dedup
    )

    counts = Counter(row["category"] for row in index)
    filled = Counter(row["category"] for row in index if row["rarity"])
    print(f"\nindexed {len(index)} icons -> {args.index}")
    for cat, n in sorted(counts.items()):
        rar = f"  ({filled[cat]} with rarity)" if filled[cat] else ""
        print(f"  {cat:9} {n}{rar}")
    obt = Counter(row["obtainable"] for row in index)
    role = Counter(row.get("role") for row in index)
    side = Counter(row.get("side") for row in index)
    described = sum(1 for row in index if row["desc"])
    aliased = sum(1 for row in index if row.get("aliases"))
    n_alias = sum(len(row.get("aliases") or []) for row in index)
    print(f"aliases: {n_alias} over {aliased} rows (old names + wiki redirects; searched AND matched)")
    print(f"obtainable: {obt.get('normal', 0)} normal, {obt.get('event', 0)} event, "
          f"{obt.get('unavailable', 0)} unavailable   |   {described} with a description")
    print(f"role: {role.get('killer', 0)} killer, {role.get('survivor', 0)} survivor, "
          f"{role.get(None, 0)} unset")
    print(f"side: {side.get('killer', 0)} killer, {side.get('survivor', 0)} survivor, "
          f"{side.get(None, 0)} shared/unknown")
    if skipped:
        print(f"skipped {len(skipped)} asset(s) that wouldn't decode after retries: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
