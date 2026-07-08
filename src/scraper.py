"""scrape dbd icon assets + metadata from deadbydaylight.wiki.gg into a local library.

the whole detector leans on this: we identify on-screen bloodweb icons by matching them against
these exact sprites. we pull every category (perks, items, addons, offerings, powers), even ones
we'd never buy, so unwanted nodes still get identified instead of causing false matches.

source is the mediawiki api (action=query&list=allimages) filtered by the wiki's icon filename
prefixes (cargo tables aren't exposed here, but the prefixes reliably enumerate each category).
rarity isn't in the file metadata, so it's pulled in the same pass from the wiki's rarity categories
(item/addon/offering only; perks/powers have none and stay null). it's a soft cross-check; detection
reads rarity live from the on-screen disk color.

one pass does it all: download (or reuse cached) -> normalize+phash -> annotate rarity -> dedup ->
write. just `python -m src.scraper`.

writes:
  data/icons/<category>/<key>.png   the raw sprite
  data/icons_index.json             one row per icon: key, name, category, rarity, obtainable,
                                    role, owner, side, desc, file, phash, url

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
from collections import Counter
from pathlib import Path

import imagehash
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .node import dedup_index_rows

API = "https://deadbydaylight.wiki.gg/api.php"

# wiki icon filename prefix -> our category name; the prefixes namespace the sprite files, so they
# double as a clean way to enumerate each category. the wiki also files some newer/special assets
# under an inconsistent 'Icons' (extra s) prefix, so we enumerate those too (gold event offerings,
# plus a batch of newer perks) or we'd silently miss them.
PREFIXES = {
    "IconPerks_": "perk",
    "IconItems_": "item",
    "IconAddon_": "addon",
    "IconFavors_": "offering",
    "IconPowers_": "power",
    "IconsFavors_": "offering",   # Icons-variant prefix: special/event offerings (10thAnniversary, ...)
    "IconsPerks_": "perk",        # Icons-variant prefix: newer perks missing from IconPerks_
}

# prefixes whose icons are the gold "event" disk tier; they carry no wiki rarity category, so tag
# them 'event' directly (lets the soft cross-check + priority rules target them).
EVENT_PREFIXES = {"IconsFavors_"}

# event icons the curated Icon* prefixes don't cover: the newest event content is on the wiki only as
# raw 'T_UI_' texture uploads (no curated IconItems_ redirect yet), so prefix enumeration never sees
# it. list those explicitly by exact File: name -> (key, category), since the in-game names
# ('Banquet ...') don't derive from the texture stem (identity is read from the glyph, so the key is
# just a stable id). all tagged 'event'; add a line per new event icon and re-scrape to fetch + hash.
EXTRA_EVENT_ICONS = {
    "T_UI_iconItems_toolbox_anniversary2026.png":     ("banquetToolbox",   "item"),
    "T_UI_iconItems_flashlight_anniversary2026.png":  ("banquetFlashlight", "item"),
    "T_UI_iconItems_medkit_anniversary2026.png":      ("banquetMedKit",    "item"),
    "T_UI_iconItems_poison_anniversary2026.png":      ("banquetPoison",    "item"),
}

# repo root is two levels up (src/scraper.py -> repo root), so defaults land in data/
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "icons"
DEFAULT_INDEX = ROOT / "data" / "icons_index.json"

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
        # relative to the index file so the library stays portable; detect resolves it
        # against index_path.parent
        "file": os.path.relpath(dest, index_path.parent).replace("\\", "/"),
        "phash": icon_phash(img),
        "url": url,
    }


def scrape(categories, out_dir: Path, index_path: Path, limit=None, force=False, delay=0.1, dedup=True):
    session = _session()
    prefixes = {p: c for p, c in PREFIXES.items() if c in categories}
    # rarity in the same pass: pull the wiki's rarity categories once up front, then look each
    # icon up as we build its row, so one scrape yields a fully-populated index (no second pass).
    # only item/addon/offering have rarity categories, so skip the fetch unless one's selected.
    rmap = fetch_rarity_map(session) if any(c in RARITY_TYPES for c in categories) else {}
    # obtainability (retired/event sets) in the same up-front pass as rarity, so one scrape yields a
    # fully-annotated index. cheap (a few category calls); always pulled since any category can hold
    # event/retired glyphs and powers are flagged by category regardless.
    obmap = fetch_obtainability(session)
    # role (killer/survivor) for the ui filter, same up-front-map pattern: only perks need the wiki
    # lookup, so skip the category calls unless perks are selected (items/powers go by category).
    rolemap = fetch_perk_roles(session) if "perk" in categories else {}
    index = []
    skipped = []
    for prefix, category in prefixes.items():
        for entry in tqdm(list_icons(session, prefix, limit=limit), desc=category, unit="icon"):
            key = Path(entry["name"][len(prefix):]).stem   # IconPerks_EyesOfBelmont.png -> EyesOfBelmont
            name = prettify(key)
            # event prefixes -> 'event' (gold tier, no wiki rarity category); else the wiki
            # rarity, or null for perks/powers and ~visceral top-tier icons
            rarity = "event" if prefix in EVENT_PREFIXES else rmap.get(category, {}).get(_norm(name))
            obtainable = _obtainable(category, name, rarity, obmap)
            role = _role(category, name, rolemap)
            row = _index_one(session, entry["url"], key, name, category, rarity, obtainable, role,
                             out_dir, index_path, force, delay, skipped)
            if row:
                index.append(row)

    # explicit event icons the prefixes miss (raw T_UI_ textures); look them up by exact name
    # and tag 'event'. skip ones whose category wasn't requested so --categories stays honest.
    extras = {f: kc for f, kc in EXTRA_EVENT_ICONS.items() if kc[1] in categories}
    if extras:
        extra_urls = resolve_image_urls(session, list(extras))
        for fname, (key, category) in extras.items():
            url = extra_urls.get(fname)
            if url is None:                  # not found on the wiki (renamed/removed), log + skip
                skipped.append(fname)
                continue
            obtainable = _obtainable(category, prettify(key), "event", obmap)
            role = _role(category, prettify(key), rolemap)
            row = _index_one(session, url, key, prettify(key), category, "event", obtainable, role,
                             out_dir, index_path, force, delay, skipped)
            if row:
                index.append(row)

    # fill the wiki lead-sentence tooltips in one batched pass now that every row's name is known
    fill_descriptions(session, index)

    # owner + side, last: owner is parsed from the add-on lead sentence (so it needs desc filled
    # first), and the survivor item types that classify an add-on's side are scraped live here.
    fill_sources(index, fetch_survivor_item_types(session))

    if dedup:
        index = dedup_index_rows(index)
    index.sort(key=lambda row: (row["category"], row["key"]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index, skipped


def _norm(name):
    """squash a name to lowercase alphanumerics so the wiki page 'Amanda's Letter' and our
    prettified key 'Amandas Letter' compare equal."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def category_members(session, cat_title):
    """all page titles in a wiki category, paged via list=categorymembers (500/call)."""
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


def fetch_descriptions(session, titles):
    """{normalized_name: lead_sentence} for the given wiki page titles, via the TextExtracts api.
    the lead is the wiki's one-line classifier (e.g. 'Spring Clamp is an Uncommon Add-on for
    Toolboxes.'), shown as a hover tooltip in the ui. the real effect text isn't reachable from the
    api (it's lua-rendered into the page html), so the lead sentence is all we keep. batch in 20s:
    exlimit caps at 20 for extracts, and a bigger batch silently returns empty for the overflow."""
    out = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        data = session.get(API, params={
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "exintro": 1, "exsentences": 1, "exlimit": "20",
            "redirects": 1, "titles": "|".join(batch),
        }, timeout=30).json()
        for pg in data.get("query", {}).get("pages", {}).values():
            ex = (pg.get("extract") or "").strip().replace("\n", " ")
            if ex:
                out[_norm(pg["title"])] = ex
    return out


def fill_descriptions(session, rows):
    """populate each row's `desc` in place by matching its name to a real wiki title (via
    fetch_page_titles) and pulling that page's lead sentence. one batched pass over the whole index
    after the rows are built; rows with no matching article (powers' internal icon names, a few
    event items) keep an empty desc."""
    norm2title = {}
    for title in fetch_page_titles(session):
        norm2title.setdefault(_norm(title), title)
    needed = sorted({norm2title[_norm(r["name"])] for r in rows if _norm(r["name"]) in norm2title})
    descmap = fetch_descriptions(session, needed)
    for r in rows:
        r["desc"] = descmap.get(_norm(r["name"]), "")


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
    args = ap.parse_args()

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
