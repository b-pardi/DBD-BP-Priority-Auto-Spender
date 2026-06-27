"""scrape dbd icon assets + metadata from deadbydaylight.wiki.gg into a local library.

the whole detector leans on this: we identify on-screen bloodweb icons by matching them
against these exact sprites. we pull every category (perks, items, addons, offerings,
powers), even ones we'd never buy, so unwanted nodes still get identified instead of
causing false matches.

source is the mediawiki api (action=query&list=allimages) filtered by the wiki's icon
filename prefixes. cargo tables aren't exposed on this wiki, but the prefixes are a
reliable way to enumerate each category. rarity isn't in the file metadata, so it's pulled
in the same pass from the wiki's rarity categories (item/addon/offering only; perks/powers
have none and stay null). it's a soft cross-check; detection reads rarity live from the
on-screen disk color.

one pass does it all: download (or reuse cached) -> normalize+phash -> annotate rarity ->
dedup -> write. just `python -m src.scraper`.

writes:
  data/icons/<category>/<key>.png   the raw sprite
  data/icons_index.json             one row per icon: key, name, category, rarity, file, phash, url

the phash is precomputed here (via normalize_sprite, kept in sync with detect.normalize_glyph)
so nearest-neighbor lookup at detect time is a cheap hamming distance over ~2k templates,
instead of full template matching on every candidate. if the framing changes on either side,
re-run the scrape to recompute hashes from the cached pngs, no re-download needed.
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

API = "https://deadbydaylight.wiki.gg/api.php"

# wiki icon filename prefix -> our category name. these prefixes are how the wiki
# namespaces its sprite files, so they double as a clean way to enumerate each category.
# the wiki also files some newer/special assets under an inconsistent 'Icons' (extra s)
# prefix, so we enumerate those too or we'd silently miss them (the gold event offerings,
# plus a batch of newer perks).
PREFIXES = {
    "IconPerks_": "perk",
    "IconItems_": "item",
    "IconAddon_": "addon",
    "IconFavors_": "offering",
    "IconPowers_": "power",
    "IconsFavors_": "offering",   # Icons-variant prefix: special/event offerings (10thAnniversary, ...)
    "IconsPerks_": "perk",        # Icons-variant prefix: newer perks missing from IconPerks_
}

# prefixes whose icons are the gold "event" disk tier -- they carry no wiki rarity category,
# so tag them 'event' directly (lets the soft cross-check + priority rules target them).
EVENT_PREFIXES = {"IconsFavors_"}

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

    kept in sync with detect.normalize_glyph by hand on purpose -- the two sides do different
    cleanup (detect removes the disk, handles off-center crops) but must agree on this final
    framing. returns an 'L' (grayscale) image ready to phash.
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


def scrape(categories, out_dir: Path, index_path: Path, limit=None, force=False, delay=0.1, dedup=True):
    session = _session()
    prefixes = {p: c for p, c in PREFIXES.items() if c in categories}
    # rarity in the same pass: pull the wiki's rarity categories once up front, then look each
    # icon up as we build its row, so one scrape yields a fully-populated index (no second pass).
    # only item/addon/offering have rarity categories, so skip the fetch unless one's selected.
    rmap = fetch_rarity_map(session) if any(c in RARITY_TYPES for c in categories) else {}
    index = []
    skipped = []
    for prefix, category in prefixes.items():
        icons = list_icons(session, prefix, limit=limit)
        for entry in tqdm(icons, desc=category, unit="icon"):
            fname = entry["name"]            # e.g. IconPerks_EyesOfBelmont.png
            stem = fname[len(prefix):]       # EyesOfBelmont.png
            key = Path(stem).stem            # EyesOfBelmont
            dest = out_dir / category / f"{key}.png"
            if dest.exists() and not force:
                img = Image.open(dest).convert("RGBA")
            else:
                img = download_icon(session, entry["url"], dest)
                if img is None:              # undecodable asset, log it and move on
                    skipped.append(fname)
                    continue
                time.sleep(delay)            # be polite to the wiki between downloads
            name = prettify(key)
            index.append({
                "key": key,
                "name": name,
                "category": category,
                # event prefixes -> 'event' (gold tier, no wiki rarity category); else the
                # wiki rarity, or null for perks/powers and ~visceral top-tier icons
                "rarity": "event" if prefix in EVENT_PREFIXES else rmap.get(category, {}).get(_norm(name)),
                # relative to the index file so the library stays portable; detect
                # resolves it against index_path.parent
                "file": os.path.relpath(dest, index_path.parent).replace("\\", "/"),
                "phash": icon_phash(img),
                "url": entry["url"],
            })
    if dedup:
        index = dedup_index(index)
    index.sort(key=lambda row: (row["category"], row["key"]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index, skipped


def dedup_index(rows):
    """drop rows whose phash exactly duplicates another in the same category, keeping the
    most informative copy (rarity known beats null). the wiki uploads the same sprite under
    several filenames (sportFlashlight vs flashlightSport, Telephone vs telephone), which
    bloats the match pool and can make id_icon's 2nd-best margin meaninglessly tiny. scoped
    per category so a rare cross-category hash collision isn't merged."""
    kept = {}
    for r in rows:
        key = (r["category"], r["phash"])
        cur = kept.get(key)
        if cur is None or (cur.get("rarity") is None and r.get("rarity") is not None):
            kept[key] = r
    return list(kept.values())


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
    if skipped:
        print(f"skipped {len(skipped)} asset(s) that wouldn't decode after retries: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
