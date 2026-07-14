"""in-memory model of the icon library plus its thumbnail cache.

rows come straight from data/icons_index.json (key, name, category, rarity, file) via
detect.load_rows, so the ui and the detector share one source of truth. filtering is a plain
predicate over the rows; the windowed list renders only the visible slice.

thumbnails are downscaled to a small size and cached by sprite path. the whole library at 34px is
only ~7mb of pixels, so we cache all of it rather than evicting: a bounded cache just meant scrolling
the 1600-row list kept re-decoding sprites it had already seen (~1.9ms each, mostly png decode),
which is what made scrolling feel sluggish. prewarm() does that decoding up front on a worker thread
so the first scroll through is already warm -- PIL work is thread-safe, but tk objects are not, so
the thread only produces PIL images and the CTkImage wrapper is built on the main thread (free,
~1us) the first time a row is actually drawn.
"""

import re
import threading
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from src import detect
from src.node import dedup_index_rows, normalize_name, row_names

from .theme import THUMB_PX

def tooltip_text(row):
    """the hover-tooltip body for a row: the wiki lead sentence plus, when the scrape rendered
    one, the actual gameplay text underneath (scraper.fill_effects; '' on pre-effect indexes)."""
    if not row:
        return ""
    return "\n\n".join(p for p in (row.get("desc") or "", row.get("effect") or "") if p)


class Library:
    def __init__(self, rows=None, thumb_px=THUMB_PX):
        if rows is None:
            rows = detect.load_rows()
        # drop the wiki's swapped-name duplicate uploads (e.g. flashlightSport vs sportFlashlight),
        # which otherwise show up twice: once as the canonical card and once as a bare icon with no
        # rarity or tooltip. done at the ui boundary so detection's positional row<->phash arrays are
        # untouched; a re-scrape persists the same dedup to the index file.
        self.rows = dedup_index_rows(rows)
        self.thumb_px = thumb_px
        # sprite "file" is relative to the index file's dir, the same base detect resolves against.
        self._base = Path(detect.DEFAULT_INDEX).parent
        self._cache = {}      # row["file"] -> CTkImage (main thread only)
        self._pil = {}        # row["file"] -> PIL thumbnail, filled by the prewarm thread
        self._prewarm = None  # the running prewarm thread, if any
        self._reindex()

    def _reindex(self):
        """name-key -> row, so a placed chip finds its thumbnail without scanning all 1600 rows.
        lookup_row used to be a linear scan folding every name with a regex, which cost ~1ms a call
        and ran once per chip built.
        aliases are indexed too (in a second pass, so a canonical name always beats another row's
        alias), which is what lets a config rule saved under an old name -- '10th Anniversary' before
        the wiki article resolved it to 'Toothy Torte' -- still find its row."""
        self._by_key = {}
        for r in self.rows:
            self._by_key.setdefault(normalize_name(r.get("name", "")), r)
        for r in self.rows:
            for alias in r.get("aliases") or []:
                self._by_key.setdefault(normalize_name(alias), r)
        self._killers = self._killer_labels()

    def _killer_labels(self):
        """{filter label: set of owner keys} for every killer with add-ons in the index.
        the label is the row's own scraped `killer` name (joined off the wiki's lua data modules,
        see scraper.fetch_killer_map), falling back to the power name parsed off one of its add-ons'
        lead sentences ("... is a Rare Add-on for Bear Trap.") -- which is what an index scraped
        before that field existed reads as, so those still filter, just labelled by power.
        survivor item families (medkits, flashlights, ...) share the addon category but sit on the
        survivor side, so they never land here."""
        labels = {}
        for r in self.rows:
            if r.get("category") != "addon" or r.get("side") != "killer":
                continue
            owner = r.get("owner")
            if not owner or owner == "unused":
                continue
            label = r.get("killer")
            if label is None:
                m = re.search(r"Add-on for (.+?)\.", r.get("desc") or "")
                label = m.group(1) if m else owner
            labels.setdefault(label, set()).add(owner)
        return labels

    def killer_names(self):
        """sorted filter labels for the per-killer role dropdown."""
        return sorted(self._killers)

    def filter(self, query="", category="all", rarity="all", role="all",
               show_event=True, show_na=False):
        """rows matching the search box + dropdowns. case/punctuation-insensitive name search
        (folded like the detector via node.normalize_name), category exact, rarity exact with a
        'none' bucket for null-rarity rows (perks/powers).

        the search also looks through a row's aliases (the wiki's redirects to its article, plus the
        name we used before that article was resolved), so the offering the game calls "Toothy Torte"
        is found by that name as well as by the "10th Anniversary" its sprite file is named after, and
        a perk renamed off its lost licence still answers to what people call it ("Decisive Strike",
        "DS" -> Will to Live).

        role ('all'|'killer'|'survivor'|a killer name from killer_names()) filters by who plays the
        glyph, strictly: a side pick shows only rows whose role/side is exactly that side. items
        (survivor), powers (killer), and perks (per the wiki role categories) carry a role; add-ons
        instead carry a `side` (whose power/item they belong to), so we fall back to that when role
        is absent. offerings (and any row with neither field) have no side on the wiki, so a
        killer/survivor pick hides them; only 'all' shows them. a specific killer narrows further:
        killer-side rows, with add-ons kept only when they belong to that killer's power — i.e. that
        killer's own bloodweb content (shared killer perks included).
        rows from an index predating the role/side scrape are all null and so only appear under 'all'.

        the reveal toggles hide glyphs you can't buy in a current bloodweb: show_event (default on)
        governs the 'event' tier (past-event skins), show_na (default off) the 'unavailable' bucket
        (killer powers, retired offerings). rows from an index predating the obtainability scrape
        lack the field and count as 'normal'."""
        q = normalize_name(query)
        side, owners = role, None
        if role not in ("all", "killer", "survivor"):   # a specific killer from killer_names()
            side, owners = "killer", self._killers.get(role, set())
        out = []
        for r in self.rows:
            obt = r.get("obtainable", "normal")
            if obt == "event" and not show_event:
                continue
            if obt == "unavailable" and not show_na:
                continue
            if category != "all" and r.get("category") != category:
                continue
            if side != "all":
                rr = r.get("role") or r.get("side")
                if rr != side:   # strict: no side (offerings/old rows) only shows under 'all'
                    continue
                if owners is not None and r.get("category") == "addon" \
                        and r.get("owner") not in owners:
                    continue
            if rarity == "none":
                if r.get("rarity") is not None:
                    continue
            elif rarity != "all" and r.get("rarity") != rarity:
                continue
            if q and not any(q in n for n in row_names(r)):
                continue
            out.append(r)
        return out

    # thumbnails
    def _load_pil(self, fkey):
        """decode one sprite to a thumbnail-sized PIL image. safe to call off the main thread."""
        pil = Image.open(self._base / fkey)
        if pil.mode != "RGBA":
            pil = pil.convert("RGBA")
        pil.thumbnail((self.thumb_px, self.thumb_px), Image.LANCZOS)
        return pil

    def thumbnail(self, row):
        """a small CTkImage for a row, cached by sprite path. None if the sprite won't load.
        the CTkImage is created lazily at render time (a Tk root exists by then); if prewarm has
        already decoded the sprite this is just the wrapper, otherwise it decodes inline."""
        fkey = row.get("file")
        img = self._cache.get(fkey)
        if img is not None:
            return img
        pil = self._pil.get(fkey)
        if pil is None:
            try:
                pil = self._load_pil(fkey)
            except Exception:
                return None
            self._pil[fkey] = pil
        img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self._cache[fkey] = img
        return img

    def prewarm(self):
        """decode every sprite to a thumbnail on a worker thread, so scrolling never waits on a png.
        idempotent and fire-and-forget: it only fills the PIL dict (no tk objects), a dict set is
        atomic under the gil, and a row raced by thumbnail() just gets decoded twice. harmless."""
        if self._prewarm is not None and self._prewarm.is_alive():
            return

        def work(rows):
            for r in rows:
                fkey = r.get("file")
                if fkey in self._pil:
                    continue
                try:
                    self._pil[fkey] = self._load_pil(fkey)
                except Exception:
                    pass  # missing/corrupt sprite: thumbnail() will return None for it

        self._prewarm = threading.Thread(target=work, args=(list(self.rows),), daemon=True)
        self._prewarm.start()

    def lookup_row(self, name):
        """the index row whose name folds to the same key as `name`, or None.
        used by a placed item-rule chip to find its thumbnail + library rarity."""
        return self._by_key.get(normalize_name(name))

    def lookup_rarity(self, name):
        """library rarity for an item name (None for perks/powers/visceral or unknown)."""
        row = self.lookup_row(name)
        return row.get("rarity") if row else None

    def clear_thumbnail_cache(self):
        """drop the cached thumbnails (used after a re-scrape changes the sprites)."""
        self._cache.clear()
        self._pil.clear()

    def reload(self):
        """re-read the index from disk and drop cached thumbnails, in place.
        keeps the same Library object so widgets holding a reference (cards, chips) pick up the new
        rows without being rebuilt. used after a scrape, including the first-run one that fills an
        initially empty library."""
        self.rows = dedup_index_rows(detect.load_rows())
        self._base = Path(detect.DEFAULT_INDEX).parent
        self.clear_thumbnail_cache()
        self._reindex()
        self.prewarm()
