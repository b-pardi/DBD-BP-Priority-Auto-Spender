"""in-memory model of the icon library plus a lazy thumbnail cache.

rows come straight from data/icons_index.json (key, name, category, rarity, file) via
detect.load_index, so the ui and the detector share one source of truth. thumbnails load lazily,
downscaled to a small size, and are held in an LRU keyed by the sprite path, so we never keep ~1609
full 256x256 sprites resident (that would be hundreds of MB). filtering is a plain predicate over
the rows; the windowed list renders only the visible slice.
"""

from collections import OrderedDict
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from src import detect
from src.node import dedup_index_rows, normalize_name

from .theme import THUMB_PX


class Library:
    def __init__(self, rows=None, thumb_px=THUMB_PX, cache_size=400):
        if rows is None:
            rows, _ = detect.load_index()
        # drop the wiki's swapped-name duplicate uploads (e.g. flashlightSport vs sportFlashlight),
        # which otherwise show up twice: once as the canonical card and once as a bare icon with no
        # rarity or tooltip. done at the ui boundary so detection's positional row<->phash arrays are
        # untouched; a re-scrape persists the same dedup to the index file.
        self.rows = dedup_index_rows(rows)
        self.thumb_px = thumb_px
        self.cache_size = cache_size
        # sprite "file" is relative to the index file's dir, the same base detect resolves against.
        self._base = Path(detect.DEFAULT_INDEX).parent
        self._cache = OrderedDict()  # row["file"] -> CTkImage, LRU

    def filter(self, query="", category="all", rarity="all", role="all", show_unavailable=False):
        """rows matching the search box + dropdowns. case/punctuation-insensitive name search
        (folded like the detector via node.normalize_name), category exact, rarity exact with a
        'none' bucket for null-rarity rows (perks/powers/visceral).

        role ('all'|'killer'|'survivor') filters by who plays the glyph. items (survivor), powers
        (killer), and perks (per the wiki role categories) carry a role; add-ons instead carry a
        `side` (whose power/item they belong to), so we fall back to that when role is absent.
        offerings have neither on the wiki (and most suit either side), so they pass both filters
        rather than being wrongly hidden. rows from an index predating these scrapes are all null
        and likewise unaffected.

        show_unavailable=False (the default) hides glyphs you can't buy in a current bloodweb:
        'event' (past-event skins) and 'unavailable' (killer powers, retired offerings). rows from
        an index predating the obtainability scrape lack the field and count as 'normal'."""
        q = normalize_name(query)
        out = []
        for r in self.rows:
            if not show_unavailable and r.get("obtainable", "normal") != "normal":
                continue
            if category != "all" and r.get("category") != category:
                continue
            if role != "all":
                rr = r.get("role") or r.get("side")
                if rr is not None and rr != role:
                    continue
            if rarity == "none":
                if r.get("rarity") is not None:
                    continue
            elif rarity != "all" and r.get("rarity") != rarity:
                continue
            if q and q not in normalize_name(r.get("name", "")):
                continue
            out.append(r)
        return out

    def thumbnail(self, row):
        """a small CTkImage for a row, cached LRU by sprite path. None if the sprite won't load.
        created lazily at render time (a Tk root exists by then)."""
        fkey = row.get("file")
        img = self._cache.get(fkey)
        if img is not None:
            self._cache.move_to_end(fkey)
            return img
        try:
            pil = Image.open(self._base / fkey).convert("RGBA")
        except Exception:
            return None
        pil.thumbnail((self.thumb_px, self.thumb_px), Image.LANCZOS)
        img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self._cache[fkey] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict the least-recently-used
        return img

    def lookup_row(self, name):
        """first index row whose name folds to the same key as `name`, or None.
        used by a placed item-rule chip to find its thumbnail + library rarity."""
        nn = normalize_name(name)
        for r in self.rows:
            if normalize_name(r.get("name", "")) == nn:
                return r
        return None

    def lookup_rarity(self, name):
        """library rarity for an item name (None for perks/powers/visceral or unknown)."""
        row = self.lookup_row(name)
        return row.get("rarity") if row else None

    def clear_thumbnail_cache(self):
        """drop the in-memory thumbnail cache (used after a re-scrape changes the sprites)."""
        self._cache.clear()

    def reload(self):
        """re-read the index from disk and drop cached thumbnails, in place.
        keeps the same Library object so widgets holding a reference (cards, chips) pick up the new
        rows without being rebuilt. used after a scrape, including the first-run one that fills an
        initially empty library."""
        rows, _ = detect.load_index()
        self.rows = dedup_index_rows(rows)
        self._base = Path(detect.DEFAULT_INDEX).parent
        self._cache.clear()
