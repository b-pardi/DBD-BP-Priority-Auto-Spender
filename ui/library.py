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
from src.node import normalize_name

from .theme import THUMB_PX


class Library:
    def __init__(self, rows=None, thumb_px=THUMB_PX, cache_size=400):
        if rows is None:
            rows, _ = detect.load_index()
        self.rows = rows
        self.thumb_px = thumb_px
        self.cache_size = cache_size
        # sprite "file" is relative to the index file's dir, the same base detect resolves against.
        self._base = Path(detect.DEFAULT_INDEX).parent
        self._cache = OrderedDict()  # row["file"] -> CTkImage, LRU

    def filter(self, query="", category="all", rarity="all"):
        """rows matching the search box + dropdowns. case/punctuation-insensitive name search
        (folded like the detector via node.normalize_name), category exact, rarity exact with a
        'none' bucket for null-rarity rows (perks/powers/visceral)."""
        q = normalize_name(query)
        out = []
        for r in self.rows:
            if category != "all" and r.get("category") != category:
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
