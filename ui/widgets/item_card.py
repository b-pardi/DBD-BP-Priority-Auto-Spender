"""the library "rectangle": a small thumbnail, the name, the rarity, and a rarity accent bar.

reused by the windowed list, so it exposes bind_row(row) to re-skin in place rather than being
rebuilt per scroll. a left/right click forwards the bound row to on_activate(row, button), so the
priorities screen can add it (left) without the card knowing what a tier is.
"""

import customtkinter as ctk

from ..theme import (
    ACCENT_W, FONT_BODY, FONT_SMALL, MUTED_TEXT_COLOR, PAD, ROW_H, THUMB_PX, rarity_color,
)
from .dragdrop import Draggable
from .tooltip import bind_tooltip


class ItemCard(ctk.CTkFrame):
    def __init__(self, master, library, height=ROW_H, on_activate=None,
                 on_drop=None, on_drag_hover=None):
        # ctk 6.x: the height must be set here, and propagation turned off, or the card grows to
        # fit its children (the accent CTkFrame alone defaults to 200px) and the rows overlap.
        super().__init__(master, height=height, corner_radius=6)
        self.pack_propagate(False)
        self.library = library
        self.on_activate = on_activate
        self.on_drop = on_drop            # (row, x_root, y_root) when the card is dragged onto a tier
        self.row = None

        self.accent = ctk.CTkFrame(self, width=ACCENT_W, corner_radius=0)  # rarity tint bar
        self.accent.pack(side="left", fill="y")
        self.icon = ctk.CTkLabel(self, text="", width=THUMB_PX)
        self.icon.pack(side="left", padx=(PAD, PAD // 2))
        self.name = ctk.CTkLabel(self, text="", anchor="w", font=FONT_BODY)
        self.name.pack(side="left", fill="x", expand=True)
        self._name_color = self.name.cget("text_color")  # default, restored for obtainable rows
        self.rar = ctk.CTkLabel(self, text="", anchor="e", font=FONT_SMALL, width=72)
        self.rar.pack(side="right", padx=PAD)

        # left button is drag-aware (tk doesn't bubble, so every sub-widget is a source): a plain
        # click still adds the row to the selected tier, but a drag drops it onto a specific tier.
        # right-click stays a plain activate.
        cells = (self, self.accent, self.icon, self.name, self.rar)
        Draggable(
            cells,
            get_ghost_text=lambda: (self.row or {}).get("name", "item"),
            on_click=lambda: self._activate(1),
            on_drop=self._drop,
            on_hover=on_drag_hover,
        )
        for w in cells:
            w.bind("<Button-3>", lambda e: self._activate(3))
        # hover tooltip: the wiki lead sentence for whatever row this (recycled) card holds.
        bind_tooltip(
            [self, self.accent, self.icon, self.name, self.rar],
            lambda: (self.row or {}).get("desc", ""),
        )

    def bind_row(self, row):
        """re-skin the card to a new row (the recycling hook the windowed list calls)."""
        self.row = row
        self.icon.configure(image=self.library.thumbnail(row))  # CTkImage or None
        self.name.configure(text=row.get("name", "?"))
        rar = row.get("rarity")
        self.accent.configure(fg_color=rarity_color(rar))
        # obtainability marker: rows that can't appear in a current bloodweb show their state in the
        # rarity slot and dim, so the few that surface (when the filter reveals them) read as unbuyable.
        obt = row.get("obtainable", "normal")
        self.rar.configure(text={"event": "event", "unavailable": "n/a"}.get(obt, rar or ""))
        self.name.configure(text_color=MUTED_TEXT_COLOR if obt != "normal" else self._name_color)

    def _activate(self, button):
        if self.on_activate and self.row is not None:
            self.on_activate(self.row, button)

    def _drop(self, x_root, y_root):
        if self.on_drop and self.row is not None:
            self.on_drop(self.row, x_root, y_root)
