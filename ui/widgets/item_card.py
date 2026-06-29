"""the library "rectangle": a small thumbnail, the name, the rarity, and a rarity accent bar.

reused by the windowed list, so it exposes bind_row(row) to re-skin in place rather than being
rebuilt per scroll. a left/right click forwards the bound row to on_activate(row, button), so the
priorities screen can add it (left) without the card knowing what a tier is.
"""

import customtkinter as ctk

from ..theme import FONT_BODY, FONT_SMALL, PAD, THUMB_PX, rarity_color


class ItemCard(ctk.CTkFrame):
    def __init__(self, master, library, on_activate=None):
        super().__init__(master, corner_radius=6)
        self.library = library
        self.on_activate = on_activate
        self.row = None

        self.accent = ctk.CTkFrame(self, width=5, corner_radius=0)  # rarity tint bar
        self.accent.pack(side="left", fill="y")
        self.icon = ctk.CTkLabel(self, text="", width=THUMB_PX)
        self.icon.pack(side="left", padx=PAD)
        self.name = ctk.CTkLabel(self, text="", anchor="w", font=FONT_BODY)
        self.name.pack(side="left", fill="x", expand=True)
        self.rar = ctk.CTkLabel(self, text="", anchor="e", font=FONT_SMALL, width=80)
        self.rar.pack(side="right", padx=PAD)

        # bind clicks on every sub-widget so the whole card is clickable (tk doesn't bubble).
        for w in (self, self.accent, self.icon, self.name, self.rar):
            w.bind("<Button-1>", lambda e: self._activate(1))
            w.bind("<Button-3>", lambda e: self._activate(3))

    def bind_row(self, row):
        """re-skin the card to a new row (the recycling hook the windowed list calls)."""
        self.row = row
        self.icon.configure(image=self.library.thumbnail(row))  # CTkImage or None
        self.name.configure(text=row.get("name", "?"))
        rar = row.get("rarity")
        self.rar.configure(text=rar or "")
        self.accent.configure(fg_color=rarity_color(rar))

    def _activate(self, button):
        if self.on_activate and self.row is not None:
            self.on_activate(self.row, button)
