"""priority-selection screen: library on the left, the priority tier stack on the right.

click a library card to add it as an item rule to the selected tier (its rarity defaults to the
card's, toggleable on the placed chip). the rule builder adds category rules. Save validates and
writes the whole config through the shared serializer; the revert button reloads the last saved file.
the screen edits the tier list's own copy and only writes app.app_state.config on Save.
"""

import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import detect, spender

from .. import config_io, theme
from ..library import Library
from ..widgets.item_card import ItemCard
from ..widgets.rule_builder import RuleBuilder
from ..widgets.tier_list import TierList
from ..widgets.windowed_list import WindowedList


class PrioritiesScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.library = None
        self.dirty = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._ensure_library()
        self._build_left()
        self._build_right()
        self._apply_filter()
        self.tiers.set_tiers((self.app.app_state.config or {}).get("priorities", []))
        self._set_dirty(False)

    def _ensure_library(self):
        if self.app.app_state.library is None:
            self.app.app_state.library = Library()
        self.library = self.app.app_state.library

    # left pane: search/filter + virtualized library
    def _build_left(self):
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(theme.PAD, theme.PAD // 2), pady=theme.PAD)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        self.search = ctk.CTkEntry(bar, placeholder_text="search items...")
        self.search.pack(side="left", fill="x", expand=True, padx=(0, theme.PAD))
        self.search.bind("<KeyRelease>", lambda e: self._apply_filter())
        self.category = ctk.CTkOptionMenu(
            bar, width=110, values=["all"] + sorted(spender.VALID_CATEGORIES),
            command=lambda v: self._apply_filter())
        self.category.set("all")
        self.category.pack(side="left", padx=(0, theme.PAD))
        self.rarity = ctk.CTkOptionMenu(
            bar, width=110, values=["all"] + list(detect.RARITIES) + ["none"],
            command=lambda v: self._apply_filter())
        self.rarity.set("all")
        self.rarity.pack(side="left")

        self.list = WindowedList(left, make_card=self._make_card, row_h=64, buffer=6)
        self.list.grid(row=1, column=0, sticky="nsew", padx=theme.PAD, pady=(0, theme.PAD))

    # right pane: save/revert + tier stack + category rule builder
    def _build_right(self):
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(theme.PAD // 2, theme.PAD), pady=theme.PAD)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(right, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(top, text="Priorities", font=theme.FONT_TITLE).pack(side="left")
        self.revert_btn = ctk.CTkButton(top, text="↺ revert", width=80, command=self._revert)
        self.revert_btn.pack(side="right", padx=(theme.PAD, 0))
        self.save_btn = ctk.CTkButton(top, text="Save", width=80, command=self._save)
        self.save_btn.pack(side="right")

        self.tiers = TierList(right, self.library, on_change=lambda: self._set_dirty(True))
        self.tiers.grid(row=1, column=0, sticky="nsew", padx=theme.PAD, pady=(0, theme.PAD))

        self.builder = RuleBuilder(right, on_add=self._add_category_rule)
        self.builder.grid(row=2, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))

    # library wiring
    def _make_card(self, master):
        return ItemCard(master, self.library, on_activate=self._on_card)

    def _on_card(self, row, button):
        if button != 1:
            return
        rule = {"type": "item", "name": row.get("name")}
        if row.get("rarity"):
            rule["rarity"] = row["rarity"]   # default to the card's rarity, toggleable on the chip
        self.tiers.add_rule(rule)

    def _add_category_rule(self, rule):
        self.tiers.add_rule(rule)

    def _apply_filter(self):
        rows = self.library.filter(self.search.get(), self.category.get(), self.rarity.get())
        self.list.set_model(rows)

    # save / revert
    def _set_dirty(self, dirty):
        self.dirty = dirty
        self.save_btn.configure(text="Save *" if dirty else "Save")

    def _save(self):
        cfg = dict(self.app.app_state.config or {})
        cfg["priorities"] = self.tiers.cleaned_tiers()
        try:
            config_io.save(cfg)
        except ValueError as e:
            messagebox.showerror("invalid priorities", str(e))
            return
        self.app.app_state.config = cfg
        self._set_dirty(False)

    def _revert(self):
        self.app.app_state.load_config()
        if self.app.app_state.config_error:
            messagebox.showerror("config error", self.app.app_state.config_error)
        cfg = self.app.app_state.config or {}
        self.tiers.set_tiers(cfg.get("priorities", []))
        self._set_dirty(False)
