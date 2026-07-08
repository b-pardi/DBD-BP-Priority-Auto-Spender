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
from ..templates import TEMPLATE_LABELS, TIER_TEMPLATES
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

        cfg = self.app.app_state.config or {}
        # profiles: named priority lists (config_io.ensure_profiles guarantees these exist). the tier
        # editor always edits the active profile; switching stashes the current edits back first.
        self.profiles = {n: spender.copy_tiers(tiers)
                         for n, tiers in (cfg.get("profiles") or {}).items()}
        if not self.profiles:
            self.profiles = {config_io.DEFAULT_PROFILE: spender.copy_tiers(cfg.get("priorities", []))}
        self.active = cfg.get("active_profile") or next(iter(self.profiles))

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._ensure_library()
        self._build_left()
        self._build_right()
        self._apply_filter()
        self.tiers.set_tiers(self.profiles.get(self.active, []))
        self._set_dirty(False)

    def _ensure_library(self):
        if self.app.app_state.library is None:
            try:
                self.app.app_state.library = Library()
            except Exception:
                # no index on disk yet (first run): fall back to an empty library so the screen
                # still builds; the nav "Update icons" button / first-run prompt fetches it.
                self.app.app_state.library = Library(rows=[])
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
        self.rarity.pack(side="left", padx=(0, theme.PAD))
        # killer/survivor split: only perks/items/powers carry a role, so add-ons + offerings (null
        # role) stay visible under either pick (see Library.filter). populated by the role scrape.
        self.role = ctk.CTkOptionMenu(
            bar, width=110, values=["all", "killer", "survivor"],
            command=lambda v: self._apply_filter())
        self.role.set("all")
        self.role.pack(side="left")
        # reveal glyphs you can't buy in a current bloodweb (powers, retired offerings, event skins).
        # hidden by default; the choice persists via cfg["hide_unavailable"] on the next config save.
        hide = bool((self.app.app_state.config or {}).get("hide_unavailable", True))
        self.show_unavail = ctk.BooleanVar(value=not hide)
        ctk.CTkCheckBox(bar, text="event / n/a", font=theme.FONT_SMALL, width=20,
                        variable=self.show_unavail, command=self._on_show_unavail).pack(
            side="left", padx=(theme.PAD, 0))

        self.list = WindowedList(left, make_card=self._make_card, row_h=theme.ROW_H, buffer=6)
        self.list.grid(row=1, column=0, sticky="nsew", padx=theme.PAD, pady=(0, theme.PAD))

    # right pane: profile bar + save/revert + tier stack + template/category builders
    def _build_right(self):
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(theme.PAD // 2, theme.PAD), pady=theme.PAD)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_profile_bar(right)

        top = ctk.CTkFrame(right, fg_color="transparent")
        top.grid(row=1, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))
        ctk.CTkLabel(top, text="Priorities", font=theme.FONT_TITLE).pack(side="left")
        self.revert_btn = ctk.CTkButton(top, text="↺ revert", width=80, command=self._revert)
        self.revert_btn.pack(side="right", padx=(theme.PAD, 0))
        self.save_btn = ctk.CTkButton(top, text="Save", width=80, command=self._save)
        self.save_btn.pack(side="right")

        self.tiers = TierList(right, self.library, on_change=lambda: self._set_dirty(True))
        self.tiers.grid(row=2, column=0, sticky="nsew", padx=theme.PAD, pady=(0, theme.PAD))

        tools = ctk.CTkFrame(right, fg_color="transparent")
        tools.grid(row=3, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))
        # template tiers: pick a catch-all preset, drop it in as a new tier.
        ctk.CTkLabel(tools, text="template:", font=theme.FONT_SMALL).pack(side="left")
        self.template_menu = ctk.CTkOptionMenu(tools, width=180, values=TEMPLATE_LABELS)
        self.template_menu.set(TEMPLATE_LABELS[0])
        self.template_menu.pack(side="left", padx=theme.PAD)
        ctk.CTkButton(tools, text="+ add as tier", command=self._add_template_tier).pack(side="left")

        self.builder = RuleBuilder(right, on_add=self._add_category_rule)
        self.builder.grid(row=4, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))

    def _build_profile_bar(self, right):
        bar = ctk.CTkFrame(right)
        bar.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(bar, text="Profile", font=theme.FONT_SMALL).pack(side="left", padx=(theme.PAD, 0))
        self.profile_menu = ctk.CTkOptionMenu(
            bar, width=170, values=list(self.profiles), command=self._on_profile_select)
        self.profile_menu.set(self.active)
        self.profile_menu.pack(side="left", padx=theme.PAD)
        ctk.CTkButton(bar, text="+ new", width=64, command=self._new_profile).pack(side="left")
        ctk.CTkButton(bar, text="rename", width=64, command=self._rename_profile).pack(
            side="left", padx=theme.PAD)
        ctk.CTkButton(bar, text="delete", width=64, fg_color="#a83232",
                      command=self._delete_profile).pack(side="left")

    # library wiring
    def _make_card(self, master, height):
        return ItemCard(master, self.library, height=height, on_activate=self._on_card)

    def _on_card(self, row, button):
        if button != 1:
            return
        rule = {"type": "item", "name": row.get("name")}
        if row.get("rarity"):
            rule["rarity"] = row["rarity"]   # default to the card's rarity, toggleable on the chip
        self.tiers.add_rule(rule)

    def _add_category_rule(self, rule):
        self.tiers.add_rule(rule)

    def _add_template_tier(self):
        rules = TIER_TEMPLATES.get(self.template_menu.get())
        if rules:
            self.tiers.add_template_tier(rules)

    # profiles
    def _stash_current(self):
        """write the tier editor's current state back into the active profile (preserving empties)."""
        self.profiles[self.active] = self.tiers.get_tiers()

    def _on_profile_select(self, name):
        if name == self.active:
            return
        self._stash_current()
        self.active = name
        self.tiers.set_tiers(self.profiles.get(name, []))
        self._set_dirty(True)

    def _refresh_profile_menu(self):
        self.profile_menu.configure(values=list(self.profiles))
        self.profile_menu.set(self.active)

    def _new_profile(self):
        name = (ctk.CTkInputDialog(text="New profile name:", title="New profile").get_input() or "").strip()
        if not name:
            return
        if name in self.profiles:
            messagebox.showerror("profile exists", f"a profile named {name!r} already exists.")
            return
        self._stash_current()
        self.profiles[name] = []
        self.active = name
        self._refresh_profile_menu()
        self.tiers.set_tiers([])
        self._set_dirty(True)

    def _rename_profile(self):
        new = (ctk.CTkInputDialog(
            text=f"Rename {self.active!r} to:", title="Rename profile").get_input() or "").strip()
        if not new or new == self.active:
            return
        if new in self.profiles:
            messagebox.showerror("profile exists", f"a profile named {new!r} already exists.")
            return
        self._stash_current()
        # rebuild the dict swapping the one key, so the menu order is preserved.
        self.profiles = {new if k == self.active else k: v for k, v in self.profiles.items()}
        self.active = new
        self._refresh_profile_menu()
        self._set_dirty(True)

    def _delete_profile(self):
        if len(self.profiles) <= 1:
            messagebox.showinfo("can't delete", "at least one profile is required.")
            return
        if not messagebox.askyesno("delete profile", f"delete profile {self.active!r}?"):
            return
        del self.profiles[self.active]
        self.active = next(iter(self.profiles))
        self._refresh_profile_menu()
        self.tiers.set_tiers(self.profiles[self.active])
        self._set_dirty(True)

    def _on_show_unavail(self):
        """remember the reveal choice in memory (persists on the next config save) and refilter."""
        if self.app.app_state.config is not None:
            self.app.app_state.config["hide_unavailable"] = not self.show_unavail.get()
        self._apply_filter()

    def _apply_filter(self):
        rows = self.library.filter(self.search.get(), self.category.get(), self.rarity.get(),
                                   self.role.get(), show_unavailable=self.show_unavail.get())
        self.list.set_model(rows)

    def refresh_after_scrape(self):
        """re-show the library + tier chips after the icon library was (re)scraped in place."""
        self._apply_filter()
        self.tiers.refresh()

    # save / revert
    def _set_dirty(self, dirty):
        self.dirty = dirty
        self.save_btn.configure(text="Save *" if dirty else "Save")

    def _save(self):
        self._stash_current()
        cfg = dict(self.app.app_state.config or {})
        # tidy each profile (drop tiers with no rules) for the persisted file; mirror the active one
        # into top-level `priorities` so the engine/cli keep reading the right list. tiers stay in
        # canonical shape here; spender.save_config serializes them to the compact on-disk form.
        cfg["profiles"] = {name: [spender.copy_tier(t) for t in tiers if spender.tier_rules(t)]
                           for name, tiers in self.profiles.items()}
        cfg["active_profile"] = self.active
        cfg["priorities"] = cfg["profiles"].get(self.active, [])
        try:
            config_io.save(cfg)
        except ValueError as e:
            messagebox.showerror("invalid priorities", str(e))
            return
        self.app.app_state.config = cfg
        self.profiles = {name: spender.copy_tiers(tiers)
                         for name, tiers in cfg["profiles"].items()}
        self._set_dirty(False)

    def _revert(self):
        self.app.app_state.load_config()
        if self.app.app_state.config_error:
            messagebox.showerror("config error", self.app.app_state.config_error)
        cfg = self.app.app_state.config or {}
        self.profiles = {n: spender.copy_tiers(tiers)
                         for n, tiers in (cfg.get("profiles") or {}).items()}
        if not self.profiles:
            self.profiles = {config_io.DEFAULT_PROFILE: spender.copy_tiers(cfg.get("priorities", []))}
        self.active = cfg.get("active_profile") or next(iter(self.profiles))
        self._refresh_profile_menu()
        self.tiers.set_tiers(self.profiles[self.active])
        self._set_dirty(False)
