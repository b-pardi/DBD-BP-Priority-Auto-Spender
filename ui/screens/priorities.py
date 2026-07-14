"""priority-selection screen: library on the left, the priority tier stack on the right.

click a library card to add it as an item rule to the selected tier (its rarity defaults to the
card's, toggleable on the placed chip). the rule builder adds category rules. Save validates and
writes the whole config through the shared serializer; the revert button reloads the last saved file.
the screen edits the tier list's own copy and only writes app.app_state.config on Save — except a
plain profile switch with nothing unsaved, which quietly persists just the new active profile
(otherwise a restart snaps back, and marking a mere switch dirty was read as data loss).

the two panes sit in a tk.PanedWindow so the divider is draggable; profiles are grouped
survivor/killer in the picker via a per-profile role tag (config `profile_roles`).
"""

import tkinter as tk
import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import detect, spender

from .. import config_io, scrape_runner, theme
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
        # role tags ('survivor'/'killer', absent = unsorted) grouping the profile picker
        self.roles = dict(cfg.get("profile_roles") or {})

        # the two panes hang in a paned window so the divider between them is draggable (it opens
        # at roughly the old 35/65 split); tk's own PanedWindow since ctk has no paned container.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.paned = tk.PanedWindow(
            self, orient="horizontal", sashwidth=8, bd=0, bg=theme.BORDER,
            sashcursor="sb_h_double_arrow")
        self.paned.grid(row=0, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD)

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
        cfg = self.app.app_state.config or {}
        left = ctk.CTkFrame(self.paned)
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # search row, filters on their own row below it: the pane is user-resizable now, so the
        # controls can't count on one row's worth of width.
        srow = ctk.CTkFrame(left, fg_color="transparent")
        srow.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=(theme.PAD, 4))
        self.search = ctk.CTkEntry(srow, placeholder_text="search items...")
        self.search.pack(side="left", fill="x", expand=True)
        self.search.bind("<KeyRelease>", lambda e: self._apply_filter())
        # a ✕ overlaid on the entry's right edge; _apply_filter shows it only while there's text.
        self.clear_btn = ctk.CTkButton(
            self.search, text="✕", width=24, height=20, fg_color="transparent",
            hover_color=theme.BG_HOVER, command=self._clear_search)

        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))
        self.category = ctk.CTkOptionMenu(
            bar, width=100, values=["all"] + sorted(spender.VALID_CATEGORIES),
            command=lambda v: self._apply_filter())
        self.category.set("all")
        self.category.pack(side="left", padx=(0, theme.PAD))
        self.rarity = ctk.CTkOptionMenu(
            bar, width=100, values=["all"] + list(detect.RARITIES) + ["none"],
            command=lambda v: self._apply_filter())
        self.rarity.set("all")
        self.rarity.pack(side="left", padx=(0, theme.PAD))
        # killer/survivor split, strict (see Library.filter), plus one entry per killer: a killer
        # pick shows their own add-ons and the shared killer perks — their bloodweb, nothing else.
        self.role = ctk.CTkOptionMenu(
            bar, width=150, values=self._role_values(),
            command=lambda v: self._apply_filter())
        self.role.set("all")
        self.role.pack(side="left", padx=(0, theme.PAD))
        # reveal toggles for glyphs that can't turn up in a current bloodweb, split so the common
        # case (event skins, buyable while their event runs) doesn't drag in the never-buyable
        # bucket (powers, retired offerings). both persist on the next config save.
        self.show_event = ctk.BooleanVar(value=bool(cfg.get("show_event", True)))
        ctk.CTkCheckBox(bar, text="event", font=theme.FONT_SMALL, width=20,
                        variable=self.show_event, command=self._on_reveal).pack(
            side="left", padx=(theme.PAD, 0))
        self.show_na = ctk.BooleanVar(value=bool(cfg.get("show_na", False)))
        ctk.CTkCheckBox(bar, text="n/a", font=theme.FONT_SMALL, width=20,
                        variable=self.show_na, command=self._on_reveal).pack(
            side="left", padx=(theme.PAD, 0))

        self.list = WindowedList(left, make_card=self._make_card, row_h=theme.ROW_H, buffer=6)
        self.list.grid(row=2, column=0, sticky="nsew", padx=theme.PAD, pady=(0, theme.PAD))

        # ~35/65 opening split (the sash is draggable from there); stretch shares later resizes.
        self.paned.add(left, width=540, minsize=340, stretch="always")

    def _role_values(self):
        return ["all", "survivor", "killer (all)"] + self.library.killer_names()

    # right pane: profile bar + save/revert + tier stack + template/category builders
    def _build_right(self):
        right = ctk.CTkFrame(self.paned)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.paned.add(right, minsize=460, stretch="always")

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
        # INSET (theme.INSET, and the same on every control below): a panel gets no height of its own,
        # it only gets what its children ask for. with no pady the bar ends up exactly as tall as the
        # controls in it, so their plates run edge to edge and spill a pixel or two past its rounded
        # corners, which reads as widgets sagging out the bottom of the bar. the padding IS the bar.
        bar = ctk.CTkFrame(right)
        bar.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(bar, text="Profile", font=theme.FONT_SMALL).pack(
            side="left", padx=(theme.PAD, 0), pady=theme.INSET)
        # a button + grouped tk menu instead of a flat CTkOptionMenu: the menu carries survivor /
        # killer section headers (from the role tags), which an option menu can't render.
        self.profile_btn = ctk.CTkButton(bar, width=190, anchor="w", text=f"{self.active}  ▾",
                                         command=self._open_profile_menu)
        self.profile_btn.pack(side="left", padx=theme.PAD, pady=theme.INSET)
        ctk.CTkButton(bar, text="+ new", width=64, command=self._new_profile).pack(
            side="left", pady=theme.INSET)
        ctk.CTkButton(bar, text="rename", width=64, command=self._rename_profile).pack(
            side="left", padx=theme.PAD, pady=theme.INSET)
        ctk.CTkButton(bar, text="delete", width=64, fg_color=theme.DANGER,
                      hover_color=theme.DANGER_HOVER,
                      command=self._delete_profile).pack(side="left", pady=theme.INSET)
        # the active profile's role tag, i.e. which group it files under in the picker.
        ctk.CTkLabel(bar, text="side", font=theme.FONT_SMALL).pack(
            side="left", padx=(2 * theme.PAD, 0), pady=theme.INSET)
        self.role_menu = ctk.CTkOptionMenu(bar, width=110,
                                           values=["unsorted", "survivor", "killer"],
                                           command=self._on_role_tag)
        self.role_menu.set(self.roles.get(self.active) or "unsorted")
        self.role_menu.pack(side="left", padx=theme.PAD, pady=theme.INSET)

    def _profile_groups(self):
        """profiles bucketed by role tag, each bucket in the dict's own order."""
        groups = {"survivor": [], "killer": [], "": []}
        for name in self.profiles:
            role = self.roles.get(name)
            groups[role if role in groups else ""].append(name)
        return groups

    def _open_profile_menu(self):
        """pop the grouped profile picker under its button. rebuilt per open (it's tiny), themed by
        hand since it's a raw tk.Menu rather than ctk's flat DropdownMenu."""
        menu = tk.Menu(self, tearoff=0, bd=0, relief="flat", font=theme.FONT_BODY,
                       bg=theme.BG_PANEL, fg=theme.BONE,
                       activebackground=theme.BG_HOVER, activeforeground=theme.BONE,
                       disabledforeground=theme.ASH)
        groups = self._profile_groups()
        tagged = bool(groups["survivor"] or groups["killer"])   # untagged-only lists skip headers
        first = True
        for key, header in (("survivor", "survivor"), ("killer", "killer"), ("", "unsorted")):
            names = groups[key]
            if not names:
                continue
            if not first:
                menu.add_separator()
            first = False
            if tagged:
                menu.add_command(label=f"—— {header} ——", state="disabled")
            for n in names:
                mark = "✓ " if n == self.active else "   "
                menu.add_command(label=(("  " if tagged else "") + mark + n),
                                 command=lambda v=n: self._on_profile_select(v))
        menu.tk_popup(self.profile_btn.winfo_rootx(),
                      self.profile_btn.winfo_rooty() + self.profile_btn.winfo_height() + 2)

    # library wiring
    def _make_card(self, master, height):
        # on_drop/on_drag_hover read self.tiers lazily (cards are built on first relayout, after the
        # right pane exists), so a dragged card can land on a specific tier at a specific position.
        return ItemCard(master, self.library, height=height, on_activate=self._on_card,
                        on_drop=self._on_card_drop,
                        on_drag_hover=lambda x, y: self.tiers.drag_highlight(x, y))

    @staticmethod
    def _row_to_rule(row):
        rule = {"type": "item", "name": row.get("name")}
        if row.get("rarity"):
            rule["rarity"] = row["rarity"]   # default to the card's rarity, toggleable on the chip
        return rule

    def _on_card(self, row, button):
        if button != 1:
            return
        self.tiers.add_rule(self._row_to_rule(row))   # plain click -> add to the selected tier

    def _on_card_drop(self, row, x_root, y_root):
        # dragged from the library onto a tier: add at the drop position on whichever tier it landed.
        self.tiers.drop_item_rule(self._row_to_rule(row), x_root, y_root)

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
        was_dirty = self.dirty
        self._stash_current()
        self.active = name
        self.tiers.set_tiers(self.profiles.get(name, []))
        self._refresh_profile_menu()
        # a plain switch isn't an edit, so it must not raise the dirty star — but the new active
        # profile has to reach disk or a restart snaps back to the old one. so: with nothing
        # unsaved, quietly write through; with edits pending, stay dirty and let Save carry both.
        if was_dirty:
            self._set_dirty(True)
        else:
            self._save()

    def _on_role_tag(self, value):
        """re-tag the active profile survivor/killer (or back to unsorted), regrouping the picker."""
        role = None if value == "unsorted" else value
        if self.roles.get(self.active) == role:
            return
        if role is None:
            self.roles.pop(self.active, None)
        else:
            self.roles[self.active] = role
        self._set_dirty(True)

    def _refresh_profile_menu(self):
        self.profile_btn.configure(text=f"{self.active}  ▾")
        self.role_menu.set(self.roles.get(self.active) or "unsorted")

    def _new_profile(self):
        # styled between construction and get_input(): the ctor returns before it blocks on the dialog
        dlg = ctk.CTkInputDialog(text="New profile name:", title="New profile")
        scrape_runner.style_child_window(dlg)
        name = (dlg.get_input() or "").strip()
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
        dlg = ctk.CTkInputDialog(text=f"Rename {self.active!r} to:", title="Rename profile")
        scrape_runner.style_child_window(dlg)
        new = (dlg.get_input() or "").strip()
        if not new or new == self.active:
            return
        if new in self.profiles:
            messagebox.showerror("profile exists", f"a profile named {new!r} already exists.")
            return
        self._stash_current()
        # rebuild the dict swapping the one key, so the menu order is preserved.
        self.profiles = {new if k == self.active else k: v for k, v in self.profiles.items()}
        if self.active in self.roles:   # the role tag follows the rename
            self.roles[new] = self.roles.pop(self.active)
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
        self.roles.pop(self.active, None)
        self.active = next(iter(self.profiles))
        self._refresh_profile_menu()
        self.tiers.set_tiers(self.profiles[self.active])
        self._set_dirty(True)

    def _on_reveal(self):
        """remember the reveal choices in memory (they persist on the next config save) and refilter."""
        if self.app.app_state.config is not None:
            self.app.app_state.config["show_event"] = bool(self.show_event.get())
            self.app.app_state.config["show_na"] = bool(self.show_na.get())
        self._apply_filter()

    def _clear_search(self):
        self.search.delete(0, "end")
        self.search.focus_set()
        self._apply_filter()

    def _apply_filter(self):
        role = self.role.get()
        if role == "killer (all)":
            role = "killer"   # the menu label; the library filter knows the bare side
        rows = self.library.filter(self.search.get(), self.category.get(), self.rarity.get(),
                                   role, show_event=self.show_event.get(),
                                   show_na=self.show_na.get())
        self.list.set_model(rows)
        # the clear ✕ only earns its overlay while there's something to clear
        if self.search.get():
            self.clear_btn.place(relx=1.0, rely=0.5, x=-4, anchor="e")
        else:
            self.clear_btn.place_forget()

    def refresh_after_scrape(self):
        """re-show the library + tier chips after the icon library was (re)scraped in place."""
        self.role.configure(values=self._role_values())   # a new chapter can add a killer
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
        # role tags, pruned to live profiles and real sides (unsorted = absent)
        cfg["profile_roles"] = {n: r for n, r in self.roles.items()
                                if n in cfg["profiles"] and r in ("survivor", "killer")}
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
        self.roles = dict(cfg.get("profile_roles") or {})
        self._refresh_profile_menu()
        self.tiers.set_tiers(self.profiles[self.active])
        self._set_dirty(False)
