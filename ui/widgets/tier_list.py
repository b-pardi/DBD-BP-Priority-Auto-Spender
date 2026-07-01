"""the right-pane priority tier stack.

owns the in-memory tier model (a list of tiers, each a list of rule dicts) and renders it top
(highest) to bottom. one tier is the selected add-target so library clicks and the rule builder land
there. tiers can be removed and reordered (up/down, and jump to top/bottom); individual rules can be
nudged into the adjacent tier from their chip. within a tier there is no order, since the engine
picks randomly among matches, so the ui says so once and never implies an ordering.

small enough that a plain CTkScrollableFrame rebuilt on each edit is fine (only the library needs
virtualization). edits go through this widget's methods; it deep-copies on set_tiers so nothing is
mutated until the screen saves.
"""

import customtkinter as ctk

from ..theme import FONT_BODY, FONT_SMALL, NAV_ACTIVE_COLOR, PAD
from .rule_chip import RuleChip


class TierList(ctk.CTkScrollableFrame):
    def __init__(self, master, library, on_change=None):
        super().__init__(master, label_text="Priority tiers")
        self.library = library
        self.on_change = on_change   # called after any edit so the screen can mark itself dirty
        self.tiers = [[]]
        self.selected = 0
        self._render()

    # model edits
    def set_tiers(self, tiers):
        """load tiers from a config (deep-copied so edits don't touch the source until save)."""
        self.tiers = [list(t) for t in tiers] if tiers else [[]]
        self.selected = 0
        self._render()

    def cleaned_tiers(self):
        """tiers with empties dropped, ready to persist."""
        return [list(t) for t in self.tiers if t]

    def get_tiers(self):
        """a deep copy of the current tiers including empty ones (used to stash a profile on switch)."""
        return [list(t) for t in self.tiers]

    def add_tier(self):
        self.tiers.append([])
        self.selected = len(self.tiers) - 1
        self._changed()

    def remove_tier(self, i):
        if 0 <= i < len(self.tiers):
            del self.tiers[i]
            if not self.tiers:
                self.tiers = [[]]
            self.selected = max(0, min(self.selected, len(self.tiers) - 1))
            self._changed()

    def move_tier(self, i, delta):
        j = i + delta
        if 0 <= i < len(self.tiers) and 0 <= j < len(self.tiers):
            self.tiers[i], self.tiers[j] = self.tiers[j], self.tiers[i]
            if self.selected == i:
                self.selected = j
            elif self.selected == j:
                self.selected = i
            self._changed()

    def move_tier_to(self, i, j):
        """pull tier i out and reinsert at index j (used by the to-top / to-bottom buttons)."""
        if not (0 <= i < len(self.tiers)) or i == j:
            return
        j = max(0, min(j, len(self.tiers) - 1))
        tier = self.tiers.pop(i)
        self.tiers.insert(j, tier)
        if self.selected == i:
            self.selected = j
        elif i < self.selected <= j:
            self.selected -= 1
        elif j <= self.selected < i:
            self.selected += 1
        self._changed()

    def add_template_tier(self, rules):
        """append a new tier pre-filled with template rules (e.g. 'any perk') and select it."""
        self.tiers.append([dict(r) for r in rules])
        self.selected = len(self.tiers) - 1
        self._changed()

    def _move_rule(self, ti, rule, delta):
        """move one rule into the adjacent tier (delta -1 up / +1 down), de-duped; no-op at the ends."""
        j = ti + delta
        if not (0 <= ti < len(self.tiers)) or not (0 <= j < len(self.tiers)):
            return
        if rule in self.tiers[ti]:
            self.tiers[ti].remove(rule)
        if rule not in self.tiers[j]:
            self.tiers[j].append(dict(rule))
        self._changed()

    def select(self, i):
        """set the add-target tier (not a config edit, so no on_change)."""
        self.selected = i
        self._render()

    def refresh(self):
        """re-render in place (e.g. after a scrape fills in previously-missing chip thumbnails)."""
        self._render()

    def add_rule(self, rule):
        """add a rule to the selected tier, de-duped within that tier."""
        tier = self.tiers[self.selected]
        if rule not in tier:
            tier.append(dict(rule))
            self._changed()

    def _remove_from(self, tier, rule):
        if rule in tier:
            tier.remove(rule)
        self._changed()

    def _toggle_rarity(self, rule):
        """flip an item rule between its pinned rarity and 'any rarity of this item'."""
        if rule.get("rarity"):
            rule.pop("rarity", None)
        else:
            rar = self.library.lookup_rarity(rule.get("name", ""))
            if rar:
                rule["rarity"] = rar
        self._changed()

    # rendering
    def _changed(self):
        self._render()
        if self.on_change:
            self.on_change()

    def _render(self):
        for w in self.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self, font=FONT_SMALL, justify="left", wraplength=380,
            text="Higher tiers win. Within a tier the node is chosen at random (no order).",
        ).pack(anchor="w", padx=PAD, pady=(0, PAD))
        for ti, tier in enumerate(self.tiers):
            self._render_tier(ti, tier)
        ctk.CTkButton(self, text="+ add tier", command=self.add_tier).pack(
            fill="x", padx=PAD, pady=PAD
        )

    def _render_tier(self, ti, tier):
        sel = ti == self.selected
        box = ctk.CTkFrame(self, border_width=2,
                           border_color=(NAV_ACTIVE_COLOR if sel else "gray30"))
        box.pack(fill="x", padx=PAD, pady=(0, PAD))

        header = ctk.CTkFrame(box, fg_color="transparent")
        header.pack(fill="x", padx=PAD, pady=(PAD, 0))
        title = f"Tier {ti + 1}" + (" (highest)" if ti == 0 else "")
        ctk.CTkButton(
            header, text=title, anchor="w", font=FONT_BODY,
            fg_color=(NAV_ACTIVE_COLOR if sel else "transparent"),
            command=lambda i=ti: self.select(i),
        ).pack(side="left", fill="x", expand=True)
        last = len(self.tiers) - 1
        ctk.CTkButton(header, text="⤒", width=24,
                      command=lambda i=ti: self.move_tier_to(i, 0)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="▲", width=24,
                      command=lambda i=ti: self.move_tier(i, -1)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="▼", width=24,
                      command=lambda i=ti: self.move_tier(i, 1)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="⤓", width=24,
                      command=lambda i=ti: self.move_tier_to(i, last)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="x", width=24,
                      command=lambda i=ti: self.remove_tier(i)).pack(side="left", padx=(2, 0))

        if not tier:
            ctk.CTkLabel(box, text="(empty — click a library item or add a category rule)",
                         font=FONT_SMALL).pack(anchor="w", padx=PAD, pady=PAD)
        for rule in tier:
            RuleChip(
                box, rule, self.library,
                on_remove=lambda r, t=tier: self._remove_from(t, r),
                on_toggle_rarity=self._toggle_rarity,
                on_move=lambda r, d, i=ti: self._move_rule(i, r, d),
            ).pack(fill="x", padx=PAD, pady=2)
