"""the right-pane priority tier stack.

owns the in-memory tier model (a list of tiers, each {"rules": [rule, ...], "ordered": bool}) and
renders it top (highest) to bottom. one tier is the selected add-target so library clicks and the
rule builder land there. tiers can be removed and reordered (up/down, and jump to top/bottom);
individual rules can be nudged into the adjacent tier from their chip.

each tier has a Random/Ordered toggle. Random (the default) means the engine picks at random across
every match in the tier, as it always has. Ordered means the rules are ranked top to bottom and the
engine prefers the highest-ranked match (see spender.choose_next); in that mode each chip shows its
rank and gains within-tier up/down arrows.

small enough that a plain CTkScrollableFrame rebuilt on each edit is fine (only the library needs
virtualization). edits go through this widget's methods; it copies on set_tiers so nothing is mutated
until the screen saves.
"""

import customtkinter as ctk

from src import spender

from ..theme import FONT_BODY, FONT_SMALL, NAV_ACTIVE_COLOR, PAD
from .rule_chip import RuleChip


class TierList(ctk.CTkScrollableFrame):
    def __init__(self, master, library, on_change=None):
        super().__init__(master, label_text="Priority tiers")
        self.library = library
        self.on_change = on_change   # called after any edit so the screen can mark itself dirty
        self.tiers = [self._new_tier()]
        self.selected = 0
        self._render()

    @staticmethod
    def _new_tier():
        return {"rules": [], "ordered": False}

    # model edits
    def set_tiers(self, tiers):
        """load tiers from a config (copied so edits don't touch the source until save)."""
        self.tiers = spender.copy_tiers(tiers) if tiers else [self._new_tier()]
        self.selected = 0
        self._render()

    def cleaned_tiers(self):
        """tiers with empties dropped, ready to persist."""
        return [spender.copy_tier(t) for t in self.tiers if spender.tier_rules(t)]

    def get_tiers(self):
        """a copy of the current tiers including empty ones (used to stash a profile on switch)."""
        return spender.copy_tiers(self.tiers)

    def add_tier(self):
        self.tiers.append(self._new_tier())
        self.selected = len(self.tiers) - 1
        self._changed()

    def remove_tier(self, i):
        if 0 <= i < len(self.tiers):
            del self.tiers[i]
            if not self.tiers:
                self.tiers = [self._new_tier()]
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
        self.tiers.append({"rules": [dict(r) for r in rules], "ordered": False})
        self.selected = len(self.tiers) - 1
        self._changed()

    def toggle_ordered(self, ti):
        """flip a tier between Random and Ordered within-tier selection."""
        if 0 <= ti < len(self.tiers):
            self.tiers[ti]["ordered"] = not self.tiers[ti].get("ordered", False)
            self._changed()

    def _move_rule(self, ti, rule, delta):
        """move one rule into the adjacent tier (delta -1 up / +1 down), de-duped; no-op at the ends."""
        j = ti + delta
        if not (0 <= ti < len(self.tiers)) or not (0 <= j < len(self.tiers)):
            return
        src, dst = self.tiers[ti]["rules"], self.tiers[j]["rules"]
        if rule in src:
            src.remove(rule)
        if rule not in dst:
            dst.append(dict(rule))
        self._changed()

    def _reorder_rule(self, ti, rule, delta):
        """move one rule up/down within its own tier (ordered tiers), so its within-tier rank
        changes; no-op at the ends. clamped to the tier so it can't spill into a neighbour."""
        if not (0 <= ti < len(self.tiers)):
            return
        rules = self.tiers[ti]["rules"]
        if rule not in rules:
            return
        i = rules.index(rule)
        j = max(0, min(i + delta, len(rules) - 1))
        if i != j:
            rules.insert(j, rules.pop(i))
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
        rules = self.tiers[self.selected]["rules"]
        if rule not in rules:
            rules.append(dict(rule))
            self._changed()

    def _remove_from(self, rules, rule):
        if rule in rules:
            rules.remove(rule)
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
            text="Higher tiers win. A tier is Random by default; switch it to Ordered to rank the "
                 "items inside it (top = picked first).",
        ).pack(anchor="w", padx=PAD, pady=(0, PAD))
        for ti, tier in enumerate(self.tiers):
            self._render_tier(ti, tier)
        ctk.CTkButton(self, text="+ add tier", command=self.add_tier).pack(
            fill="x", padx=PAD, pady=PAD
        )

    def _render_tier(self, ti, tier):
        sel = ti == self.selected
        ordered = bool(tier.get("ordered", False))
        rules = tier["rules"]
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

        # within-tier selection mode. the segmented button reads as the current mode; Ordered turns
        # on the per-chip rank + reorder arrows below (only meaningful with 2+ rules).
        modebar = ctk.CTkFrame(box, fg_color="transparent")
        modebar.pack(fill="x", padx=PAD, pady=(4, 0))
        ctk.CTkLabel(modebar, text="within tier:", font=FONT_SMALL).pack(side="left")
        seg = ctk.CTkSegmentedButton(
            modebar, values=["Random", "Ordered"], font=FONT_SMALL,
            command=lambda v, i=ti: self._on_mode(i, v))
        seg.set("Ordered" if ordered else "Random")
        seg.pack(side="left", padx=PAD)
        if ordered and len(rules) > 1:
            ctk.CTkLabel(modebar, text="top = first pick", font=FONT_SMALL,
                         text_color="gray60").pack(side="left")

        if not rules:
            ctk.CTkLabel(box, text="(empty — click a library item or add a category rule)",
                         font=FONT_SMALL).pack(anchor="w", padx=PAD, pady=PAD)
        for ri, rule in enumerate(rules):
            RuleChip(
                box, rule, self.library,
                on_remove=lambda r, rs=rules: self._remove_from(rs, r),
                on_toggle_rarity=self._toggle_rarity,
                on_move=lambda r, d, i=ti: self._move_rule(i, r, d),
                rank=(ri + 1 if ordered else None),
                on_reorder=(lambda r, d, i=ti: self._reorder_rule(i, r, d)) if ordered else None,
            ).pack(fill="x", padx=PAD, pady=2)

    def _on_mode(self, ti, value):
        # the segmented button fires on every set (including our own seg.set on rebuild); only act
        # when the chosen mode actually differs from the stored one so a re-render can't loop.
        want = (value == "Ordered")
        if 0 <= ti < len(self.tiers) and bool(self.tiers[ti].get("ordered", False)) != want:
            self.toggle_ordered(ti)
