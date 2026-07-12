"""the right-pane priority tier stack.

owns the in-memory tier model (a list of tiers, each {"rules": [rule, ...], "ordered": bool}) and
renders it top (highest) to bottom. one tier is the selected add-target so library clicks and the
rule builder land there. tiers can be removed and reordered (up/down, and jump to top/bottom);
individual rules can be nudged into the adjacent tier from their chip.

each tier has a Random/Ordered toggle. Random (the default) means the engine picks at random across
every match in the tier, as it always has. Ordered means the rules are ranked top to bottom and the
engine prefers the highest-ranked match (see spender.choose_next); in that mode each chip shows its
rank and gains within-tier up/down arrows.

rendering is a reconcile (_sync), not a rebuild. it used to tear down every widget and rebuild the
whole stack after any edit, which cost ~3s on a 22-rule profile -- every ctk widget is a canvas it
draws itself onto, so a rule chip is ~10 of them -- and it ran on *every* interaction, including just
clicking a tier header to select it. now a tier box is built only when its tier is new and a chip
only when its rule is new; order, ranks, titles and the selection tint are a repack plus a few
configure calls. the enabler is that callbacks capture the tier *dict*, not its index, so reordering
tiers can't strand them (see _make_chip).

edits go through this widget's methods; it copies on set_tiers so nothing is mutated until the
screen saves.
"""

import customtkinter as ctk

from src import spender

from ..theme import ACCENT, ASH, BORDER, FONT_BODY, FONT_SMALL, FOG, PAD
from .rule_chip import RuleChip

DROP_COLOR = FOG  # tier border tint while a drag hovers it: bright + neutral, so it can't be
                  # confused with the ember tint on the tier that happens to be selected


class _TierBox:
    """the widgets for one tier, kept across syncs. holds the tier dict itself (never its index), so
    its buttons keep pointing at the right tier when the stack is reordered."""

    def __init__(self, tier, frame, header_btn, seg, hint, empty):
        self.tier = tier
        self.frame = frame
        self.header_btn = header_btn
        self.seg = seg          # Random/Ordered segmented button
        self.hint = hint        # "top = first pick", shown only for an ordered tier with 2+ rules
        self.empty = empty      # the "(empty)" placeholder
        self.chips = []         # RuleChip, in pack order
        # what _sync last applied, so an unchanged tier costs zero configure calls
        self.title = None
        self.sel = None
        self.ordered = None


class TierList(ctk.CTkScrollableFrame):
    def __init__(self, master, library, on_change=None):
        super().__init__(master, label_text="Priority tiers")
        self.library = library
        self.on_change = on_change   # called after any edit so the screen can mark itself dirty
        self.tiers = [self._new_tier()]
        self.selected = 0
        self._boxes = []             # _TierBox, aligned to self.tiers
        self._drag_target = None     # index tinted by the current drag, so we only retint on change

        # permanent chrome: built once, never torn down, so the tier boxes pack between them.
        self.blurb = ctk.CTkLabel(
            self, font=FONT_SMALL, justify="left", wraplength=380,
            text="Higher tiers win. A tier is Random by default; switch it to Ordered to rank the "
                 "items inside it (top = picked first).",
        )
        self.blurb.pack(anchor="w", padx=PAD, pady=(0, PAD))
        self.add_btn = ctk.CTkButton(self, text="+ add tier", command=self.add_tier)
        self.add_btn.pack(fill="x", padx=PAD, pady=PAD)
        self._sync()

    @staticmethod
    def _new_tier():
        return {"rules": [], "ordered": False}

    def _index_of(self, tier):
        """position of a tier in the model, by identity (two tiers can hold equal rules)."""
        for i, t in enumerate(self.tiers):
            if t is tier:
                return i
        return -1

    # model edits
    def set_tiers(self, tiers):
        """load tiers from a config (copied so edits don't touch the source until save)."""
        self.tiers = spender.copy_tiers(tiers) if tiers else [self._new_tier()]
        self.selected = 0
        self._sync()   # all-new tier dicts, so this rebuilds; profile switches are rare

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

    def remove_tier(self, tier):
        i = self._index_of(tier)
        if i < 0:
            return
        del self.tiers[i]
        if not self.tiers:
            self.tiers = [self._new_tier()]
        self.selected = max(0, min(self.selected, len(self.tiers) - 1))
        self._changed()

    def move_tier(self, tier, delta):
        i = self._index_of(tier)
        j = i + delta
        if i < 0 or not (0 <= j < len(self.tiers)):
            return
        self.tiers[i], self.tiers[j] = self.tiers[j], self.tiers[i]
        if self.selected == i:
            self.selected = j
        elif self.selected == j:
            self.selected = i
        self._changed()

    def move_tier_to(self, tier, j):
        """pull a tier out and reinsert at index j (used by the to-top / to-bottom buttons)."""
        i = self._index_of(tier)
        if i < 0 or i == j:
            return
        j = max(0, min(j, len(self.tiers) - 1))
        self.tiers.insert(j, self.tiers.pop(i))
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

    def toggle_ordered(self, tier):
        """flip a tier between Random and Ordered within-tier selection."""
        tier["ordered"] = not tier.get("ordered", False)
        self._changed()

    def add_rule(self, rule):
        """add a rule to the selected tier, de-duped within that tier."""
        rules = self.tiers[self.selected]["rules"]
        if rule in rules:
            return
        rules.append(dict(rule))
        self._changed()

    def select(self, i):
        """set the add-target tier by index (not a config edit, so no on_change)."""
        self.selected = max(0, min(i, len(self.tiers) - 1))
        self._sync()

    def refresh(self):
        """rebuild from scratch. a chip reads its thumbnail once at build time, so this is what
        picks up sprites that only exist after a scrape."""
        for box in self._boxes:
            box.frame.destroy()
        self._boxes = []
        self._sync()

    def _remove_rule(self, tier, rule):
        if rule in tier["rules"]:
            tier["rules"].remove(rule)
        self._changed()

    def _move_rule(self, tier, rule, delta):
        """move one rule into the adjacent tier (delta -1 up / +1 down), de-duped; no-op at the ends."""
        i = self._index_of(tier)
        j = i + delta
        if i < 0 or not (0 <= j < len(self.tiers)):
            return
        src, dst = tier["rules"], self.tiers[j]["rules"]
        if rule in src:
            src.remove(rule)
        if rule not in dst:
            dst.append(dict(rule))
        self._changed()

    def _reorder_rule(self, tier, rule, delta):
        """move one rule up/down within its own tier (ordered tiers), so its within-tier rank
        changes; no-op at the ends. clamped to the tier so it can't spill into a neighbour."""
        rules = tier["rules"]
        if rule not in rules:
            return
        i = rules.index(rule)
        j = max(0, min(i + delta, len(rules) - 1))
        if i != j:
            rules.insert(j, rules.pop(i))
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

    def _on_mode(self, tier, value):
        # the segmented button fires on every set (including our own seg.set in _sync); only act when
        # the chosen mode actually differs from the stored one so a sync can't loop.
        want = (value == "Ordered")
        if bool(tier.get("ordered", False)) != want:
            self.toggle_ordered(tier)

    # drag and drop (see widgets/dragdrop.py). geometry hit-testing maps a drop point to a tier +
    # insert index, then the model edit goes through the same rule lists the chips render from.
    def _hit_tier(self, x_root, y_root):
        """index of the tier box under the given screen point, or None (dropped outside)."""
        for i, box in enumerate(self._boxes):
            f = box.frame
            if not f.winfo_ismapped():
                continue
            bx, by = f.winfo_rootx(), f.winfo_rooty()
            if bx <= x_root <= bx + f.winfo_width() and by <= y_root <= by + f.winfo_height():
                return i
        return None

    def _insert_index(self, ti, y_root):
        """where in tier ti's rule list a drop at y_root lands (0..n), by existing chip midpoints."""
        for idx, chip in enumerate(self._boxes[ti].chips):
            if y_root < chip.winfo_rooty() + chip.winfo_height() / 2:
                return idx
        return len(self._boxes[ti].chips)

    def drag_highlight(self, x_root, y_root):
        """tint the tier under the cursor while a drag hovers it; x_root None clears back to normal.
        fires on every mouse move during a drag, so it only touches the two boxes that changed."""
        target = None if x_root is None else self._hit_tier(x_root, y_root)
        if target == self._drag_target:
            return
        self._drag_target = target
        for i, box in enumerate(self._boxes):
            tint = DROP_COLOR if i == target else (
                ACCENT if i == self.selected else BORDER)
            box.frame.configure(border_color=tint)
            box.sel = None   # we painted behind _sync's back; make it repaint the border next time

    def drop_item_rule(self, rule, x_root, y_root):
        """insert a library-dragged item rule into the tier it was dropped on, at the drop position;
        de-duped within that tier. no-op when dropped outside every tier."""
        ti = self._hit_tier(x_root, y_root)
        if ti is None:
            self.drag_highlight(None, None)
            return
        rules = self.tiers[ti]["rules"]
        self.selected = ti
        if rule not in rules:                          # already present: just move the selection
            rules.insert(self._insert_index(ti, y_root), dict(rule))
        self._changed()

    def _drop_chip(self, tier, rule, x_root, y_root):
        """move a dragged chip's rule to the drop tier + position (reorder within, or across tiers)."""
        dst_ti = self._hit_tier(x_root, y_root)
        src_ti = self._index_of(tier)
        if dst_ti is None or src_ti < 0:
            self.drag_highlight(None, None)
            return
        src = tier["rules"]
        if rule not in src:
            return
        insert = self._insert_index(dst_ti, y_root)
        old = src.index(rule)
        if src_ti == dst_ti and old < insert:
            insert -= 1   # popping the rule first shifts everything after it up by one
        moving = src.pop(old)
        dst = self.tiers[dst_ti]["rules"]
        if moving not in dst:   # a cross-tier drop onto an equal rule just consolidates (no dup)
            dst.insert(max(0, min(insert, len(dst))), moving)
        self.selected = dst_ti
        self._changed()

    # rendering
    def _changed(self):
        self._sync()
        if self.on_change:
            self.on_change()

    def _sync(self):
        """reconcile the widgets to the model, reusing every box and chip whose tier/rule survived."""
        self._drag_target = None
        boxes = [self._box_for(tier) for tier in self.tiers]
        for box in self._boxes:
            if box not in boxes:
                box.frame.destroy()
        if self._boxes != boxes:          # membership or order changed -> repack, add-tier last
            self.add_btn.pack_forget()
            for box in boxes:
                box.frame.pack_forget()
            for box in boxes:
                box.frame.pack(fill="x", padx=PAD, pady=(0, PAD))
            self.add_btn.pack(fill="x", padx=PAD, pady=PAD)
        self._boxes = boxes

        for i, box in enumerate(boxes):
            self._sync_box(i, box)

    def _sync_box(self, i, box):
        tier = box.tier
        ordered = bool(tier.get("ordered", False))
        sel = (i == self.selected)
        title = f"Tier {i + 1}" + (" (highest)" if i == 0 else "")
        if box.title != title:
            box.header_btn.configure(text=title)
            box.title = title
        if box.sel != sel:
            box.frame.configure(border_color=(ACCENT if sel else BORDER))
            box.header_btn.configure(fg_color=(ACCENT if sel else "transparent"))
            box.sel = sel
        if box.ordered != ordered:
            box.seg.set("Ordered" if ordered else "Random")   # no-ops in _on_mode, guarded there
            box.ordered = ordered

        rules = tier["rules"]
        want_empty = not rules
        if want_empty != bool(box.empty.winfo_manager()):
            if want_empty:
                box.empty.pack(anchor="w", padx=PAD, pady=PAD)
            else:
                box.empty.pack_forget()
        want_hint = ordered and len(rules) > 1   # ranking is only meaningful with something to rank
        if want_hint != bool(box.hint.winfo_manager()):
            if want_hint:
                box.hint.pack(side="left")
            else:
                box.hint.pack_forget()

        chips = []
        for ri, rule in enumerate(rules):
            chip = next((c for c in box.chips if c.rule is rule and c not in chips), None)
            if chip is None:   # rank at build time, so a new chip in an ordered tier isn't repacked
                chip = self._make_chip(box, tier, rule, (ri + 1) if ordered else None)
            chips.append(chip)
        for chip in box.chips:
            if chip not in chips:
                chip.destroy()
        if box.chips != chips:            # a rule was added/removed/reordered -> repack in model order
            for chip in chips:
                chip.pack_forget()
            for chip in chips:
                chip.pack(fill="x", padx=PAD, pady=2)
            box.chips = chips
        for ri, chip in enumerate(chips):
            chip.sync(rank=(ri + 1) if ordered else None)

    def _box_for(self, tier):
        """the _TierBox already bound to this tier dict, or a freshly built one."""
        for box in self._boxes:
            if box.tier is tier:
                return box
        return self._make_box(tier)

    def _make_box(self, tier):
        """build one tier's widgets. every callback captures the tier dict, so it survives a reorder;
        colors/titles/ranks are all left to _sync."""
        frame = ctk.CTkFrame(self, border_width=2)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=PAD, pady=(PAD, 0))
        header_btn = ctk.CTkButton(header, anchor="w", font=FONT_BODY,
                                   command=lambda t=tier: self.select(self._index_of(t)))
        header_btn.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(header, text="⤒", width=24,
                      command=lambda t=tier: self.move_tier_to(t, 0)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="▲", width=24,
                      command=lambda t=tier: self.move_tier(t, -1)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="▼", width=24,
                      command=lambda t=tier: self.move_tier(t, 1)).pack(side="left", padx=2)
        ctk.CTkButton(header, text="⤓", width=24,
                      command=lambda t=tier: self.move_tier_to(t, len(self.tiers) - 1)).pack(
            side="left", padx=2)
        ctk.CTkButton(header, text="x", width=24,
                      command=lambda t=tier: self.remove_tier(t)).pack(side="left", padx=(2, 0))

        # within-tier selection mode. the segmented button reads as the current mode; Ordered turns
        # on the per-chip rank + reorder arrows below (only meaningful with 2+ rules).
        modebar = ctk.CTkFrame(frame, fg_color="transparent")
        modebar.pack(fill="x", padx=PAD, pady=(4, 0))
        ctk.CTkLabel(modebar, text="within tier:", font=FONT_SMALL).pack(side="left")
        seg = ctk.CTkSegmentedButton(modebar, values=["Random", "Ordered"], font=FONT_SMALL,
                                     command=lambda v, t=tier: self._on_mode(t, v))
        seg.pack(side="left", padx=PAD)
        hint = ctk.CTkLabel(modebar, text="top = first pick", font=FONT_SMALL, text_color=ASH)

        empty = ctk.CTkLabel(frame, text="(empty — click a library item or add a category rule)",
                             font=FONT_SMALL)
        return _TierBox(tier, frame, header_btn, seg, hint, empty)

    def _make_chip(self, box, tier, rule, rank=None):
        """build a chip for one rule. like the tier buttons, its callbacks capture the tier dict, so
        moving the tier around the stack can't strand them on a stale index."""
        return RuleChip(
            box.frame, rule, self.library,
            on_remove=lambda r, t=tier: self._remove_rule(t, r),
            on_toggle_rarity=self._toggle_rarity,
            on_move=lambda r, d, t=tier: self._move_rule(t, r, d),
            rank=rank,
            on_reorder=lambda r, d, t=tier: self._reorder_rule(t, r, d),
            on_drag=lambda r, x, y, t=tier: self._drop_chip(t, r, x, y),
            on_drag_hover=self.drag_highlight,
        )
