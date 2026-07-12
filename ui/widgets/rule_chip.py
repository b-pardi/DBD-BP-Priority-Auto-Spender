"""a placed priority rule inside a tier.

two shapes: an item rule (thumbnail + name + a rarity badge you can toggle between the pinned rarity
and "any rarity of this item") and a category rule (a text label like "any ultra rare item"). both
carry a remove (x) and a pair of up/down arrows on the right that move the rule into the adjacent
tier. in an ordered tier the chip also shows a leading rank badge and a second pair of up/down arrows
on the left that reorder the rule within its own tier (top = first pick). the chip never mutates the
rule itself; it calls back to the tier list, which owns the model.

a chip is built once per rule and then re-skinned in place by sync(), which is the only thing the
tier list needs after an edit: the rank badge and within-tier arrows are always built and just
packed/unpacked, and the rarity display is re-read from the rule. so reordering a tier, flipping it
between Random and Ordered, or toggling a rarity all cost a few configure calls instead of tearing
down and rebuilding ~10 ctk widgets per chip.
"""

import customtkinter as ctk

from ..theme import ACCENT_W, CHIP_H, FONT_BODY, FONT_SMALL, PAD, THUMB_PX, rarity_color
from .dragdrop import Draggable
from .tooltip import bind_tooltip


def _category_text(rule):
    rar = rule.get("rarity")
    return f"any {rar + ' ' if rar else ''}{rule.get('category', '?')}"


class RuleChip(ctk.CTkFrame):
    def __init__(self, master, rule, library, on_remove, on_toggle_rarity, on_move=None,
                 rank=None, on_reorder=None, on_drag=None, on_drag_hover=None):
        # ctk 6.x: fix the height in the constructor + stop propagation, else the accent CTkFrame
        # (200px default) blows the chip up and the tier list eats the whole pane.
        super().__init__(master, height=CHIP_H, corner_radius=6)
        self.pack_propagate(False)
        self.rule = rule
        self.rarity_btn = None   # item chips only; category rules bake their rarity into the label
        self.badge = None        # ordered-tier controls, built on demand by _ensure_ordered()
        self._on_reorder = on_reorder
        self._rank = None        # the rank sync() last applied (None = tier is Random)
        # rarity is painted from the constructor below, not by the first sync: a ctk color has to be
        # passed in or configured, and configuring is a second full redraw of the widget.
        self._rar = rar = rule.get("rarity")
        # widgets a drag can start from (the frame + its non-button labels); the x/arrow/rarity
        # buttons keep their own click, so dragging never fights the controls. filled as we build.
        self._drag_cells = [self]

        self.accent = ctk.CTkFrame(self, width=ACCENT_W, corner_radius=0, fg_color=rarity_color(rar))
        self.accent.pack(side="left", fill="y")
        self._drag_cells.append(self.accent)

        # right cluster, far right first: [x] then the up/down arrows so it reads "▲ ▼ x".
        # up/down move the rule into the tier above/below (no-op at the ends).
        ctk.CTkButton(self, text="x", width=24, font=FONT_SMALL,
                      command=lambda: on_remove(self.rule)).pack(side="right", padx=(2, PAD))
        if on_move is not None:
            dn_out = ctk.CTkButton(self, text="▼", width=22, font=FONT_SMALL,
                                   command=lambda: on_move(self.rule, 1))
            dn_out.pack(side="right", padx=2)
            up_out = ctk.CTkButton(self, text="▲", width=22, font=FONT_SMALL,
                                   command=lambda: on_move(self.rule, -1))
            up_out.pack(side="right", padx=2)
            bind_tooltip([up_out], lambda: "move to the tier above")
            bind_tooltip([dn_out], lambda: "move to the tier below")

        if rule.get("type") == "item":
            row = library.lookup_row(rule.get("name", ""))
            tip_targets = [self]
            if row is not None:
                thumb = ctk.CTkLabel(self, text="", image=library.thumbnail(row), width=THUMB_PX)
                thumb.pack(side="left", padx=(PAD, 0))
                tip_targets.append(thumb)
                self._drag_cells.append(thumb)
            name_lbl = ctk.CTkLabel(self, text=rule.get("name", "?"), anchor="w", font=FONT_BODY)
            name_lbl.pack(side="left", padx=PAD)
            tip_targets.append(name_lbl)
            self._drag_cells.append(name_lbl)
            # the leftmost body widget, so the ordered-tier controls below know what to pack before.
            self._anchor = thumb if row is not None else name_lbl
            # hover tooltip: the looked-up library row's lead sentence (a chip is bound to one rule
            # for its whole life, so the text is fixed at build time). no row / no desc -> no popup.
            desc = row.get("desc", "") if row is not None else ""
            bind_tooltip(tip_targets, lambda d=desc: d)
            # rarity badge doubles as the pin/any toggle; sync() re-skins it when it's toggled.
            self.rarity_btn = ctk.CTkButton(self, width=70, font=FONT_SMALL,
                                            text=(rar or "any rarity"), fg_color=rarity_color(rar),
                                            command=lambda: on_toggle_rarity(self.rule))
            self.rarity_btn.pack(side="left", padx=PAD)
        else:
            cat_lbl = ctk.CTkLabel(self, text=_category_text(rule), anchor="w", font=FONT_BODY)
            cat_lbl.pack(side="left", padx=PAD)
            self._drag_cells.append(cat_lbl)
            self._anchor = cat_lbl

        # a drag from any non-button cell moves this rule (within its tier or into another); the
        # ghost shows the rule's label. buttons above are untouched, so clicking a control still works.
        if on_drag is not None:
            ghost = rule.get("name") if rule.get("type") == "item" else _category_text(rule)
            Draggable(self._drag_cells, get_ghost_text=lambda t=ghost: t,
                      on_drop=lambda x, y: on_drag(self.rule, x, y), on_hover=on_drag_hover)

        self.sync(rank)

    def _ensure_ordered(self):
        """build the ordered-tier controls (left side): a within-tier rank badge + up/down arrows
        that reorder the rule inside this tier. distinct from the right-side cross-tier arrows.

        built on the first Ordered sync rather than up front: most tiers are Random, and these are
        three more ctk widgets (~3ms and ~9 tk widgets) per chip that a random tier never shows. once
        built they're kept and just packed/unpacked, so flipping a tier's mode stays cheap."""
        if self.badge is not None:
            return
        self.badge = ctk.CTkLabel(self, text="", width=16, font=FONT_SMALL)
        bind_tooltip([self.badge], lambda: "within-tier rank (top = picked first)")
        self.up_in = ctk.CTkButton(self, text="▲", width=18, font=FONT_SMALL,
                                   command=lambda: self._reorder(-1))
        self.dn_in = ctk.CTkButton(self, text="▼", width=18, font=FONT_SMALL,
                                   command=lambda: self._reorder(1))
        bind_tooltip([self.up_in], lambda: "move up within this tier")
        bind_tooltip([self.dn_in], lambda: "move down within this tier")

    def _reorder(self, delta):
        if self._on_reorder:
            self._on_reorder(self.rule, delta)

    def sync(self, rank=None):
        """re-skin to the tier's current state. rank=None means the tier is Random (no rank badge,
        no within-tier arrows); an int means Ordered and is the chip's 1-based rank. also re-reads
        the rule's rarity, which the tier list toggles in place.

        every branch is guarded against the value it last applied: the tier list re-syncs every chip
        after any edit, and a ctk configure() is a full canvas redraw, so an unguarded sync made an
        edit anywhere cost a repaint of every chip everywhere."""
        if (rank is None) != (self._rank is None):
            if rank is None:
                for w in (self.badge, self.up_in, self.dn_in):
                    w.pack_forget()
            else:
                self._ensure_ordered()
                # before=the body, so they land between the accent bar and the thumbnail; the pack
                # order is what fixes left-to-right, and these three are built after the body.
                self.badge.pack(side="left", padx=(4, 0), before=self._anchor)
                self.up_in.pack(side="left", padx=(2, 0), before=self._anchor)
                self.dn_in.pack(side="left", padx=(2, 4), before=self._anchor)
        if rank is not None and rank != self._rank:
            self.badge.configure(text=str(rank))
        self._rank = rank

        rar = self.rule.get("rarity")
        if rar != self._rar:
            self._rar = rar
            self.accent.configure(fg_color=rarity_color(rar))
            if self.rarity_btn is not None:
                self.rarity_btn.configure(text=(rar or "any rarity"), fg_color=rarity_color(rar))
