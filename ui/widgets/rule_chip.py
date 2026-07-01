"""a placed priority rule inside a tier.

two shapes: an item rule (thumbnail + name + a rarity badge you can toggle between the pinned rarity
and "any rarity of this item") and a category rule (a text label like "any ultra rare item"). both
carry a remove (x) and a pair of up/down arrows that move the rule into the adjacent tier. the chip
never mutates the rule itself; it calls back to the tier list, which owns the model.
"""

import customtkinter as ctk

from ..theme import ACCENT_W, CHIP_H, FONT_BODY, FONT_SMALL, PAD, THUMB_PX, rarity_color
from .tooltip import bind_tooltip


def _category_text(rule):
    rar = rule.get("rarity")
    return f"any {rar + ' ' if rar else ''}{rule.get('category', '?')}"


class RuleChip(ctk.CTkFrame):
    def __init__(self, master, rule, library, on_remove, on_toggle_rarity, on_move=None):
        # ctk 6.x: fix the height in the constructor + stop propagation, else the accent CTkFrame
        # (200px default) blows the chip up and the tier list eats the whole pane.
        super().__init__(master, height=CHIP_H, corner_radius=6)
        self.pack_propagate(False)
        self.rule = rule

        rar = rule.get("rarity")
        self.accent = ctk.CTkFrame(self, width=ACCENT_W, corner_radius=0,
                                   fg_color=rarity_color(rar))
        self.accent.pack(side="left", fill="y")

        # right cluster, far right first: [x] then the up/down arrows so it reads "▲ ▼ x".
        # up/down move the rule into the tier above/below (no-op at the ends).
        ctk.CTkButton(self, text="x", width=24, font=FONT_SMALL,
                      command=lambda: on_remove(self.rule)).pack(side="right", padx=(2, PAD))
        if on_move is not None:
            ctk.CTkButton(self, text="▼", width=22, font=FONT_SMALL,
                          command=lambda: on_move(self.rule, 1)).pack(side="right", padx=2)
            ctk.CTkButton(self, text="▲", width=22, font=FONT_SMALL,
                          command=lambda: on_move(self.rule, -1)).pack(side="right", padx=2)

        if rule.get("type") == "item":
            row = library.lookup_row(rule.get("name", ""))
            tip_targets = [self]
            if row is not None:
                thumb = ctk.CTkLabel(self, text="", image=library.thumbnail(row), width=THUMB_PX)
                thumb.pack(side="left", padx=(PAD, 0))
                tip_targets.append(thumb)
            name_lbl = ctk.CTkLabel(self, text=rule.get("name", "?"), anchor="w", font=FONT_BODY)
            name_lbl.pack(side="left", padx=PAD)
            tip_targets.append(name_lbl)
            # hover tooltip: the looked-up library row's lead sentence (chips aren't recycled, so the
            # text is fixed at build time). no row / no desc -> the provider returns "" and nothing shows.
            desc = row.get("desc", "") if row is not None else ""
            bind_tooltip(tip_targets, lambda d=desc: d)
            # rarity badge doubles as the pin/any toggle.
            ctk.CTkButton(
                self, width=70, font=FONT_SMALL,
                text=(rar or "any rarity"),
                fg_color=rarity_color(rar),
                command=lambda: on_toggle_rarity(self.rule),
            ).pack(side="left", padx=PAD)
        else:
            ctk.CTkLabel(self, text=_category_text(rule), anchor="w", font=FONT_BODY).pack(
                side="left", padx=PAD
            )
