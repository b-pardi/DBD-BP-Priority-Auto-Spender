"""a placed priority rule inside a tier.

two shapes: an item rule (thumbnail + name + a rarity badge you can toggle between the pinned rarity
and "any rarity of this item") and a category rule (a text label like "any ultra rare item"). both
carry a remove (x). the chip never mutates the rule itself; it calls back to the tier list, which
owns the model.
"""

import customtkinter as ctk

from ..theme import FONT_BODY, FONT_SMALL, PAD, rarity_color


def _category_text(rule):
    rar = rule.get("rarity")
    return f"any {rar + ' ' if rar else ''}{rule.get('category', '?')}"


class RuleChip(ctk.CTkFrame):
    def __init__(self, master, rule, library, on_remove, on_toggle_rarity):
        super().__init__(master, corner_radius=6)
        self.rule = rule

        rar = rule.get("rarity")
        self.accent = ctk.CTkFrame(self, width=5, corner_radius=0,
                                   fg_color=rarity_color(rar))
        self.accent.pack(side="left", fill="y")

        if rule.get("type") == "item":
            row = library.lookup_row(rule.get("name", ""))
            if row is not None:
                ctk.CTkLabel(self, text="", image=library.thumbnail(row), width=40).pack(
                    side="left", padx=(PAD, 0)
                )
            ctk.CTkLabel(self, text=rule.get("name", "?"), anchor="w", font=FONT_BODY).pack(
                side="left", padx=PAD
            )
            # rarity badge doubles as the pin/any toggle.
            badge = ctk.CTkButton(
                self, width=70, font=FONT_SMALL,
                text=(rar or "any rarity"),
                fg_color=rarity_color(rar),
                command=lambda: on_toggle_rarity(self.rule),
            )
            badge.pack(side="left", padx=PAD)
        else:
            ctk.CTkLabel(self, text=_category_text(rule), anchor="w", font=FONT_BODY).pack(
                side="left", padx=PAD
            )

        ctk.CTkButton(self, text="x", width=24, font=FONT_SMALL,
                      command=lambda: on_remove(self.rule)).pack(side="right", padx=PAD)
