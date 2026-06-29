"""the "any [rarity] [category] [+]" control for adding category rules to the selected tier.

specific items come from the library, but rules like "any offering" or "any ultra rare item" have no
single icon, so they are built here. both dropdowns are fed from the validators (single source of
truth); rarity is optional ("any"). clicking + hands a category rule dict to on_add.
"""

import customtkinter as ctk

from src import detect, spender

from ..theme import FONT_SMALL, PAD


class RuleBuilder(ctk.CTkFrame):
    def __init__(self, master, on_add):
        super().__init__(master)
        self.on_add = on_add

        ctk.CTkLabel(self, text="any", font=FONT_SMALL).pack(side="left", padx=(PAD, 0))
        self.rarity = ctk.CTkOptionMenu(self, width=110, values=["any"] + list(detect.RARITIES))
        self.rarity.set("any")
        self.rarity.pack(side="left", padx=PAD)
        self.category = ctk.CTkOptionMenu(self, width=110, values=sorted(spender.VALID_CATEGORIES))
        self.category.set("offering")
        self.category.pack(side="left", padx=PAD)
        ctk.CTkButton(self, text="+ add to tier", command=self._add).pack(side="left", padx=PAD)

    def _add(self):
        rule = {"type": "category", "category": self.category.get()}
        if self.rarity.get() != "any":
            rule["rarity"] = self.rarity.get()
        self.on_add(rule)
