"""preset "template tiers": a label -> list of catch-all rules, added as a brand-new tier.

these are just shortcuts for the rules you could already build by hand with the "any [rarity]
[category] +" builder; the dropdown saves the clicks for the common cases (e.g. a catch-all "any
perk" tier at the bottom of the list). keep the rule dicts in the same shape spender validates:
a category rule is {"type": "category", "category": <cat>, optional "rarity": <rar>}.
"""

# label shown in the dropdown -> the rules that fill the new tier.
TIER_TEMPLATES = {
    "All perks": [{"type": "category", "category": "perk"}],
    "All offerings": [{"type": "category", "category": "offering"}],
    "All items": [{"type": "category", "category": "item"}],
    "All add-ons": [{"type": "category", "category": "addon"}],
    "All powers": [{"type": "category", "category": "power"}],
    "All ultra rare add-ons": [{"type": "category", "category": "addon", "rarity": "ultra rare"}],
    "All very rare add-ons": [{"type": "category", "category": "addon", "rarity": "very rare"}],
    "All ultra rare items": [{"type": "category", "category": "item", "rarity": "ultra rare"}],
    "All event items": [{"type": "category", "category": "item", "rarity": "event"}],
}

TEMPLATE_LABELS = list(TIER_TEMPLATES)
