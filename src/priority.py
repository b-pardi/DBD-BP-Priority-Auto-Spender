"""load the priority config and pick the next node to buy.

ranks detected nodes against the ordered rules: specific-item or category, with an
optional rarity filter, and a random tie-break when several nodes match one category
rule. if nothing matches, signal the center auto-spend fallback.
"""

# TODO: load_config(path) -> config dict (validate the rule schema)
# TODO: choose_next(nodes, config) -> chosen node, or None for center auto-spend


def choose_next(nodes, config):
    """return the highest-priority node to buy, or None to use center auto-spend."""
    raise NotImplementedError
