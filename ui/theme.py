"""display constants for the ui: rarity accent colors, fonts, spacing.

cosmetic only. these are NOT the detection hsv anchors (those live in src/detect + usr/); they are
just card/chip tints matching the post-8.7.0 rarity recolor, so tune freely. blue (rare) and purple
(very rare) are deliberately distinct here even though they sit adjacent in-game, since the ui shows
known library rarity rather than reading it off a disk.
"""

# rarity -> accent hex. null rarity (perks, powers, visceral top-tier) -> neutral grey.
RARITY_COLORS = {
    "common": "#7a5b3a",      # brown
    "uncommon": "#3f8f4f",    # green
    "rare": "#2f6db0",        # blue
    "very rare": "#7a4fb0",   # purple
    "ultra rare": "#c95aa6",  # pink/iri
    "event": "#c9a13a",       # gold
}
NULL_RARITY_COLOR = "#6b6b6b"  # perks/powers/visceral, no rarity disk


def rarity_color(rarity):
    """accent color for a rarity string, neutral grey for null/unknown."""
    return RARITY_COLORS.get(rarity, NULL_RARITY_COLOR)


# nav rail highlight for the active screen button (inactive buttons are transparent).
NAV_ACTIVE_COLOR = "#1f6aa5"

# spacing / sizing
PAD = 8
THUMB_PX = 56  # library card thumbnail size

# fonts as (family, size[, style]) tuples consumed by widget font=
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 10)
