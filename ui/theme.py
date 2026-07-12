"""display constants for the ui: the palette, fonts, spacing, and two ctk draw overrides.

cosmetic only. these are NOT the detection hsv anchors (those live in src/detect + usr/); they are
just ui tints, so tune freely.

the palette is dead-by-daylight by way of a dark room: warm soot charcoal instead of ctk's neutral
blue-grey, bone text, dried blood for anything destructive, and a deep ember/bloodpoint amber as the
one accent. everything is desaturated a stop or two from the in-game colors, so the ui reads as
atmospheric rather than as a fansite. one accent hue only, so the eye always knows where the app
wants it: gold = "this is selected / this is live".
"""

import customtkinter as ctk
from customtkinter.windows.widgets.core_rendering import CTkCanvas

# two customtkinter defaults cost us far more than anything we do ourselves. both are set here (theme
# is imported before any widget exists) and both are safe: they change how ctk draws, not what.

# 1. every ctk widget is a tk canvas it draws its own rounded rect onto, and ctk picks how to draw the
# corners per platform. on windows it defaults to "font_shapes", which stamps each corner as a glyph
# from a bundled shapes font: 10 canvas items per widget, ~31ms to build one rule chip. macos already
# defaults to "polygon_shapes", which draws the same corners as a single smoothed polygon: 1 canvas
# item, ~9ms. identical look, so we just take the fast one everywhere. with ~1600 widgets in the app
# that's 1.6k canvas items instead of 16k, which is most of the build cost, most of the redraw cost,
# and most of the lag while dragging the window around.
ctk.DrawEngine.preferred_drawing_method = "polygon_shapes"

# 2. CTkScrollbar._draw and CTkOptionMenu._draw both end with self._canvas.update_idletasks(). tk's
# update_idletasks is not scoped to the widget you call it on: it flushes every pending geometry and
# redraw task in the whole app. so one scrollbar tick (we send one per wheel notch) or one option-menu
# redraw drags a full-tree layout pass behind it, and it gets worse the more widgets exist -- the
# settings screen builds in 343ms on its own but 1.9s once the app's other ~1000 widgets are there.
#
# in both cases it's the last statement of _draw and purely cosmetic (a forced repaint after
# recoloring canvas items); nothing reads geometry after it, and tk repaints on the next idle cycle
# anyway. CTkCanvas doesn't otherwise override update_idletasks and those two are its only internal
# callers, so dropping it here removes exactly those forced flushes and nothing else.
CTkCanvas.update_idletasks = lambda self: None


# ---------------------------------------------------------------- palette
# the charcoal spine. everything you READ off stays here and stays quiet: a wall of 1600 icons and a
# page of instructions have to stay legible, and dark red behind small text is a headache.
BG_DEEP = "#120d0e"     # the window itself, the darkest thing on screen. red-cast, not blue
BG_PANEL = "#1c1819"    # content panels, library cards, rule chips, entries, the run log

# oxblood: the deep-red secondary. it takes the app's *chrome* -- the rail that frames everything and
# every control you press -- so red runs through the structure without ever sitting behind copy. three
# steps rather than one, so a raised button still reads against the rail it sits on.
RAIL = "#1f1013"        # the nav rail: the deepest oxblood, the app's left edge
BLOOD = "#331a1e"       # raised controls: buttons, option menus, segmented tracks, tooltips
BLOOD_LIFT = "#45252a"  # hover on them
BORDER = "#4d2d31"      # hairlines and unselected borders, so the red threads the layout too

BG_RAISED = BLOOD       # role aliases: widgets ask for what a surface *is*, not what color it is
BG_HOVER = BLOOD_LIFT

# ember: the bloodpoint amber, taken well down in saturation. the app's only accent, so it means
# exactly one thing wherever it shows up: selected, active, or live. deep enough that BONE text sits
# on it comfortably.
ACCENT = "#8a6224"
ACCENT_HOVER = "#a2762e"
ACCENT_BRIGHT = "#d4a94e"   # borders, badges, links, the bloodweb mark -- never a text background

# fresh blood: destructive only (delete a profile, stop a run). it has to run hotter and more
# saturated than the oxblood chrome it now sits among, or "careful" stops reading as different from
# "button".
DANGER = "#a13a3e"
DANGER_HOVER = "#b94a4e"

# text
BONE = "#e6ded4"        # primary, a warm off-white rather than a hard #fff
ASH = "#9a8e85"         # secondary / hints
FOG = "#ded2c0"         # the drop-target indicator: bright and neutral, so it can't be mistaken for
                        # the ember "selected" tint it appears next to

# muted text for library cards/chips flagged not-currently-obtainable (event/retired/powers); they
# only show at all when the "show event/unavailable" filter reveals them, so dim reads as "can't buy".
MUTED_TEXT_COLOR = ("gray45", "#6b615c")

# rarity -> accent hex. null rarity (perks, powers, visceral top-tier) -> neutral grey. these keep the
# game's hue order (brown < green < blue < purple < pink, plus event gold) but are pulled toward the
# palette so a row of them doesn't fight the warm base. hue still does all the separating.
RARITY_COLORS = {
    "common": "#8a6a45",      # brown
    "uncommon": "#4a7a4f",    # green
    "rare": "#3a6690",        # blue
    "very rare": "#6b4d8f",   # purple
    "ultra rare": "#a8558c",  # pink/iri
    "event": "#c2953f",       # gold
}
NULL_RARITY_COLOR = "#5f5654"  # perks/powers/visceral, no rarity disk


def rarity_color(rarity):
    """accent color for a rarity string, neutral grey for null/unknown."""
    return RARITY_COLORS.get(rarity, NULL_RARITY_COLOR)


def _apply_palette():
    """repaint every ctk widget class from the palette above.

    this is what set_default_color_theme() does under the hood (it just loads a json into
    ThemeManager.theme), but done in code we don't have to ship and resolve a data file in the frozen
    build. the app is dark-only, so each color goes into both the [light, dark] slots and it can't
    look wrong if the appearance mode ever moves.
    """
    t = ctk.ThemeManager.theme

    def paint(widget, **colors):
        for key, value in colors.items():
            # "transparent" is a sentinel ctk compares as a bare string (`if fg_color ==
            # "transparent"`), so it must NOT be wrapped in a [light, dark] pair or the check misses
            # and tk gets handed a literal color name it doesn't know.
            t[widget][key] = value if value == "transparent" else [value, value]

    paint("CTk", fg_color=BG_DEEP)
    paint("CTkToplevel", fg_color=BG_DEEP)
    # top_fg_color is what ctk gives a frame nested directly inside another default-colored frame, so
    # it's the field the library cards and tier boxes end up floating on. oxblood there is what puts
    # the red behind the whole layout while the cards themselves stay charcoal.
    paint("CTkFrame", fg_color=BG_PANEL, top_fg_color=BLOOD, border_color=BORDER)
    paint("CTkButton", fg_color=BG_RAISED, hover_color=BG_HOVER, border_color=BORDER,
          text_color=BONE, text_color_disabled="#635857")
    paint("CTkLabel", text_color=BONE)
    paint("CTkEntry", fg_color=BG_PANEL, border_color=BORDER, text_color=BONE,
          placeholder_text_color=ASH)
    paint("CTkCheckBox", fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER,
          checkmark_color=BONE, text_color=BONE, text_color_disabled="#635857")
    paint("CTkSwitch", fg_color=BG_RAISED, progress_color=ACCENT, button_color=BONE,
          button_hover_color=FOG, text_color=BONE, text_color_disabled="#635857")
    paint("CTkSlider", fg_color=BG_RAISED, progress_color=ACCENT, button_color=ACCENT_BRIGHT,
          button_hover_color=ACCENT_HOVER)
    paint("CTkProgressBar", fg_color=BG_RAISED, progress_color=ACCENT, border_color=BORDER)
    paint("CTkOptionMenu", fg_color=BG_RAISED, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
          text_color=BONE, text_color_disabled="#635857")
    paint("CTkComboBox", fg_color=BG_PANEL, border_color=BORDER, button_color=ACCENT,
          button_hover_color=ACCENT_HOVER, text_color=BONE, text_color_disabled="#635857")
    paint("CTkScrollbar", fg_color="transparent", button_color="#463b3c",
          button_hover_color="#5d4f50")
    paint("CTkSegmentedButton", fg_color=BG_RAISED, selected_color=ACCENT,
          selected_hover_color=ACCENT_HOVER, unselected_color=BG_RAISED,
          unselected_hover_color=BG_HOVER, text_color=BONE, text_color_disabled="#635857")
    paint("CTkTextbox", fg_color=BG_PANEL, border_color=BORDER, text_color=BONE,
          scrollbar_button_color="#463b3c", scrollbar_button_hover_color="#5d4f50")
    paint("CTkRadioButton", fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER,
          text_color=BONE, text_color_disabled="#635857")
    paint("CTkScrollableFrame", label_fg_color=BG_RAISED)
    paint("DropdownMenu", fg_color=BG_PANEL, hover_color=BG_HOVER, text_color=BONE)


_apply_palette()


def mix(a, b, t):
    """blend two #rrggbb colors, t=0 -> a, t=1 -> b. used by the pulsing bloodweb mark."""
    ar, ag, ab = (int(a[i:i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (round(ar + (br - ar) * t), round(ag + (bg - ag) * t),
                              round(ab + (bb - ab) * t))


# spacing / sizing
PAD = 8
THUMB_PX = 34   # library card / chip thumbnail size (small, so rows stay compact)
ROW_H = 44      # library card height; the windowed list pitches rows by this
CHIP_H = 40     # placed-rule chip height in a tier
ACCENT_W = 4    # width of the rarity accent bar on cards/chips

# note for ctk 6.x: a CTkFrame's size must be set in the constructor (width=/height=), not via a
# later .configure(); and to actually hold that size against packed children you must turn off
# geometry propagation (pack_propagate(False)). cards/chips rely on this to stay row-height.
#
# the flip side of the same gotcha: a CTkFrame with NO pack/grid children of its own (an accent strip,
# a divider) keeps its *requested* size, and ctk's default is 200x200. so any bare strip needs an
# explicit height=1 or it silently props its parent open to 200px. that's what made the instructions
# callouts 229px tall for 60px of text.

# fonts as (family, size[, style]) tuples consumed by widget font=
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 10)
