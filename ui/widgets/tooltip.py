"""a shared hover tooltip popup for library cards and tier chips.

one borderless toplevel is reused for every hover (not one per widget -- there are ~1600 cards), shown
after a short delay near the pointer and hidden when the pointer leaves. bind_tooltip() takes the text
as a callable so recycled cards (see windowed_list) show whatever row they currently hold.

globally gated by set_enabled() so the settings "show tooltips" switch flips every tooltip at once
without re-binding any widget. the gate is a module flag the app sets at startup and settings updates
on save.
"""

import customtkinter as ctk

from ..theme import BLOOD, BORDER, FONT_SMALL

_enabled = True
_tips = {}            # toplevel -> _TipWindow, one shared popup per top-level window
_TIP_WRAP = 380       # px; tooltips now carry the rendered effect text too, so give the longer
                      # copy a little more line before it wraps (still narrow enough to stay a tip)


def set_enabled(value):
    """turn all hover tooltips on/off (the settings switch). hides any visible popup when off."""
    global _enabled
    _enabled = bool(value)
    if not _enabled:
        for tip in _tips.values():
            tip.hide()


class _TipWindow:
    """the single reused popup for one top-level window. built lazily on first show."""

    def __init__(self, master):
        self.master = master
        self.win = None
        self.label = None
        self.visible = False

    def _ensure(self):
        if self.win is not None and self.win.winfo_exists():
            return
        self.win = ctk.CTkToplevel(self.master)
        self.win.withdraw()
        self.win.overrideredirect(True)              # no title bar / border, just the box
        self.win.attributes("-topmost", True)
        self.win.configure(fg_color=BORDER)          # 1px outer tint reads as a thin border
        # the field oxblood, not the control fill: a popup floats over content rather than sitting on
        # a surface, and BLOOD_LIFT is close enough to BORDER that the ring above would stop reading.
        frame = ctk.CTkFrame(self.win, fg_color=BLOOD, corner_radius=6)
        frame.pack(padx=1, pady=1)
        self.label = ctk.CTkLabel(frame, justify="left", wraplength=_TIP_WRAP, font=FONT_SMALL)
        self.label.pack(padx=8, pady=5)

    def show(self, text, x, y):
        self._ensure()
        self.label.configure(text=text)
        self.win.geometry(f"+{x}+{y}")
        self.win.deiconify()
        self.win.lift()
        self.visible = True

    def move(self, x, y):
        if self.visible and self.win is not None and self.win.winfo_exists():
            self.win.geometry(f"+{x}+{y}")

    def hide(self):
        if self.win is not None and self.win.winfo_exists():
            self.win.withdraw()
        self.visible = False


def _tip_for(widget):
    top = widget.winfo_toplevel()
    tip = _tips.get(top)
    if tip is None:
        tip = _tips[top] = _TipWindow(top)
    return tip


def bind_tooltip(widgets, text_provider, show_delay=450, hide_delay=120):
    """show a hover tooltip over each widget in `widgets`, all sharing one popup.

    text_provider() is called at hover time and returns the current text ("" -> no popup), so a
    recycled card shows whatever row it currently holds. bind every sub-widget of a card/chip (tk
    crossing events don't bubble); the short hide_delay debounces moving between those siblings so
    the popup doesn't flicker. globally suppressed while set_enabled(False).
    """
    if not widgets:
        return
    host = widgets[0]
    tip = _tip_for(host)
    timers = {"show": None, "hide": None}
    pos = {"x": 0, "y": 0}

    def cancel(which):
        if timers[which] is not None:
            try:
                host.after_cancel(timers[which])
            except Exception:
                pass
            timers[which] = None

    def do_show():
        timers["show"] = None
        if not _enabled:
            return
        text = text_provider() or ""
        if text:
            tip.show(text, pos["x"] + 14, pos["y"] + 20)

    def on_enter(event):
        pos["x"], pos["y"] = event.x_root, event.y_root
        cancel("hide")                       # crossed in from a sibling; keep the popup
        if not _enabled:
            return
        if tip.visible:
            tip.move(pos["x"] + 14, pos["y"] + 20)
            return
        cancel("show")
        timers["show"] = host.after(show_delay, do_show)

    def on_motion(event):
        pos["x"], pos["y"] = event.x_root, event.y_root
        if tip.visible:
            tip.move(pos["x"] + 14, pos["y"] + 20)

    def on_leave(_event):
        cancel("show")
        cancel("hide")
        timers["hide"] = host.after(hide_delay, tip.hide)

    for w in widgets:
        w.bind("<Enter>", on_enter, add="+")
        w.bind("<Leave>", on_leave, add="+")
        w.bind("<Motion>", on_motion, add="+")
