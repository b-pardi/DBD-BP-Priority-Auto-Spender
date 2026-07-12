"""minimal manual drag-and-drop, since tkinter/customtkinter ship no native dnd.

Draggable attaches press/motion/release to a set of widgets. a left press that then moves past a small
threshold starts a drag: a borderless "ghost" label follows the cursor and on_hover(x_root, y_root)
fires each move so a target can show a drop indicator. releasing over a target calls
on_drop(x_root, y_root); a press that never moved is a plain click and calls on_click instead (so a
library card keeps its click-to-add behavior). tkinter gives the pressed widget an implicit pointer
grab, so motion/release keep flowing to the source even while the cursor is over another pane; the
target is found separately with winfo_containing / geometry hit-testing at drop time.

the callables (get_ghost_text, on_click, on_drop, on_hover) are read lazily so a recycled widget
(the library cards are pooled) always drags whatever row it currently holds.
"""

import tkinter as tk

from ..theme import ACCENT, BONE


class _Ghost:
    """a small borderless, semi-transparent label that trails the cursor during a drag."""

    def __init__(self, root, text):
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)          # no titlebar/border, so it reads as a floating chip
        for attr, val in (("-topmost", True), ("-alpha", 0.85)):
            try:
                self.top.attributes(attr, val)
            except tk.TclError:
                pass                              # platform without that attr, harmless
        tk.Label(self.top, text=text or "item", bg=ACCENT, fg=BONE,
                 padx=8, pady=4, bd=0).pack()

    def move(self, x_root, y_root):
        self.top.geometry(f"+{x_root + 14}+{y_root + 10}")

    def destroy(self):
        try:
            self.top.destroy()
        except tk.TclError:
            pass


class Draggable:
    """make `widgets` a drag source. see the module docstring for the click-vs-drag split."""

    THRESHOLD = 6  # px of movement before a press becomes a drag rather than a click

    def __init__(self, widgets, get_ghost_text, on_click=None, on_drop=None, on_hover=None):
        self.widgets = list(widgets)
        self.get_ghost_text = get_ghost_text
        self.on_click = on_click
        self.on_drop = on_drop
        self.on_hover = on_hover
        self._origin = None   # (x_root, y_root) of the press, until release
        self._ghost = None    # the _Ghost once movement crosses THRESHOLD, else None
        for w in self.widgets:
            w.bind("<ButtonPress-1>", self._press, add="+")
            w.bind("<B1-Motion>", self._motion, add="+")
            w.bind("<ButtonRelease-1>", self._release, add="+")

    def _press(self, event):
        self._origin = (event.x_root, event.y_root)
        self._ghost = None
        return None  # don't swallow; let normal focus/press handling proceed

    def _motion(self, event):
        if self._origin is None:
            return
        if self._ghost is None:
            dx = event.x_root - self._origin[0]
            dy = event.y_root - self._origin[1]
            if abs(dx) < self.THRESHOLD and abs(dy) < self.THRESHOLD:
                return  # still within the click deadzone
            self._ghost = _Ghost(self.widgets[0].winfo_toplevel(), self.get_ghost_text())
        self._ghost.move(event.x_root, event.y_root)
        if self.on_hover:
            self.on_hover(event.x_root, event.y_root)

    def _release(self, event):
        origin, ghost = self._origin, self._ghost
        self._origin = None
        self._ghost = None
        if ghost is not None:
            ghost.destroy()
            if self.on_hover:
                self.on_hover(None, None)      # let the target clear any drop indicator
            if self.on_drop:
                self.on_drop(event.x_root, event.y_root)
            return "break"                     # a completed drag isn't also a click
        if origin is not None and self.on_click:
            self.on_click()
