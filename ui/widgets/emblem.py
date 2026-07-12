"""the nav-rail brand mark: an actual bloodweb, drawn from the detector's own lattice.

not a generic web. these are the real numbers the detector snaps nodes to (Resolution.LATTICE_*): a
center node plus rings of 6, 12 and 12 slots at fixed radii and phases. so the mark is literally a
picture of the thing the app reads, and if the game ever moves the lattice the logo moves with it.

the center node -- the one the spender clicks to auto-spend the rest of a level -- breathes, because
that's the node the whole tool is walking toward. the pulse is two itemconfig calls on a slow timer,
so it costs nothing.
"""

import math
import tkinter as tk

from src.resolution import Resolution

from ..theme import ACCENT, ACCENT_BRIGHT, RAIL, mix

WEB_LINE = "#513436"    # the web's threads: a lifted oxblood, barely there against the rail
NODE = "#7a6553"        # an unbought node
GLOW_LO = "#5c431c"     # the center node at rest / the halo at its dimmest
PULSE_MS = 110          # a slow breath; each tick is 2 itemconfigs


class BloodwebMark(tk.Canvas):
    """a small, self-contained bloodweb. plain tk.Canvas, not ctk: it's one widget with ~75 items
    drawn once, and it doesn't want a rounded rect or a theme behind it."""

    def __init__(self, master, size=124, bg=RAIL):
        super().__init__(master, width=size, height=size, highlightthickness=0, bd=0, bg=bg)
        self.size = size
        self._bg = bg           # the halo fades out to this, so the mark sits on whatever it's on
        self._phase = 0.0
        self._job = None
        self._draw()
        self._pulse()
        self.bind("<Destroy>", self._stop)

    def _rings(self):
        """the lattice in canvas coords: a list per ring of (x, y), scaled to fit the widget."""
        c = self.size / 2
        scale = (c - 7) / max(Resolution.LATTICE_RADII)   # 7px of air for the outermost node dot
        rings = []
        for radius, phase, slots in zip(Resolution.LATTICE_RADII, Resolution.LATTICE_PHASES,
                                        Resolution.LATTICE_SLOTS):
            r = radius * scale
            rings.append([(c + r * math.cos(math.radians(phase + i * 360.0 / slots)),
                           c - r * math.sin(math.radians(phase + i * 360.0 / slots)))
                          for i in range(slots)])
        return c, rings

    def _draw(self):
        c, rings = self._rings()

        # threads first, so the nodes sit on top of them. spokes out to the inner ring, then each node
        # to the two nearest on the ring outside it -- the 6 -> 12 -> 12 fan is what gives a real
        # bloodweb its lattice look rather than a plain spider web.
        for x, y in rings[0]:
            self.create_line(c, c, x, y, fill=WEB_LINE)
        for inner, outer in zip(rings, rings[1:]):
            step = len(outer) / len(inner)     # 2.0 for the 6 -> 12 fan, 1.0 for 12 -> 12
            for i, (x, y) in enumerate(inner):
                base = round(i * step)
                for d in range(max(2, int(step))):
                    ox, oy = outer[(base + d) % len(outer)]
                    self.create_line(x, y, ox, oy, fill=WEB_LINE)

        for ring in rings:
            for x, y in ring:
                self.create_oval(x - 2.6, y - 2.6, x + 2.6, y + 2.6, fill=NODE, outline="")

        # the center node, and a halo around it. both breathe (see _pulse).
        self._halo = self.create_oval(c - 9, c - 9, c + 9, c + 9, fill="", outline=GLOW_LO)
        self._core = self.create_oval(c - 4.5, c - 4.5, c + 4.5, c + 4.5, fill=ACCENT, outline="")

    def _pulse(self):
        if not self.winfo_exists():
            return
        self._phase = (self._phase + 0.02) % 1.0
        t = (math.sin(self._phase * 2 * math.pi) + 1) / 2   # 0..1, smooth at both ends
        self.itemconfig(self._core, fill=mix(GLOW_LO, ACCENT_BRIGHT, t))
        self.itemconfig(self._halo, outline=mix(self._bg, ACCENT, t))
        self._job = self.after(PULSE_MS, self._pulse)

    def _stop(self, _event=None):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
