"""a vertically virtualized list for the ~1604-row library.

customtkinter has no native virtualization (CTkScrollableFrame builds a widget per child), so we
build widgets only for the visible window plus a small buffer and recycle them on scroll. scrolling
is row-snapped (whole-row steps), which is plenty for fixed-height cards and keeps the math simple.

the pool is circular: the card for model row `mi` always lives in slot `mi % n`, so scrolling one row
re-skins exactly the one card that fell off the end and just re-places the rest. re-skinning is the
expensive part (a handful of ctk configure calls each), and the old linear pool re-skinned every
visible card on every wheel notch. wheel events are also coalesced onto one relayout per idle cycle,
so a fast flick can't queue up a relayout per notch behind it.

make_card(master, height) must return a widget exposing bind_row(row); the list calls that to
re-skin a pooled widget as it scrolls. set_model(rows) swaps the backing list. the height is passed
to the factory (not configured later) because ctk 6.x only honors a frame's size from its constructor.
"""

import customtkinter as ctk

WHEEL_ROWS = 3   # rows per wheel notch (windows' own default; 1 made a 1600-row list a long haul)


class WindowedList(ctk.CTkFrame):
    def __init__(self, master, make_card, row_h=64, buffer=4):
        super().__init__(master)
        self.make_card = make_card
        self.row_h = row_h
        self.buffer = buffer
        self.model = []
        self.first = 0          # index of the topmost rendered row
        self._pool = []         # the recycled card widgets, slot i holds model rows where mi % n == i
        self._bound = []        # slot -> the model index currently skinned into it (-1 = nothing)
        self._n = 0             # live pool window size (visible + buffer); a change invalidates _bound
        self._last_h = -1       # last viewport height a relayout ran for (only height changes matter)
        self._relayout_job = None  # pending debounced relayout, so a resize burst collapses to one
        self._wheel_rows = 0    # wheel notches accumulated since the last flush
        self._wheel_job = None  # pending coalesced wheel flush
        self._sb = None         # last (start, end) pushed to the scrollbar

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.viewport = ctk.CTkFrame(self, fg_color="transparent")
        self.viewport.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ctk.CTkScrollbar(self, command=self._on_scrollbar)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.viewport.bind("<Configure>", self._on_configure)
        # customtkinter forbids bind_all, so bind the wheel on the viewport and (in _ensure_pool) on
        # each pooled card + its children, since tk doesn't bubble wheel events to parents.
        self._bind_wheel(self.viewport)

    # public api
    def set_model(self, rows):
        self.model = rows
        self.first = 0
        self._invalidate()   # same slots, different rows behind them
        self._relayout()

    def _invalidate(self):
        self._bound = [-1] * len(self._pool)

    def _on_configure(self, event):
        # a window drag/resize fires a burst of <Configure>s; only the height changes the visible-row
        # count, so skip width-only events (window moves, horizontal resizes) and coalesce the rest
        # into a single relayout on idle. this is what kept the library feeling laggy while dragging.
        if event.height == self._last_h:
            return
        self._last_h = event.height
        if self._relayout_job is not None:
            self.after_cancel(self._relayout_job)
        self._relayout_job = self.after(30, self._debounced_relayout)

    def _debounced_relayout(self):
        self._relayout_job = None
        self._relayout()

    # internals
    def _visible_count(self):
        h = self.viewport.winfo_height()
        return max(0, h // self.row_h) if h > 1 else 0

    def _max_first(self, visible):
        return max(0, len(self.model) - visible)

    def _ensure_pool(self, n):
        while len(self._pool) < n:
            card = self.make_card(self.viewport, self.row_h)  # height must come from the constructor
            self._bind_wheel(card)             # so scrolling works while hovering a card
            self._pool.append(card)
            self._bound.append(-1)

    def _relayout(self):
        visible = self._visible_count()
        n = visible + self.buffer
        self._ensure_pool(n)
        if n != self._n:
            # the slot a row maps to is mi % n, so a resize remaps every row: drop what's bound.
            self._n = n
            self._invalidate()
        total = len(self.model)
        self.first = min(self.first, self._max_first(visible))

        used = set()
        for mi in range(self.first, min(self.first + n, total)):
            slot = mi % n
            used.add(slot)
            card = self._pool[slot]
            if self._bound[slot] != mi:        # only the rows that actually changed get re-skinned
                card.bind_row(self.model[mi])
                self._bound[slot] = mi
            card.place(x=0, y=(mi - self.first) * self.row_h, relwidth=1)  # height is the card's own
        for slot, card in enumerate(self._pool):
            if slot not in used:
                card.place_forget()
        self._update_scrollbar(visible, total)

    def _update_scrollbar(self, visible, total):
        if total <= visible or total == 0:
            vals = (0.0, 1.0)
        else:
            vals = (self.first / total, min(1.0, (self.first + visible) / total))
        if vals != self._sb:      # a scrollbar set() is a canvas redraw; skip it when nothing moved
            self._sb = vals
            self.scrollbar.set(*vals)

    def _scroll_by(self, rows):
        visible = self._visible_count()
        first = max(0, min(self.first + rows, self._max_first(visible)))
        if first == self.first:
            return          # already at the end: nothing moved, so don't redraw
        self.first = first
        self._relayout()

    def _on_scrollbar(self, *args):
        # tk scrollbar protocol: ('moveto', frac) or ('scroll', n, 'units'|'pages')
        visible = self._visible_count()
        if args[0] == "moveto":
            self.first = int(round(float(args[1]) * len(self.model)))
        elif args[0] == "scroll":
            step = int(args[1]) * (max(1, visible) if args[2] == "pages" else 1)
            self.first += step
        self.first = max(0, min(self.first, self._max_first(visible)))
        self._relayout()

    def _bind_wheel(self, w):
        """bind the wheel handler on a widget and all its descendants (tk wheel events don't bubble)."""
        w.bind("<MouseWheel>", self._on_wheel)
        for c in w.winfo_children():
            self._bind_wheel(c)

    def _on_wheel(self, e):
        # windows delivers a burst of notches for one flick. bank them and relayout once on idle,
        # else each notch drags its own relayout along behind it and the list lags the cursor.
        notches = -int(e.delta / 120) or (-1 if e.delta > 0 else 1)
        self._wheel_rows += notches * WHEEL_ROWS
        if self._wheel_job is None:
            self._wheel_job = self.after_idle(self._flush_wheel)
        return "break"

    def _flush_wheel(self):
        self._wheel_job = None
        rows, self._wheel_rows = self._wheel_rows, 0
        if rows:
            self._scroll_by(rows)
