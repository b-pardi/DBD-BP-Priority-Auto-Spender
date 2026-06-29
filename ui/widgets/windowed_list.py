"""a vertically virtualized list for the ~1609-row library.

customtkinter has no native virtualization (CTkScrollableFrame builds a widget per child), so we
build widgets only for the visible window plus a small buffer and recycle them on scroll. scrolling
is row-snapped (whole-row steps), which is plenty for fixed-height cards and keeps the math simple.

make_card(master) must return a widget exposing bind_row(row); the list calls that to re-skin a
pooled widget as it scrolls. set_model(rows) swaps the backing list.
"""

import customtkinter as ctk


class WindowedList(ctk.CTkFrame):
    def __init__(self, master, make_card, row_h=64, buffer=4):
        super().__init__(master)
        self.make_card = make_card
        self.row_h = row_h
        self.buffer = buffer
        self.model = []
        self.first = 0          # index of the topmost rendered row
        self._pool = []         # the recycled card widgets

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.viewport = ctk.CTkFrame(self, fg_color="transparent")
        self.viewport.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ctk.CTkScrollbar(self, command=self._on_scrollbar)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.viewport.bind("<Configure>", lambda e: self._relayout())
        # customtkinter forbids bind_all, so bind the wheel on the viewport and (in _ensure_pool) on
        # each pooled card + its children, since tk doesn't bubble wheel events to parents.
        self._bind_wheel(self.viewport)

    # public api
    def set_model(self, rows):
        self.model = rows
        self.first = 0
        self._relayout()

    # internals
    def _visible_count(self):
        h = self.viewport.winfo_height()
        return max(0, h // self.row_h) if h > 1 else 0

    def _max_first(self, visible):
        return max(0, len(self.model) - visible)

    def _ensure_pool(self, n):
        while len(self._pool) < n:
            card = self.make_card(self.viewport)
            card.configure(height=self.row_h)  # CTk keeps a configured size; place can't set height
            self._bind_wheel(card)             # so scrolling works while hovering a card
            self._pool.append(card)

    def _relayout(self):
        visible = self._visible_count()
        n = visible + self.buffer
        self._ensure_pool(n)
        total = len(self.model)
        self.first = min(self.first, self._max_first(visible))
        for i, card in enumerate(self._pool):
            mi = self.first + i
            if i < n and mi < total:
                card.bind_row(self.model[mi])
                card.place(x=0, y=i * self.row_h, relwidth=1)  # height comes from the card itself
            else:
                card.place_forget()
        self._update_scrollbar(visible, total)

    def _update_scrollbar(self, visible, total):
        if total <= visible or total == 0:
            self.scrollbar.set(0, 1)
        else:
            self.scrollbar.set(self.first / total, min(1.0, (self.first + visible) / total))

    def _scroll_by(self, rows):
        visible = self._visible_count()
        self.first = max(0, min(self.first + rows, self._max_first(visible)))
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
        self._scroll_by(-int(e.delta / 120) or (-1 if e.delta > 0 else 1))
        return "break"
