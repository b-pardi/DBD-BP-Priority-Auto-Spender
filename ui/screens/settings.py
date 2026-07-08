"""settings screen: a small form, each control labeled and persisted into the single config file.

controls: enable debugging, rebind the kill/start hotkeys, matching method, binarization method, and
the post-buy settle wait. all edit app.app_state.config in memory; Save writes the whole config through
the shared serializer. the debug toggle also flips the Debug nav button immediately (App.refresh_nav).
deferred (later pass): bp-threshold, capture-region picker, calibrate.
"""

import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import detect, ocr, spender

from .. import config_io, theme
from ..widgets import tooltip


class SettingsScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._key_capture = None  # (which, button, funcid) while capturing a hotkey

        cfg = self.app.app_state.config or {}

        ctk.CTkLabel(self, text="Settings", font=theme.FONT_TITLE).pack(
            anchor="w", padx=theme.PAD, pady=theme.PAD
        )
        form = ctk.CTkFrame(self)
        form.pack(fill="x", padx=theme.PAD, pady=theme.PAD)

        # enable debugging
        self.debug_var = ctk.BooleanVar(value=bool(cfg.get("debug", False)))
        ctk.CTkSwitch(form, text="Enable debugging (shows the Debug view)",
                      variable=self.debug_var, command=self._on_debug_toggle).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=theme.PAD, pady=theme.PAD)

        # show hover tooltips (the wiki lead sentence on library cards / tier chips)
        self.tooltips_var = ctk.BooleanVar(value=bool(cfg.get("show_tooltips", True)))
        ctk.CTkSwitch(form, text="Show tooltips on hover",
                      variable=self.tooltips_var, command=self._on_tooltips_toggle).grid(
            row=9, column=0, columnspan=2, sticky="w", padx=theme.PAD, pady=theme.PAD)

        # comparison-pool narrowing: only score the library icons the priority list cares about, so
        # a survivor run skips every killer's add-ons and vice versa (see spender.build_pool_mask).
        # inferred = each priority item's whole bloodweb source; exclusive = only the listed icons.
        # exclusive is a subset of inferred, so turning it on forces inferred on and locks it.
        self.pool_inferred_var = ctk.BooleanVar(value=bool(cfg.get("pool_inferred", True)))
        self.pool_inferred_sw = ctk.CTkSwitch(
            form, text="Narrow match pool to priority sources (recommended)",
            variable=self.pool_inferred_var)
        self.pool_inferred_sw.grid(row=10, column=0, columnspan=2, sticky="w",
                                   padx=theme.PAD, pady=(theme.PAD, 4))

        self.pool_exclusive_var = ctk.BooleanVar(value=bool(cfg.get("pool_exclusive", False)))
        ctk.CTkSwitch(form, text="Only compare against the priority list (strict)",
                      variable=self.pool_exclusive_var, command=self._on_pool_exclusive).grid(
            row=11, column=0, columnspan=2, sticky="w", padx=theme.PAD, pady=(4, theme.PAD))
        self._on_pool_exclusive()   # reflect the loaded config's lock state

        # weak-match fallback: by default a node whose ocr hover reads nothing falls back to its
        # (weak) icon match for item rules rather than being skipped. flip this on to restore the
        # old strict behavior where an unread node is skipped (config weak_match_fallback=False).
        self.skip_weak_var = ctk.BooleanVar(
            value=not bool(cfg.get("weak_match_fallback", True)))
        ctk.CTkSwitch(form, text="Skip nodes when a weak match's OCR read fails (no icon fallback)",
                      variable=self.skip_weak_var).grid(
            row=12, column=0, columnspan=2, sticky="w", padx=theme.PAD, pady=(4, theme.PAD))

        # keybinds
        ctk.CTkLabel(form, text="Start / pause hotkey", anchor="w").grid(
            row=1, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.start_btn = ctk.CTkButton(
            form, width=120, text=cfg.get("start_key", spender.START_KEY),
            command=lambda: self._capture_key("start_key", self.start_btn))
        self.start_btn.grid(row=1, column=1, sticky="w", padx=theme.PAD, pady=4)

        ctk.CTkLabel(form, text="Kill (panic) hotkey", anchor="w").grid(
            row=2, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.kill_btn = ctk.CTkButton(
            form, width=120, text=cfg.get("kill_key", spender.KILL_KEY),
            command=lambda: self._capture_key("kill_key", self.kill_btn))
        self.kill_btn.grid(row=2, column=1, sticky="w", padx=theme.PAD, pady=4)

        # matching method (cnn is the default learned matcher; ncc/ncc_masked/phash are classical)
        ctk.CTkLabel(form, text="Matching method", anchor="w").grid(
            row=3, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.matcher = ctk.CTkOptionMenu(form, width=180, values=list(detect.MATCHERS))
        self.matcher.set(cfg.get("matcher", "cnn"))
        self.matcher.grid(row=3, column=1, sticky="w", padx=theme.PAD, pady=4)

        # binarization method (the thresholding find_circles preprocesses with)
        ctk.CTkLabel(form, text="Binarization method", anchor="w").grid(
            row=4, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.thresh = ctk.CTkOptionMenu(
            form, width=180, values=["adaptive_gaussian", "otsu", "canny"])
        self.thresh.set(cfg.get("thresh_method", "adaptive_gaussian"))
        self.thresh.grid(row=4, column=1, sticky="w", padx=theme.PAD, pady=4)

        # node-localization method: the contour pass (default) vs opencv HoughCircles (detect.py)
        ctk.CTkLabel(form, text="Node detection", anchor="w").grid(
            row=5, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.node_finder = ctk.CTkOptionMenu(form, width=180, values=["contours", "hough"])
        self.node_finder.set(cfg.get("node_finder", "contours"))
        self.node_finder.grid(row=5, column=1, sticky="w", padx=theme.PAD, pady=4)

        # post-buy settle wait: pause after each buy before the next pick on the same web
        ctk.CTkLabel(form, text="Post-buy settle wait (seconds)", anchor="w").grid(
            row=6, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.settle = ctk.CTkEntry(form, width=120)
        self.settle.insert(0, str(cfg.get("settle_s", spender.SETTLE_S)))
        self.settle.grid(row=6, column=1, sticky="w", padx=theme.PAD, pady=4)

        # ocr tooltip wait: how long to let dbd's name tooltip fade in before reading it.
        # raise this if a live run logs a lot of failed ocr reads.
        ctk.CTkLabel(form, text="OCR tooltip wait (seconds)", anchor="w").grid(
            row=7, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.hover = ctk.CTkEntry(form, width=120)
        self.hover.insert(0, str(cfg.get("ocr_hover_s", ocr.HOVER_DELAY_S)))
        self.hover.grid(row=7, column=1, sticky="w", padx=theme.PAD, pady=4)

        # level transition wait: pause after the center auto-spend for the fill + next web to render
        ctk.CTkLabel(form, text="Level transition wait (seconds)", anchor="w").grid(
            row=8, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.advance = ctk.CTkEntry(form, width=120)
        self.advance.insert(0, str(cfg.get("advance_s", spender.ADVANCE_S)))
        self.advance.grid(row=8, column=1, sticky="w", padx=theme.PAD, pady=4)

        ctk.CTkButton(self, text="Save settings", command=self._save).pack(
            anchor="w", padx=theme.PAD, pady=theme.PAD)

        # wiki attribution: icon art, names, and descriptions used for training and shown here come
        # from deadbydaylight.wiki.gg. unofficial fan tool, not affiliated with Behaviour Interactive.
        ctk.CTkLabel(
            self,
            text=("icon data from deadbydaylight.wiki.gg, used for model training and shown in this ui.\n"
                  "unofficial fan tool, not affiliated with Behaviour Interactive. game assets © Behaviour Interactive."),
            font=theme.FONT_SMALL, text_color="gray", justify="left", anchor="w",
        ).pack(side="bottom", anchor="w", padx=theme.PAD, pady=theme.PAD)

    def _on_debug_toggle(self):
        # write through immediately so the Debug nav button can appear/disappear right away.
        if self.app.app_state.config is not None:
            self.app.app_state.config["debug"] = bool(self.debug_var.get())
        self.app.refresh_nav()

    def on_show(self):
        # picks up a debug change made on the Run screen since this screen was built.
        self.debug_var.set(bool((self.app.app_state.config or {}).get("debug", False)))

    def _on_tooltips_toggle(self):
        # apply live (the gate is a module flag) so hovering reflects the switch before Save.
        on = bool(self.tooltips_var.get())
        if self.app.app_state.config is not None:
            self.app.app_state.config["show_tooltips"] = on
        tooltip.set_enabled(on)

    def _on_pool_exclusive(self):
        # exclusive is a strict subset of inferred, so when it's on we force inferred on and disable
        # its switch (you can't have the priority-only pool without the inferred narrowing implied).
        if bool(self.pool_exclusive_var.get()):
            self.pool_inferred_var.set(True)
            self.pool_inferred_sw.configure(state="disabled")
        else:
            self.pool_inferred_sw.configure(state="normal")

    def _capture_key(self, which, btn):
        """capture the next keypress as the hotkey for `which`, storing its keysym (e.g. 'f8')."""
        if self._key_capture is not None:  # already capturing -> ignore
            return
        btn.configure(text="press a key…")
        top = self.winfo_toplevel()
        funcid = top.bind("<Key>", lambda e: self._on_key(which, btn, e), add="+")
        self._key_capture = (which, btn, funcid)
        top.focus_set()

    def _on_key(self, which, btn, event):
        key = event.keysym.lower()
        if self.app.app_state.config is not None:
            self.app.app_state.config[which] = key
        btn.configure(text=key)
        which_, btn_, funcid = self._key_capture
        self.winfo_toplevel().unbind("<Key>", funcid)
        self._key_capture = None

    def _save(self):
        cfg = dict(self.app.app_state.config or {})
        cfg["debug"] = bool(self.debug_var.get())
        cfg["start_key"] = self.start_btn.cget("text")
        cfg["kill_key"] = self.kill_btn.cget("text")
        cfg["matcher"] = self.matcher.get()
        cfg["thresh_method"] = self.thresh.get()
        cfg["node_finder"] = self.node_finder.get()
        cfg["show_tooltips"] = bool(self.tooltips_var.get())
        cfg["pool_exclusive"] = bool(self.pool_exclusive_var.get())
        # exclusive implies inferred (it forces it on in the ui), so persist that coupling too
        cfg["pool_inferred"] = bool(self.pool_inferred_var.get()) or cfg["pool_exclusive"]
        # switch is worded as the skip behavior, so it's the inverse of the fallback flag
        cfg["weak_match_fallback"] = not bool(self.skip_weak_var.get())
        for key, entry, label in (("settle_s", self.settle, "settle wait"),
                                  ("ocr_hover_s", self.hover, "OCR tooltip wait"),
                                  ("advance_s", self.advance, "level transition wait")):
            try:
                cfg[key] = float(entry.get())
            except ValueError:
                messagebox.showerror("invalid value", f"{label} must be a number (seconds).")
                return
        try:
            config_io.save(cfg)
        except ValueError as e:
            messagebox.showerror("config error", str(e))
            return
        self.app.app_state.config = cfg
        self.app.refresh_nav()
