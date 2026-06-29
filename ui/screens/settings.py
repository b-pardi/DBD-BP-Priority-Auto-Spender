"""settings screen: a small form, each control labeled and persisted into the single config file.

controls: enable debugging, rebind the kill/start hotkeys, matching method, binarization method, and
the post-buy settle wait. all edit app.app_state.config in memory; Save writes the whole config through
the shared serializer. the debug toggle also flips the Debug nav button immediately (App.refresh_nav).
deferred (later pass): bp-threshold, capture-region picker, calibrate.
"""

import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import detect, spender

from .. import config_io, theme

CNN_PLACEHOLDER = "CNN (coming soon)"


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

        # matching method (CNN is shown but not selectable yet)
        ctk.CTkLabel(form, text="Matching method", anchor="w").grid(
            row=3, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.matcher = ctk.CTkOptionMenu(
            form, width=180, values=list(detect.MATCHERS) + [CNN_PLACEHOLDER],
            command=self._on_matcher)
        self.matcher.set(cfg.get("matcher", "ncc"))
        self.matcher.grid(row=3, column=1, sticky="w", padx=theme.PAD, pady=4)

        # binarization method
        ctk.CTkLabel(form, text="Binarization method", anchor="w").grid(
            row=4, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.thresh = ctk.CTkOptionMenu(
            form, width=180, values=["adaptive_gaussian", "otsu", "canny"])
        self.thresh.set(cfg.get("thresh_method", "adaptive_gaussian"))
        self.thresh.grid(row=4, column=1, sticky="w", padx=theme.PAD, pady=4)

        # post-buy settle wait
        ctk.CTkLabel(form, text="Post-buy settle wait (seconds)", anchor="w").grid(
            row=5, column=0, sticky="w", padx=theme.PAD, pady=4)
        self.settle = ctk.CTkEntry(form, width=120)
        self.settle.insert(0, str(cfg.get("settle_s", spender.SETTLE_S)))
        self.settle.grid(row=5, column=1, sticky="w", padx=theme.PAD, pady=4)

        ctk.CTkButton(self, text="Save settings", command=self._save).pack(
            anchor="w", padx=theme.PAD, pady=theme.PAD)

    def _on_debug_toggle(self):
        # write through immediately so the Debug nav button can appear/disappear right away.
        if self.app.app_state.config is not None:
            self.app.app_state.config["debug"] = bool(self.debug_var.get())
        self.app.refresh_nav()

    def _on_matcher(self, value):
        if value == CNN_PLACEHOLDER:  # not built yet -> bounce back to the previous real matcher
            self.matcher.set(self.app.app_state.config.get("matcher", "ncc"))
            messagebox.showinfo("not available", "the CNN matcher is not implemented yet.")

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
        try:
            cfg["settle_s"] = float(self.settle.get())
        except ValueError:
            messagebox.showerror("invalid value", "settle wait must be a number (seconds).")
            return
        try:
            config_io.save(cfg)
        except ValueError as e:
            messagebox.showerror("config error", str(e))
            return
        self.app.app_state.config = cfg
        self.app.refresh_nav()
