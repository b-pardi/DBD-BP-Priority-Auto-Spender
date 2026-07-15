"""settings screen: grouped controls in a scrollable body, each persisted into the single config file.

controls are organized into labeled groups (display/accessibility, hotkeys, detection & matching,
match pool, timing, stops & prestige). the middle is a CTkScrollableFrame so a short window can still
reach every control; the title stays pinned above it and the Save bar pinned below. all controls edit
app.app_state.config in memory; Save writes the whole config through the shared serializer. the debug
toggle also flips the Debug nav button immediately (App.refresh_nav).
deferred (later pass): capture-region picker, calibrate.
"""

import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import defaults, detect, ocr, spender
from src.version import __version__

from .. import config_io, theme
from ..widgets import tooltip

# accessibility text-size presets: menu label -> ctk widget-scaling factor (applied app-wide via
# ctk.set_widget_scaling, which scales fonts and widget dimensions together). 1.0 is the app default.
TEXT_SIZES = {
    "Small": 0.9,
    "Normal": 1.0,
    "Large": 1.15,
    "Larger": 1.3,
    "Largest": 1.5,
}
DEFAULT_TEXT_SIZE = "Normal"

LABEL_W = 210  # field-label column width, so the value widgets line up down each group


def _text_size_label(scale):
    """nearest preset label for a stored ui_scale float, so a hand-edited value still resolves."""
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return DEFAULT_TEXT_SIZE
    return min(TEXT_SIZES, key=lambda name: abs(TEXT_SIZES[name] - scale))


class SettingsScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._key_capture = None  # (which, button, funcid) while capturing a hotkey

        cfg = self.app.app_state.config or {}

        # title (pinned) / scrollable body (row 1) / Save bar (pinned, row 2)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(head, text="Settings", font=theme.FONT_TITLE).pack(side="left")
        ctk.CTkLabel(head, text="every knob here is explained in the Instructions tab",
                     font=theme.FONT_SMALL, text_color="gray").pack(side="left", padx=theme.PAD)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=theme.PAD)
        self.scroll.grid_columnconfigure(0, weight=1)

        # --- display & accessibility ---
        g = self._group("Display & accessibility")
        ctk.CTkLabel(g, text="Text size", anchor="w", width=LABEL_W).grid(
            row=0, column=0, sticky="w", pady=4)
        self.text_size = ctk.CTkOptionMenu(g, width=140, values=list(TEXT_SIZES),
                                           command=self._on_text_size)
        self.text_size.set(_text_size_label(cfg.get("ui_scale", 1.0)))
        self.text_size.grid(row=0, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="scales all text and controls across the app",
                     font=theme.FONT_SMALL, text_color="gray").grid(
            row=0, column=2, sticky="w", padx=theme.PAD)
        self.debug_var = ctk.BooleanVar(value=bool(cfg.get("debug", False)))
        ctk.CTkSwitch(g, text="Enable debugging (shows the Debug view)",
                      variable=self.debug_var, command=self._on_debug_toggle).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=4)
        self.tooltips_var = ctk.BooleanVar(value=bool(cfg.get("show_tooltips", True)))
        ctk.CTkSwitch(g, text="Show tooltips on hover",
                      variable=self.tooltips_var, command=self._on_tooltips_toggle).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=4)

        # --- hotkeys ---
        # global hotkeys are wired once when the run screen builds its controller (see
        # run_controller.py), not re-read live, so a rebind needs an app restart to take effect.
        g = self._group("Hotkeys", "rebound a key? restart the app for it to take effect")
        ctk.CTkLabel(g, text="Start / pause hotkey", anchor="w", width=LABEL_W).grid(
            row=0, column=0, sticky="w", pady=4)
        self.start_btn = ctk.CTkButton(
            g, width=120, text=cfg.get("start_key", spender.START_KEY),
            command=lambda: self._capture_key("start_key", self.start_btn))
        self.start_btn.grid(row=0, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="Kill (panic) hotkey", anchor="w", width=LABEL_W).grid(
            row=1, column=0, sticky="w", pady=4)
        self.kill_btn = ctk.CTkButton(
            g, width=120, text=cfg.get("kill_key", spender.KILL_KEY),
            command=lambda: self._capture_key("kill_key", self.kill_btn))
        self.kill_btn.grid(row=1, column=1, sticky="w", pady=4)

        # --- detection & matching ---
        g = self._group("Detection & matching",
                        "leave on the defaults unless a run is misreading nodes")
        # matching method (cnn is the default learned matcher; ncc/ncc_masked/phash are classical)
        ctk.CTkLabel(g, text="Matching method", anchor="w", width=LABEL_W).grid(
            row=0, column=0, sticky="w", pady=4)
        self.matcher = ctk.CTkOptionMenu(g, width=180, values=list(detect.MATCHERS))
        self.matcher.set(cfg.get("matcher", "cnn"))
        self.matcher.grid(row=0, column=1, sticky="w", pady=4)
        # binarization method (the thresholding find_circles preprocesses with)
        ctk.CTkLabel(g, text="Binarization method", anchor="w", width=LABEL_W).grid(
            row=1, column=0, sticky="w", pady=4)
        self.thresh = ctk.CTkOptionMenu(
            g, width=180, values=["adaptive_gaussian", "otsu", "canny"])
        self.thresh.set(cfg.get("thresh_method", "adaptive_gaussian"))
        self.thresh.grid(row=1, column=1, sticky="w", pady=4)
        # node-localization method: the contour pass (default) vs opencv HoughCircles (detect.py)
        ctk.CTkLabel(g, text="Node detection", anchor="w", width=LABEL_W).grid(
            row=2, column=0, sticky="w", pady=4)
        self.node_finder = ctk.CTkOptionMenu(g, width=180, values=["contours", "hough"])
        self.node_finder.set(cfg.get("node_finder", "contours"))
        self.node_finder.grid(row=2, column=1, sticky="w", pady=4)
        # presence floor: the matched-filter score an empty lattice slot must beat to be recovered
        # as a missed node (detect.recover_missed_slots). exposed so it can be tuned live on a
        # machine whose capture reads differently from the one it was calibrated on.
        ctk.CTkLabel(g, text="Presence threshold", anchor="w", width=LABEL_W).grid(
            row=3, column=0, sticky="w", pady=4)
        self.presence = ctk.CTkEntry(g, width=120)
        self.presence.insert(0, str(cfg.get("presence_thresh", spender.PRESENCE_THRESH_DEFAULT)))
        self.presence.grid(row=3, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="lower finds more missed nodes, too low invents them",
                     font=theme.FONT_SMALL, text_color="gray").grid(
            row=3, column=2, sticky="w", padx=theme.PAD)
        # matcher rescue gate (node.set_rescue_gate): a mid-score icon match is trusted anyway when
        # its runner-up trails by at least the margin, which keeps near-certain matches off the
        # slow ocr path. score direction: higher = more similar.
        ctk.CTkLabel(g, text="Matcher rescue min score", anchor="w", width=LABEL_W).grid(
            row=4, column=0, sticky="w", pady=4)
        self.rescue_min = ctk.CTkEntry(g, width=120)
        self.rescue_min.insert(0, str(cfg.get("matcher_rescue_min", spender.RESCUE_MIN_DEFAULT)))
        self.rescue_min.grid(row=4, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="lowest match score the rescue below may vouch for",
                     font=theme.FONT_SMALL, text_color="gray").grid(
            row=4, column=2, sticky="w", padx=theme.PAD)
        ctk.CTkLabel(g, text="Matcher rescue margin", anchor="w", width=LABEL_W).grid(
            row=5, column=0, sticky="w", pady=4)
        self.rescue_margin = ctk.CTkEntry(g, width=120)
        self.rescue_margin.insert(0, str(cfg.get("matcher_rescue_margin",
                                                 spender.RESCUE_MARGIN_DEFAULT)))
        self.rescue_margin.grid(row=5, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="runner-up gap that lets a mid-score match skip OCR",
                     font=theme.FONT_SMALL, text_color="gray").grid(
            row=5, column=2, sticky="w", padx=theme.PAD)

        # --- match pool ---
        # comparison-pool narrowing: only score the library icons the priority list cares about, so
        # a survivor run skips every killer's add-ons and vice versa (see spender.build_pool_mask).
        # inferred = each priority item's whole bloodweb source; exclusive = only the listed icons.
        # exclusive is a subset of inferred, so turning it on forces inferred on and locks it.
        g = self._group("Match pool",
                        "narrow which library icons each detected node is compared against")
        self.pool_inferred_var = ctk.BooleanVar(value=bool(cfg.get("pool_inferred", True)))
        self.pool_inferred_sw = ctk.CTkSwitch(
            g, text="Narrow match pool to priority sources (recommended)",
            variable=self.pool_inferred_var)
        self.pool_inferred_sw.grid(row=0, column=0, sticky="w", pady=4)
        self.pool_exclusive_var = ctk.BooleanVar(value=bool(cfg.get("pool_exclusive", False)))
        ctk.CTkSwitch(g, text="Only compare against the priority list (strict)",
                      variable=self.pool_exclusive_var, command=self._on_pool_exclusive).grid(
            row=1, column=0, sticky="w", pady=4)
        # weak-match fallback: by default a node whose ocr hover reads nothing falls back to its
        # (weak) icon match for item rules rather than being skipped. flip this on to restore the
        # old strict behavior where an unread node is skipped (config weak_match_fallback=False).
        self.skip_weak_var = ctk.BooleanVar(
            value=not bool(cfg.get("weak_match_fallback", True)))
        ctk.CTkSwitch(g, text="Skip nodes when a weak match's OCR read fails (no icon fallback)",
                      variable=self.skip_weak_var).grid(row=2, column=0, sticky="w", pady=4)
        self._on_pool_exclusive()   # reflect the loaded config's lock state

        # --- spend order ---
        # entity race: the spender already skips nodes that are bought or entity-eaten (read off each
        # node's socket ring). this goes further and reorders ties toward whatever the entity is next
        # to. it alters which node an equal-priority tie picks, so it stays opt-in.
        g = self._group("Spend order", "how the spender breaks ties between equally-ranked nodes")
        self.entity_race_var = ctk.BooleanVar(value=bool(cfg.get("entity_race", False)))
        ctk.CTkSwitch(g, text="Race the entity (buy the most at-risk node first)",
                      variable=self.entity_race_var).grid(row=0, column=0, sticky="w", pady=4)
        ctk.CTkLabel(
            g, font=theme.FONT_SMALL, text_color="gray", anchor="w", justify="left",
            text=("Alters your priorities: when several nodes tie within a tier, the one nearest the\n"
                  "entity wins instead of a random pick, since anything it eats is gone for good.\n"
                  "Never crosses tiers, and does nothing until the entity actually appears."),
        ).grid(row=1, column=0, sticky="w", pady=(0, 4))

        # --- timing ---
        g = self._group("Timing (seconds)",
                        "raise these if a live run buys too early or logs failed OCR reads")
        # post-buy settle wait: pause after each buy before the next pick on the same web
        ctk.CTkLabel(g, text="Post-buy settle wait", anchor="w", width=LABEL_W).grid(
            row=0, column=0, sticky="w", pady=4)
        self.settle = ctk.CTkEntry(g, width=120)
        self.settle.insert(0, str(cfg.get("settle_s", spender.SETTLE_S)))
        self.settle.grid(row=0, column=1, sticky="w", pady=4)
        # entity smoke wait: extra pause before the post-buy state re-read. the entity's smoke animates
        # in, and read too early an eaten node still looks available. raise it if the debug view shows
        # entity nodes going green; 0 is fine too, since state latches and a miss is caught next buy.
        ctk.CTkLabel(g, text="Entity smoke wait", anchor="w", width=LABEL_W).grid(
            row=1, column=0, sticky="w", pady=4)
        self.entity_settle = ctk.CTkEntry(g, width=120)
        self.entity_settle.insert(0, str(cfg.get("entity_settle_s", spender.ENTITY_SETTLE_S)))
        self.entity_settle.grid(row=1, column=1, sticky="w", pady=4)
        # ocr tooltip wait: how long to let dbd's name tooltip fade in before reading it.
        ctk.CTkLabel(g, text="OCR tooltip wait", anchor="w", width=LABEL_W).grid(
            row=2, column=0, sticky="w", pady=4)
        self.hover = ctk.CTkEntry(g, width=120)
        self.hover.insert(0, str(cfg.get("ocr_hover_s", ocr.HOVER_DELAY_S)))
        self.hover.grid(row=2, column=1, sticky="w", pady=4)
        # level transition wait: pause after the center auto-spend for the fill + next web to render
        ctk.CTkLabel(g, text="Level transition wait", anchor="w", width=LABEL_W).grid(
            row=3, column=0, sticky="w", pady=4)
        self.advance = ctk.CTkEntry(g, width=120)
        self.advance.insert(0, str(cfg.get("advance_s", spender.ADVANCE_S)))
        self.advance.grid(row=3, column=1, sticky="w", pady=4)

        # --- stops & prestige (all live-only; 0 disables a threshold). see spender.run. ---
        g = self._group("Stops & prestige", "live runs only; 0 turns a stop off")
        self.auto_prestige_var = ctk.BooleanVar(value=bool(cfg.get("auto_prestige", False)))
        ctk.CTkSwitch(g, text="Auto-prestige at bloodweb level 50 (spends 20k bp each time)",
                      variable=self.auto_prestige_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=4)
        # wait after clicking the prestige star before the rewards OK button appears
        ctk.CTkLabel(g, text="Prestige animation wait (seconds)", anchor="w", width=LABEL_W).grid(
            row=1, column=0, sticky="w", pady=4)
        self.prestige_wait = ctk.CTkEntry(g, width=120)
        self.prestige_wait.insert(0, str(cfg.get("prestige_wait_s", spender.PRESTIGE_WAIT_S)))
        self.prestige_wait.grid(row=1, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="Stop at bloodpoints remaining", anchor="w", width=LABEL_W).grid(
            row=2, column=0, sticky="w", pady=4)
        self.stop_bp = ctk.CTkEntry(g, width=120)
        self.stop_bp.insert(0, str(int(cfg.get("stop_bp_threshold", 0) or 0)))
        self.stop_bp.grid(row=2, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="Stop at prestige level", anchor="w", width=LABEL_W).grid(
            row=3, column=0, sticky="w", pady=4)
        self.stop_prestige = ctk.CTkEntry(g, width=120)
        self.stop_prestige.insert(0, str(int(cfg.get("stop_prestige", 0) or 0)))
        self.stop_prestige.grid(row=3, column=1, sticky="w", pady=4)
        ctk.CTkLabel(g, text="Stop at bloodweb level", anchor="w", width=LABEL_W).grid(
            row=4, column=0, sticky="w", pady=4)
        self.stop_level = ctk.CTkEntry(g, width=120)
        self.stop_level.insert(0, str(int(cfg.get("stop_level", 0) or 0)))
        self.stop_level.grid(row=4, column=1, sticky="w", pady=4)

        # wiki attribution + running version at the end of the scrolled content. the "Check for
        # updates" button lives on the nav rail; this just tells the user what they're on.
        ctk.CTkLabel(
            self.scroll,
            text=(f"version {__version__}\n"
                  "icon data from deadbydaylight.wiki.gg, used for model training and shown in this ui.\n"
                  "unofficial fan tool, not affiliated with Behaviour Interactive. game assets © Behaviour Interactive."),
            font=theme.FONT_SMALL, text_color="gray", justify="left", anchor="w",
        ).pack(fill="x", anchor="w", padx=theme.PAD, pady=theme.PAD)

        # pinned Save bar, always reachable no matter how far the body is scrolled. "Restore defaults"
        # sits on the far side so it can't be fat-fingered in place of Save.
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkButton(bar, text="Save settings", command=self._save).pack(side="left")
        ctk.CTkButton(bar, text="Restore defaults", command=self._restore_defaults,
                      fg_color=theme.BG_RAISED, hover_color=theme.BLOOD_HI).pack(side="right")

    def _group(self, title, subtitle=None):
        """a titled card in the scroll body; returns the inner frame to grid controls into."""
        card = ctk.CTkFrame(self.scroll)
        card.pack(fill="x", pady=(0, theme.PAD))
        ctk.CTkLabel(card, text=title, font=theme.FONT_TITLE, anchor="w").pack(
            fill="x", anchor="w", padx=theme.PAD, pady=(theme.PAD, 0 if subtitle else 2))
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, font=theme.FONT_SMALL, text_color="gray",
                         anchor="w", justify="left").pack(
                fill="x", anchor="w", padx=theme.PAD, pady=(0, 2))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=theme.PAD, pady=(0, theme.PAD))
        return body

    def _on_debug_toggle(self):
        # write through immediately so the Debug nav button can appear/disappear right away.
        if self.app.app_state.config is not None:
            self.app.app_state.config["debug"] = bool(self.debug_var.get())
        self.app.refresh_nav()

    def on_show(self):
        # picks up a debug change made on the Run screen since this screen was built.
        self.debug_var.set(bool((self.app.app_state.config or {}).get("debug", False)))

    def _on_text_size(self, label):
        # apply live so the user previews the size immediately, and write through to the in-memory
        # config so it survives even before Save (Save persists it to disk).
        scale = TEXT_SIZES.get(label, 1.0)
        ctk.set_widget_scaling(scale)
        if self.app.app_state.config is not None:
            self.app.app_state.config["ui_scale"] = scale

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
        cfg["ui_scale"] = TEXT_SIZES.get(self.text_size.get(), 1.0)
        cfg["pool_exclusive"] = bool(self.pool_exclusive_var.get())
        # exclusive implies inferred (it forces it on in the ui), so persist that coupling too
        cfg["pool_inferred"] = bool(self.pool_inferred_var.get()) or cfg["pool_exclusive"]
        # switch is worded as the skip behavior, so it's the inverse of the fallback flag
        cfg["weak_match_fallback"] = not bool(self.skip_weak_var.get())
        cfg["entity_race"] = bool(self.entity_race_var.get())
        cfg["auto_prestige"] = bool(self.auto_prestige_var.get())
        for key, entry, label in (("settle_s", self.settle, "settle wait"),
                                  ("entity_settle_s", self.entity_settle, "entity smoke wait"),
                                  ("ocr_hover_s", self.hover, "OCR tooltip wait"),
                                  ("advance_s", self.advance, "level transition wait"),
                                  ("prestige_wait_s", self.prestige_wait, "prestige animation wait"),
                                  ("presence_thresh", self.presence, "presence threshold"),
                                  ("matcher_rescue_min", self.rescue_min, "matcher rescue min score"),
                                  ("matcher_rescue_margin", self.rescue_margin,
                                   "matcher rescue margin")):
            try:
                cfg[key] = float(entry.get())
            except ValueError:
                messagebox.showerror("invalid value", f"{label} must be a number.")
                return
        # the stop thresholds are whole numbers, 0 = off
        for key, entry, label in (("stop_bp_threshold", self.stop_bp, "bloodpoints remaining"),
                                  ("stop_prestige", self.stop_prestige, "prestige level"),
                                  ("stop_level", self.stop_level, "bloodweb level")):
            try:
                cfg[key] = int(float(entry.get() or 0))
            except ValueError:
                messagebox.showerror("invalid value", f"{label} must be a whole number.")
                return
        try:
            config_io.save(cfg)
        except ValueError as e:
            messagebox.showerror("config error", str(e))
            return
        self.app.app_state.config = cfg
        self.app.refresh_nav()

    def _restore_defaults(self):
        """reset every settings-screen knob to defaults.DEFAULT_SETTINGS, leaving priority profiles
        untouched. mutates + persists the config, then rebuilds this screen so every widget
        repopulates from the defaulted config rather than duplicating the population logic here."""
        if not messagebox.askyesno(
                "restore defaults",
                "Reset all settings to their defaults?\n\n"
                "This affects only the knobs on this screen (display, hotkeys, detection, match "
                "pool, spend order, timing, stops & prestige). Your priority profiles are left "
                "untouched."):
            return
        cfg = dict(self.app.app_state.config or {})
        cfg.update(defaults.DEFAULT_SETTINGS)  # settings keys only; profiles/priorities untouched
        try:
            config_io.save(cfg)
        except ValueError as e:
            messagebox.showerror("config error", str(e))
            return
        self.app.app_state.config = cfg
        # the settings that take effect through a module-level toggle (not at screen build) need
        # applying by hand, same as their _on_* handlers do.
        ctk.set_widget_scaling(cfg["ui_scale"])
        tooltip.set_enabled(cfg["show_tooltips"])
        self.app.refresh_nav()        # debug defaults off -> drop the Debug nav button
        self.app.rebuild_settings()   # repopulate every widget from the defaulted config
