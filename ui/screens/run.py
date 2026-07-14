"""run / monitor screen: start/pause + stop, status, dry-run + simulator toggles, live log.

mirrors the global hotkeys with on-screen buttons (both flip the same spender.Switch). on launch the
automation is idle and touches nothing until Start. the safe demo path is the offline simulator (no
game needed); a live, non-dry run is the only mode that actually clicks. the log pane shows the loop's
[dry-run] lines, teed from stdout via the controller's queue.
"""

import queue
import webbrowser

import customtkinter as ctk

from .. import theme
from ..library import Library
from ..run_controller import RunController

GITHUB_URL = "https://github.com/b-pardi/DBD-BP-Priority-Auto-Spender"


class RunScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.controller = None
        self.log_queue = queue.Queue()

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        keys = app.app_state.config or {}
        self._start_key = keys.get("start_key", "f7")
        self._kill_key = keys.get("kill_key", "f8")

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(head, text="Run", font=theme.FONT_TITLE).pack(side="left")
        # a soft ask, not a nag: the github page is also where updates and fixes land.
        follow = ctk.CTkLabel(
            head, text="enjoying it? ★ follow the project on GitHub",
            font=("Segoe UI", 11, "underline"), text_color=theme.ACCENT_BRIGHT, cursor="hand2")
        follow.pack(side="right")
        follow.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))

        bar = ctk.CTkFrame(self)
        bar.grid(row=1, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        # start is the app's one primary action, so it gets the ember accent; stop is destructive,
        # so it gets the only other tinted button in the app (see theme). both carry their global
        # hotkey in the label, so nobody has to dig through settings to learn the keys.
        self.start_btn = ctk.CTkButton(bar, text=f"Start ({self._start_key})", width=130,
                                       fg_color=theme.ACCENT,
                                       hover_color=theme.ACCENT_HOVER, command=self._start)
        self.start_btn.pack(side="left", padx=theme.PAD, pady=theme.PAD)
        self.stop_btn = ctk.CTkButton(bar, text=f"Stop ({self._kill_key})", width=110,
                                      fg_color=theme.DANGER,
                                      hover_color=theme.DANGER_HOVER,
                                      command=self._stop)
        self.stop_btn.pack(side="left", padx=(0, theme.PAD), pady=theme.PAD)
        self.status = ctk.CTkLabel(bar, text="Idle", font=theme.FONT_BODY)
        self.status.pack(side="left", padx=theme.PAD)

        self.sim_var = ctk.BooleanVar(value=False)  # off by default: real (non-simulator) run
        self.sim_chk = ctk.CTkCheckBox(bar, text="Use simulator", variable=self.sim_var,
                                       command=self._sync_toggles)
        self.sim_chk.pack(side="right", padx=theme.PAD)
        self.dry_var = ctk.BooleanVar(value=bool((app.app_state.config or {}).get("dry_run", False)))
        self.dry_chk = ctk.CTkCheckBox(bar, text="Dry run (no clicks)", variable=self.dry_var)
        self.dry_chk.pack(side="right", padx=theme.PAD)
        self.debug_var = ctk.BooleanVar(value=bool((app.app_state.config or {}).get("debug", False)))
        self.debug_chk = ctk.CTkCheckBox(bar, text="Debugging", variable=self.debug_var,
                                         command=self._on_debug_toggle)
        self.debug_chk.pack(side="right", padx=theme.PAD)
        self._sync_toggles()

        ctk.CTkLabel(
            self, font=theme.FONT_SMALL, justify="left",
            text=(f"hotkeys work globally, even with the game focused: {self._start_key} "
                  f"starts / pauses / resumes, {self._kill_key} stops. "
                  "live (non-dry, non-sim) runs click in-game."),
        ).grid(row=3, column=0, sticky="w", padx=theme.PAD)

        self.logbox = ctk.CTkTextbox(self, font=theme.FONT_SMALL)
        self.logbox.grid(row=2, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD)

        # arm the controller (and so the global hotkeys) at build time, not on the first Start
        # click: the start hotkey now starts runs too, and it can't do that while unregistered.
        self._ensure_controller()
        self.after(150, self._poll)

    def _sync_toggles(self):
        # the simulator path is always dry, so pin + disable the dry-run box while sim is on.
        if self.sim_var.get():
            self.dry_var.set(True)
            self.dry_chk.configure(state="disabled")
        else:
            self.dry_chk.configure(state="normal")

    def _on_debug_toggle(self):
        # write through immediately (like Settings' toggle) so the Debug nav button appears right
        # away, even mid-run, without having to leave this screen.
        if self.app.app_state.config is not None:
            self.app.app_state.config["debug"] = bool(self.debug_var.get())
        self.app.refresh_nav()

    def on_show(self):
        # picks up a debug change made on the Settings screen since this screen was built.
        self.debug_var.set(bool((self.app.app_state.config or {}).get("debug", False)))

    def _ensure_controller(self):
        if self.controller is None:
            if self.app.app_state.library is None:
                self.app.app_state.library = Library()
            rows = self.app.app_state.library.rows
            debug_screen = self.app.screens.get("debug")
            frame_sink = debug_screen.push_frame if debug_screen is not None else None
            status_sink = debug_screen.push_status if debug_screen is not None else None
            self.controller = RunController(
                rows, self.log_queue, self.app.app_state.config or {}, frame_sink=frame_sink,
                status_sink=status_sink)
            self.app.app_state.loop = self.controller  # so window-close stops the thread
        return self.controller

    def _start(self):
        cfg = self.app.app_state.config or {}
        cfg["dry_run"] = bool(self.dry_var.get())  # reflect the toggle into the config the loop reads
        self._ensure_controller().start(cfg, sim=self.sim_var.get(), dry_run=self.dry_var.get())

    def _stop(self):
        if self.controller is not None:
            self.controller.stop()

    def _poll(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.logbox.insert("end", line + "\n")
                self.logbox.see("end")
        except queue.Empty:
            pass

        # the start hotkey as a start button: pressed with no run thread alive (fresh launch, or a
        # finished/stopped run) it spawns a run with the current toggles, exactly like clicking
        # Start — so you can set up, tab into the game, and start from there. with a thread alive
        # the switch toggle has already handled pause/resume, so the event is just dropped.
        if self.controller is not None and self.controller.hotkey_start.is_set():
            self.controller.hotkey_start.clear()
            if not (self.controller.thread and self.controller.thread.is_alive()):
                self._start()

        state = self.controller.state() if self.controller else "Idle"
        self.status.configure(text=state)
        self.start_btn.configure(
            text={"Idle": "Start", "Running": "Pause", "Paused": "Resume"}[state]
                 + f" ({self._start_key})")
        # lock the mode toggles while a run thread is active so they can't change mid-run.
        active = (self.controller is not None and self.controller.thread is not None
                  and self.controller.thread.is_alive())
        if active:
            self.sim_chk.configure(state="disabled")
            self.dry_chk.configure(state="disabled")
        else:
            self.sim_chk.configure(state="normal")
            self._sync_toggles()  # restores dry_chk per the sim setting
        self.after(150, self._poll)
