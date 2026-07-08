"""run / monitor screen: start/pause + stop, status, dry-run + simulator toggles, live log.

mirrors the global hotkeys with on-screen buttons (both flip the same spender.Switch). on launch the
automation is idle and touches nothing until Start. the safe demo path is the offline simulator (no
game needed); a live, non-dry run is the only mode that actually clicks. the log pane shows the loop's
[dry-run] lines, teed from stdout via the controller's queue.
"""

import queue

import customtkinter as ctk

from .. import theme
from ..library import Library
from ..run_controller import RunController


class RunScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.controller = None
        self.log_queue = queue.Queue()

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Run", font=theme.FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=theme.PAD, pady=theme.PAD)

        bar = ctk.CTkFrame(self)
        bar.grid(row=1, column=0, sticky="ew", padx=theme.PAD, pady=theme.PAD)
        self.start_btn = ctk.CTkButton(bar, text="Start", width=110, command=self._start)
        self.start_btn.pack(side="left", padx=theme.PAD, pady=theme.PAD)
        self.stop_btn = ctk.CTkButton(bar, text="Stop", width=90, fg_color="#a83232",
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

        keys = app.app_state.config or {}
        ctk.CTkLabel(
            self, font=theme.FONT_SMALL, justify="left",
            text=(f"hotkeys: {keys.get('start_key', 'f7')} start/pause, "
                  f"{keys.get('kill_key', 'f8')} stop. live (non-dry, non-sim) runs click in-game."),
        ).grid(row=3, column=0, sticky="w", padx=theme.PAD)

        self.logbox = ctk.CTkTextbox(self, font=theme.FONT_SMALL)
        self.logbox.grid(row=2, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD)

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
            self.controller = RunController(
                rows, self.log_queue, self.app.app_state.config or {}, frame_sink=frame_sink)
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

        state = self.controller.state() if self.controller else "Idle"
        self.status.configure(text=state)
        self.start_btn.configure(
            text={"Idle": "Start", "Running": "Pause", "Paused": "Resume"}[state])
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
