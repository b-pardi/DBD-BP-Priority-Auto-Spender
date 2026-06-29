"""the app shell: CTk root, left nav rail, screen switching, shared app state.

the nav rail switches the content frame between the four screens (Priorities, Settings, Run, and the
debug view, which only appears when debugging is enabled). a single AppState is built once and handed
to every screen, holding the in-memory config (edited in place, written only on Save) plus, later,
the library cache and a handle to the running spend loop. closing the window stops that loop first so
we never leak a clicking background thread.
"""

import tkinter.messagebox as messagebox

import customtkinter as ctk

from . import config_io, theme
from .screens.priorities import PrioritiesScreen
from .screens.settings import SettingsScreen
from .screens.run import RunScreen
from .screens.debug import DebugScreen


class AppState:
    """shared, in-memory app state passed to every screen.
    holds the loaded config (edited in place, persisted only on Save) and, as later screens land,
    the library row cache and a handle to the running loop thread."""

    def __init__(self):
        self.config = None        # the loaded config dict (config_io.load())
        self.config_error = None  # str if the load failed, surfaced by the ui instead of crashing
        self.library = None       # ui.library.Library, lazy-loaded by the priorities screen
        self.loop = None          # handle to the running spend loop (run screen), None when idle
        self.load_config()

    def load_config(self):
        """(re)load the config, capturing any failure as a string for the ui rather than raising."""
        try:
            self.config = config_io.load()
            self.config_error = None
        except Exception as e:  # FileNotFoundError / ValueError / json errors -> show, don't crash
            self.config = None
            self.config_error = f"{type(e).__name__}: {e}"


class App(ctk.CTk):
    # nav rail entries always shown; the debug screen is added/removed by refresh_nav.
    NAV = [("priorities", "Priorities"), ("settings", "Settings"), ("run", "Run")]

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("dbd bloodweb auto-spender")
        self.minsize(1000, 640)

        self.app_state = AppState()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.nav = ctk.CTkFrame(self, width=160, corner_radius=0)
        self.nav.grid(row=0, column=0, sticky="nsw")
        self.nav.grid_propagate(False)  # keep the rail a fixed width regardless of button text

        self.content = ctk.CTkFrame(self, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.screens = {}
        self.nav_buttons = {}
        self._active = None
        self._build_nav()
        self._build_screens()
        self.show("priorities")

        if self.app_state.config_error:
            messagebox.showerror(
                "config error",
                f"could not load the config:\n\n{self.app_state.config_error}\n\n"
                f"file: {config_io.config_file()}",
            )

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_nav(self):
        ctk.CTkLabel(self.nav, text="dbdbp", font=theme.FONT_TITLE).pack(
            padx=theme.PAD, pady=(theme.PAD * 2, theme.PAD * 2)
        )
        for key, label in self.NAV:
            b = ctk.CTkButton(self.nav, text=label, anchor="w",
                              command=lambda k=key: self.show(k))
            b.pack(fill="x", padx=theme.PAD, pady=4)
            self.nav_buttons[key] = b
        # the debug button is packed/unpacked by refresh_nav based on the debug setting.
        self.debug_btn = ctk.CTkButton(self.nav, text="Debug", anchor="w",
                                       command=lambda: self.show("debug"))
        self.nav_buttons["debug"] = self.debug_btn
        self.refresh_nav()

    def refresh_nav(self):
        """show the Debug nav button only when debugging is enabled in the config."""
        debug_on = bool((self.app_state.config or {}).get("debug"))
        if debug_on:
            self.debug_btn.pack(fill="x", padx=theme.PAD, pady=4)
        elif self.debug_btn.winfo_manager():
            self.debug_btn.pack_forget()
            if self._active == "debug":
                self.show("priorities")  # don't strand the user on a now-hidden screen

    def _build_screens(self):
        self.screens["priorities"] = PrioritiesScreen(self.content, self)
        self.screens["settings"] = SettingsScreen(self.content, self)
        self.screens["run"] = RunScreen(self.content, self)
        self.screens["debug"] = DebugScreen(self.content, self)
        for s in self.screens.values():
            s.grid(row=0, column=0, sticky="nsew")  # stacked; show() raises one

    def show(self, key):
        """raise a screen and highlight its nav button."""
        self.screens[key].tkraise()
        self._active = key
        for k, b in self.nav_buttons.items():
            b.configure(fg_color=theme.NAV_ACTIVE_COLOR if k == key else "transparent")

    def _on_close(self):
        """stop the spend loop (if running) before tearing down, so no clicker thread leaks."""
        loop = self.app_state.loop
        if loop is not None:
            stop = getattr(loop, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        self.destroy()
