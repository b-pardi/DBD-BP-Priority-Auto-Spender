"""the app shell: CTk root, left nav rail, screen switching, shared app state.

the nav rail switches the content frame between the four screens (Priorities, Settings, Run, and the
debug view, which only appears when debugging is enabled). a single AppState is built once and handed
to every screen, holding the in-memory config (edited in place, written only on Save) plus, later,
the library cache and a handle to the running spend loop. closing the window stops that loop first so
we never leak a clicking background thread.
"""

import tkinter as tk
import tkinter.messagebox as messagebox
import webbrowser

import customtkinter as ctk

from src import paths
from . import config_io, scrape_runner, theme, updater
from .widgets import tooltip
from .widgets.emblem import BloodwebMark
from .screens.priorities import PrioritiesScreen
from .screens.settings import SettingsScreen
from .screens.run import RunScreen
from .screens.debug import DebugScreen
from .screens.instructions import InstructionsScreen

ASSETS = paths.resource_path("ui/assets")  # bundled read-only asset, _MEIPASS/ui/assets when frozen


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
    NAV = [("priorities", "Priorities"), ("settings", "Settings"), ("run", "Run"),
           ("instructions", "Instructions")]

    def __init__(self):
        self._set_app_user_model_id()  # before any window exists, so the taskbar groups us right
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("dbd bloodweb auto-spender")
        self.minsize(1000, 640)
        # explicit default window size: 1.75x the old baseline width, 1.2x its height, so the
        # two-pane priority screen has room to breathe on first launch (user can still resize).
        self.geometry("1750x768")
        self._set_window_icon()

        self.app_state = AppState()
        # apply the saved accessibility text/widget scale before any screen is built, so the whole ui
        # comes up at the chosen size (ctk.set_widget_scaling rescales live too, but building at the
        # right size avoids a visible reflow on launch).
        try:
            ctk.set_widget_scaling(float((self.app_state.config or {}).get("ui_scale", 1.0) or 1.0))
        except (TypeError, ValueError):
            pass  # a hand-edited non-numeric ui_scale shouldn't block startup
        # apply the saved hover-tooltip preference before any card/chip is built (gate is a module flag)
        tooltip.set_enabled(bool((self.app_state.config or {}).get("show_tooltips", True)))

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # the rail is the app's one full-height block of oxblood: it frames everything, and the
        # buttons on it are a lighter step of the same red so they still read as raised.
        self.nav = ctk.CTkFrame(self, width=160, corner_radius=0, fg_color=theme.RAIL)
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
        self.after(200, self._maybe_post_update_refresh)  # post-update refresh, else first-run prompt
        self.after(400, self._prewarm_library)         # ...and paint before we take cpu for icons
        self.after(1500, self._check_updates_on_launch)  # background app-update check, silent on fail

    def _prewarm_library(self):
        """decode the icon thumbnails on a worker thread, so the first scroll through the library
        never waits on a png (see Library.prewarm). deferred until after the first paint: kicked off
        during the build it competes with the main thread for ~400ms of it, and nothing needs a
        thumbnail until the user scrolls."""
        lib = self.app_state.library
        if lib is not None:
            lib.prewarm()

    # window chrome
    def _set_app_user_model_id(self):
        """give windows an explicit app id so the taskbar shows our icon (not python's) and groups
        our windows together. windows-only, harmless elsewhere."""
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("dbdbp-pas.autospender")
        except Exception:
            pass

    def _set_window_icon(self):
        """set the title-bar / taskbar icon from the bundled placeholder (replace ui/assets/icon.*).
        iconbitmap(default=) is the windows title-bar + taskbar path; iconphoto is a cross-platform
        fallback. both are guarded so a missing/!.ico asset never blocks startup."""
        ico = ASSETS / "icon.ico"
        try:
            if ico.exists():
                self.iconbitmap(default=str(ico))
        except Exception:
            pass
        png = ASSETS / "icon.png"
        try:
            if png.exists():
                self._icon_img = tk.PhotoImage(file=str(png))  # keep a ref so it isn't gc'd
                self.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _build_nav(self):
        # the brand mark is a real bloodweb, drawn from the detector's own lattice (see emblem.py).
        BloodwebMark(self.nav, size=124).pack(padx=theme.PAD, pady=(theme.PAD * 2, 4))
        ctk.CTkLabel(self.nav, text="dbdbp-pas", font=theme.FONT_TITLE).pack(
            padx=theme.PAD, pady=(0, theme.PAD * 2)
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

        # pinned to the bottom of the rail. packed side=bottom in this order, so the first packed
        # ("Update icons") sits lowest and the app self-updater sits just above it.
        #   Update icons  -> fetch/refresh the wiki icon library (needed before we can match/show).
        #   Check updates -> check github for a newer build of the app itself, install with consent.
        self.update_btn = ctk.CTkButton(self.nav, text="⟳ Update icons", command=self._update_icons)
        self.update_btn.pack(side="bottom", fill="x", padx=theme.PAD, pady=(4, theme.PAD))
        self.update_check_btn = ctk.CTkButton(self.nav, text="⭳ Check for updates",
                                              command=self._check_updates)
        self.update_check_btn.pack(side="bottom", fill="x", padx=theme.PAD, pady=4)

    def refresh_nav(self):
        """show the Debug nav button only when debugging is enabled in the config."""
        debug_on = bool((self.app_state.config or {}).get("debug"))
        if debug_on:
            self.debug_btn.pack(fill="x", padx=theme.PAD, pady=4)
        elif self.debug_btn.winfo_manager():
            self.debug_btn.pack_forget()
            if self._active == "debug":
                self.show("priorities")  # don't strand the user on a now-hidden screen

    def rebuild_settings(self):
        """tear down and rebuild the Settings screen so it repopulates every widget straight from the
        current config. used by "Restore defaults", which mutates the config then wants the screen to
        reflect it without duplicating SettingsScreen's widget-population logic."""
        old = self.screens.get("settings")
        if old is not None:
            old.destroy()
        scr = SettingsScreen(self.content, self)
        scr.grid(row=0, column=0, sticky="nsew")
        self.screens["settings"] = scr
        self.show("settings")

    def _build_screens(self):
        self.screens["priorities"] = PrioritiesScreen(self.content, self)
        self.screens["settings"] = SettingsScreen(self.content, self)
        # debug before run: the run screen arms its controller at build time (so the start hotkey
        # works from launch) and wires the debug screen's frame/status sinks into it right then.
        self.screens["debug"] = DebugScreen(self.content, self)
        self.screens["run"] = RunScreen(self.content, self)
        self.screens["instructions"] = InstructionsScreen(self.content, self)
        for s in self.screens.values():
            s.grid(row=0, column=0, sticky="nsew")  # stacked; show() raises one

    # icon library
    def _update_icons(self):
        """fetch/refresh the wiki icon library (non-force), then re-show the priorities library."""
        scrape_runner.run_scrape(self, force=False, on_done=self._after_scrape)

    def _after_scrape(self):
        scr = self.screens.get("priorities")
        if scr is not None:
            scr.refresh_after_scrape()

    # app self-update
    def _check_updates_on_launch(self):
        """silent background check on startup: only nag if there's genuinely a newer release, and
        stay quiet on any network/api error so a fresh launch offline never throws a dialog."""
        updater.check_async(self, self._on_launch_update_checked)

    def _on_launch_update_checked(self, info, err):
        if err or not info or not info.get("newer"):
            return  # offline, up to date, or no release -> say nothing on launch
        self._prompt_update(info)

    def _check_updates(self):
        """manual "Check for updates" button: always gives feedback, even when already current."""
        self.update_check_btn.configure(state="disabled", text="Checking…")
        updater.check_async(self, self._on_manual_update_checked)

    def _on_manual_update_checked(self, info, err):
        self.update_check_btn.configure(state="normal", text="⭳ Check for updates")
        if err:
            messagebox.showerror("update check failed", err)
            return
        if not info or not info.get("newer"):
            messagebox.showinfo(
                "up to date",
                f"You're on the latest version ({updater.current_version()}).")
            return
        self._prompt_update(info)

    def _prompt_update(self, info):
        """ask before downloading anything. when frozen we can self-install; from source we can only
        point at the download page."""
        if not updater.install_supported():
            if info.get("page") and messagebox.askyesno(
                "update available",
                f"A new version ({info['tag']}) is available (you have "
                f"{updater.current_version()}).\n\nSelf-install only works in the packaged app. "
                f"Open the download page?\n\n{updater.NEW_SOFTWARE_NOTE}\n\n{updater.ICONS_NOTE}",
            ):
                webbrowser.open(info["page"])
            return
        if messagebox.askyesno(
            "update available",
            f"A new version ({info['tag']}) is available (you have {updater.current_version()}).\n\n"
            "Download and install it now? The app will update itself, restart, and refresh the "
            "icon library automatically — you don't need to do anything.\n\n"
            f"{updater.NEW_SOFTWARE_NOTE}",
        ):
            updater.download_and_install(self, info)

    def _maybe_post_update_refresh(self):
        """first launch of a new build over an existing library: finish the update for the user by
        clearing the match caches and re-running the scraper, so the library, index, and template
        banks all match the new build without three manual chores (a stale bank outliving an update
        is exactly the 2026-07-16 scrambled-matches incident). frozen only -- from source a version
        bump is the developer's own business -- and detected by a version stamp rather than an
        updater marker, so manual zip swaps refresh too. falls through to the first-run prompt."""
        if paths.is_frozen():
            lib = self.app_state.library
            cur = updater.current_version()
            if updater.read_version_stamp() != cur and lib is not None and getattr(lib, "rows", None):
                # stamp BEFORE the scrape: if it fails (offline), retrying every launch would nag
                # forever; the caches are already cleared, and Update icons remains a click away.
                updater.write_version_stamp(cur)
                scrape_runner.invalidate_caches(self)
                scrape_runner.run_scrape(self, force=True, on_done=self._after_scrape)
                return
            updater.write_version_stamp(cur)
        self._maybe_first_run_scrape()

    def _maybe_first_run_scrape(self):
        """on a fresh install the index is absent and the library loads empty; offer to fetch it."""
        lib = self.app_state.library
        if lib is not None and getattr(lib, "rows", None):
            return
        if messagebox.askyesno(
            "fetch icon library",
            "No icon library was found.\n\nThe app needs the Dead by Daylight icons from the wiki "
            "before it can match or show items. Fetch them now?\n(takes a few minutes)",
        ):
            self._update_icons()

    def show(self, key):
        """raise a screen and highlight its nav button."""
        on_show = getattr(self.screens[key], "on_show", None)
        if callable(on_show):
            on_show()  # let a screen resync widgets from config before it's shown (e.g. debug toggle)
        self.screens[key].tkraise()
        self._active = key
        for k, b in self.nav_buttons.items():
            b.configure(fg_color=theme.ACCENT if k == key else "transparent")

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
