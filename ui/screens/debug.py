"""debug / capture view: the annotated detector frame plus a maintenance group.

the left panel renders detect.draw_detections() frames (numpy bgr -> PIL -> CTkImage -> label), fed by
a worker thread through a single-slot queue and drained on the main thread with after() (Tk is not
thread-safe). the run loop / a debug grab pushes the newest frame via push_frame.

the right panel is maintenance the user asked for: open/clear the cache folder, open/clear the debug
output folder, and run the scraper (with a --force checkbox, off by default). clears are scoped to
regenerable artifacts only (the ncc *.npy cache and image files), never the sprites, the index, or
unrelated files, since in dev these dirs are the repo's data/ and .tmp/.
"""

import os
import queue
import threading

import customtkinter as ctk
import cv2
from PIL import Image

from src import paths, scraper

from .. import theme

IMG_MAX = (900, 680)  # cap the rendered frame so a 3440x1440 grab fits the panel


class DebugScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._frame_q = queue.Queue(maxsize=1)  # newest annotated frame only
        self._log_q = queue.Queue()
        self._ctk_img = None       # keep a ref so the CTkImage isn't garbage-collected
        self._scraping = False

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_image_panel()
        self._build_maintenance()
        self.after(100, self._poll)

    # thread-safe producers (called from worker threads)
    def push_frame(self, bgr):
        """replace the displayed frame with the newest annotated bgr frame (drops any stale one)."""
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        self._frame_q.put(bgr)

    def log(self, line):
        self._log_q.put(line)

    # panels
    def _build_image_panel(self):
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(left, text="Detector view", font=theme.FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=theme.PAD, pady=theme.PAD)
        self.image_label = ctk.CTkLabel(
            left, text="(no frame yet — start a run with debugging on)", font=theme.FONT_BODY)
        self.image_label.grid(row=1, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD)

    def _build_maintenance(self):
        right = ctk.CTkFrame(self, width=340)
        right.grid(row=0, column=1, sticky="ns", padx=(0, theme.PAD), pady=theme.PAD)
        right.grid_propagate(False)
        ctk.CTkLabel(right, text="Maintenance", font=theme.FONT_TITLE).pack(
            anchor="w", padx=theme.PAD, pady=theme.PAD)

        cache = ctk.CTkFrame(right)
        cache.pack(fill="x", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(cache, text="Cache (ncc templates + thumbnails)", font=theme.FONT_SMALL).pack(
            anchor="w", padx=theme.PAD, pady=(theme.PAD, 0))
        ctk.CTkButton(cache, text="Open cache folder",
                      command=lambda: self._open(paths.cache_dir())).pack(
            fill="x", padx=theme.PAD, pady=2)
        ctk.CTkButton(cache, text="Clear cache", command=self._clear_cache).pack(
            fill="x", padx=theme.PAD, pady=(2, theme.PAD))

        dbg = ctk.CTkFrame(right)
        dbg.pack(fill="x", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(dbg, text="Debug output (saved overlays)", font=theme.FONT_SMALL).pack(
            anchor="w", padx=theme.PAD, pady=(theme.PAD, 0))
        ctk.CTkButton(dbg, text="Open debug folder",
                      command=lambda: self._open(paths.debug_dir())).pack(
            fill="x", padx=theme.PAD, pady=2)
        ctk.CTkButton(dbg, text="Clear debug images", command=self._clear_debug).pack(
            fill="x", padx=theme.PAD, pady=(2, theme.PAD))

        scrp = ctk.CTkFrame(right)
        scrp.pack(fill="x", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(scrp, text="Icon library scrape", font=theme.FONT_SMALL).pack(
            anchor="w", padx=theme.PAD, pady=(theme.PAD, 0))
        self.force_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(scrp, text="--force (re-download existing)",
                        variable=self.force_var, font=theme.FONT_SMALL).pack(
            anchor="w", padx=theme.PAD, pady=2)
        self.scrape_btn = ctk.CTkButton(scrp, text="Run scraper", command=self._run_scraper)
        self.scrape_btn.pack(fill="x", padx=theme.PAD, pady=(2, theme.PAD))

        self.logbox = ctk.CTkTextbox(right, height=160, font=theme.FONT_SMALL)
        self.logbox.pack(fill="both", expand=True, padx=theme.PAD, pady=(0, theme.PAD))

    # main-thread pump
    def _poll(self):
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        try:
            self._render_frame(self._frame_q.get_nowait())
        except queue.Empty:
            pass
        # re-enable the scrape button on the main thread once the worker has finished.
        if not self._scraping and str(self.scrape_btn.cget("state")) == "disabled":
            self.scrape_btn.configure(state="normal", text="Run scraper")
        self.after(100, self._poll)

    def _append_log(self, line):
        self.logbox.insert("end", line + "\n")
        self.logbox.see("end")

    def _render_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(IMG_MAX, Image.LANCZOS)
        self._ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.image_label.configure(image=self._ctk_img, text="")

    # maintenance actions
    def _open(self, path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # windows
        except Exception as e:
            self._append_log(f"could not open {path}: {e}")

    def _clear_cache(self):
        """delete the regenerable ncc template cache (*.npy) and drop the in-memory thumbnails.
        leaves the sprites and the index untouched (in dev the cache dir is the repo data/)."""
        n = 0
        for p in paths.cache_dir().glob("*.npy"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        if self.app.app_state.library is not None:
            self.app.app_state.library.clear_thumbnail_cache()
        self._append_log(f"cleared cache: {n} ncc file(s) + in-memory thumbnails")

    def _clear_debug(self):
        """delete saved debug images (*.png/*.jpg) in the debug dir. leaves any other files alone
        (in dev the debug dir is the repo .tmp/, which may hold unrelated notes)."""
        n = 0
        for pat in ("*.png", "*.jpg", "*.jpeg"):
            for p in paths.debug_dir().glob(pat):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        self._append_log(f"cleared {n} debug image(s) from {paths.debug_dir()}")

    def _run_scraper(self):
        if self._scraping:
            return
        self._scraping = True
        self.scrape_btn.configure(state="disabled", text="Scraping…")
        force = bool(self.force_var.get())
        self.log(f"scrape started (force={force}); this can take a few minutes…")
        threading.Thread(target=self._scrape_worker, args=(force,), daemon=True).start()

    def _scrape_worker(self, force):
        try:
            categories = sorted(set(scraper.PREFIXES.values()))
            index, skipped = scraper.scrape(
                categories, scraper.DEFAULT_OUT, scraper.DEFAULT_INDEX, force=force)
            self.log(f"scrape done: {len(index)} icons indexed"
                     + (f", {len(skipped)} skipped" if skipped else ""))
            # the index changed: drop the ncc cache + thumbnails + the loaded library so they rebuild.
            for p in paths.cache_dir().glob("*.npy"):
                try:
                    p.unlink()
                except OSError:
                    pass
            if self.app.app_state.library is not None:
                self.app.app_state.library.clear_thumbnail_cache()
            self.app.app_state.library = None
            self.log("caches invalidated; library reloads on next use")
        except Exception as e:
            self.log(f"scrape failed: {type(e).__name__}: {e}")
        finally:
            self._scraping = False  # _poll re-enables the button on the main thread
