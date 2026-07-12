"""debug / capture view: the annotated detector frame plus a maintenance group.

the left panel renders detect.draw_detections() frames (numpy bgr -> PIL -> ImageTk -> a scrollable
tk.Canvas), fed by a worker thread through a single-slot queue and drained on the main thread with
after() (Tk is not thread-safe). the run loop / a debug grab pushes the newest frame via push_frame.
the canvas has both scrollbars so a zoomed frame can be panned on either axis.

the right panel is maintenance the user asked for: open/clear the cache folder, open/clear the debug
output folder, and run the scraper (with a --force checkbox, off by default). clears are scoped to
regenerable artifacts only (the ncc *.npy cache and image files), never the sprites, the index, or
unrelated files, since in dev these dirs are the repo's data/ and .tmp/.
"""

import os
import queue
import threading
import time
import tkinter as tk

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from src import paths

# src.scraper pulls in requests (~300ms) and is only needed once a scrape actually starts, so it's
# imported in the worker (see _scrape_worker) rather than at ui startup.

from .. import theme

IMG_MAX = (900, 680)  # cap the fit view so a 3440x1440 grab fits the panel
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 10, 400, 25  # percent of native; min is low so a small fit can zoom out


class DebugScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._frame_q = queue.Queue(maxsize=1)  # newest annotated frame only
        self._status_q = queue.Queue(maxsize=1)  # newest ocr'd run status (bp/level/prestige)
        self._log_q = queue.Queue()
        self._tk_img = None        # keep a ref so the canvas PhotoImage isn't garbage-collected
        self._scraping = False
        self._last_frame = None    # newest raw bgr frame, kept full-res for zoom/save
        # zoom is always a real percent of native. _fit=True re-derives that percent to fit the panel
        # on each frame (and the % is shown), so zooming in/out steps from the actual fit scale rather
        # than jumping as if the fit view were 100%.
        self._fit = True
        self._zoom_pct = 100       # real percent of native; recomputed while _fit is on

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

    def push_status(self, status):
        """replace the shown ocr'd run status (dict of prestige/level/bp), dropping any stale one.
        fed by the run loop each live scan so the ocr reads behind the threshold/prestige features can
        be sanity-checked here against what the game actually shows."""
        try:
            self._status_q.get_nowait()
        except queue.Empty:
            pass
        self._status_q.put(status)

    # panels
    def _build_image_panel(self):
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=theme.PAD, pady=theme.PAD)
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(left, text="Detector view", font=theme.FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=theme.PAD, pady=theme.PAD)

        toolbar = ctk.CTkFrame(left, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=theme.PAD, pady=(0, theme.PAD))
        ctk.CTkButton(toolbar, text="-", width=32, command=self._zoom_out).pack(side="left")
        self.zoom_label = ctk.CTkLabel(toolbar, text="Fit", font=theme.FONT_SMALL, width=48)
        self.zoom_label.pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="+", width=32, command=self._zoom_in).pack(side="left")
        ctk.CTkButton(toolbar, text="Fit", width=48, command=self._zoom_reset).pack(
            side="left", padx=(4, 0))
        ctk.CTkButton(toolbar, text="Save frame", command=self._save_frame).pack(
            side="right")
        # ocr'd run status (prestige / bloodweb level / bp), fed by the run loop each live scan when a
        # threshold or auto-prestige is active, so the reads can be checked against the game.
        self.status_label = ctk.CTkLabel(
            toolbar, text="OCR: prestige — · level — · bp —", font=theme.FONT_SMALL)
        self.status_label.pack(side="right", padx=theme.PAD)

        # a plain tk.Canvas (not CTkScrollableFrame, which only scrolls one axis) so a zoomed frame
        # can be panned horizontally and vertically. the frame is drawn as one ImageTk image; the
        # scrollregion tracks its rendered size, so the bars engage exactly when it overflows.
        canvas_wrap = ctk.CTkFrame(left, fg_color="transparent")
        canvas_wrap.grid(row=2, column=0, sticky="nsew", padx=theme.PAD, pady=(0, theme.PAD))
        canvas_wrap.grid_rowconfigure(0, weight=1)
        canvas_wrap.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, highlightthickness=0, bd=0,
                                background=self._canvas_bg())
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ctk.CTkScrollbar(canvas_wrap, orientation="vertical", command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ctk.CTkScrollbar(canvas_wrap, orientation="horizontal", command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self._tk_img = None  # ImageTk.PhotoImage, kept so the canvas image isn't gc'd
        self._show_placeholder()
        # wheel over the frame: plain = pan vertically, shift = pan horizontally, ctrl = zoom (the
        # +/-/Fit buttons zoom too, so the gesture is a convenience not the only way).
        self.canvas.bind("<MouseWheel>", self._on_wheel_pan)
        self.canvas.bind("<Shift-MouseWheel>", self._on_wheel_hpan)
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_zoom)

    @staticmethod
    def _canvas_bg():
        """the current-theme CTkFrame fill, so the raw tk canvas blends with the ctk chrome."""
        fg = ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
        return fg[0 if ctk.get_appearance_mode() == "Light" else 1]

    def _show_placeholder(self):
        """clear the canvas to the 'no frame yet' hint (before the first frame arrives)."""
        self.canvas.delete("all")
        self.canvas.create_text(16, 16, anchor="nw", fill="gray",
                                text="(no frame yet — start a run with debugging on)")
        self.canvas.configure(scrollregion=(0, 0, 0, 0))

    def _build_maintenance(self):
        right = ctk.CTkFrame(self, width=340)
        right.grid(row=0, column=1, sticky="ns", padx=(0, theme.PAD), pady=theme.PAD)
        right.grid_propagate(False)
        ctk.CTkLabel(right, text="Maintenance", font=theme.FONT_TITLE).pack(
            anchor="w", padx=theme.PAD, pady=theme.PAD)

        cache = ctk.CTkFrame(right)
        cache.pack(fill="x", padx=theme.PAD, pady=theme.PAD)
        ctk.CTkLabel(cache, text="Cache (regenerable match templates)", font=theme.FONT_SMALL).pack(
            anchor="w", padx=theme.PAD, pady=(theme.PAD, 0))
        ctk.CTkLabel(cache, text="everything in this folder rebuilds on demand — safe to delete",
                     font=theme.FONT_SMALL, text_color="gray", justify="left", wraplength=300).pack(
            anchor="w", padx=theme.PAD, pady=(0, 2))
        ctk.CTkButton(cache, text="Open cache folder",
                      command=lambda: self._open(paths.template_cache_dir())).pack(
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
        try:
            self._render_status(self._status_q.get_nowait())
        except queue.Empty:
            pass
        # re-enable the scrape button on the main thread once the worker has finished.
        if not self._scraping and str(self.scrape_btn.cget("state")) == "disabled":
            self.scrape_btn.configure(state="normal", text="Run scraper")
        self.after(100, self._poll)

    def _append_log(self, line):
        self.logbox.insert("end", line + "\n")
        self.logbox.see("end")

    def _render_status(self, status):
        """update the ocr status readout; a None value shows as an em dash (couldn't read)."""
        def fmt(v):
            return "—" if v is None else v
        self.status_label.configure(
            text=f"OCR: prestige {fmt(status.get('prestige'))} · "
                 f"level {fmt(status.get('level'))} · bp {fmt(status.get('bp'))}")

    def _render_frame(self, bgr, keep=True):
        """render bgr at the current zoom. keep=True (a fresh frame off the queue) also stashes
        it full-res as self._last_frame, so zoom/save can re-render or write it without waiting
        on the next detector tick."""
        if keep:
            self._last_frame = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        if self._fit:
            # scale so the frame fits the panel cap without upscaling, and record that as the current
            # percent so a later zoom-in/out steps from the real fit scale (e.g. 30%), not from 100%.
            scale = min(IMG_MAX[0] / w, IMG_MAX[1] / h, 1.0)
            self._zoom_pct = max(ZOOM_MIN, round(scale * 100))
        else:
            scale = self._zoom_pct / 100
        disp = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self.canvas.configure(scrollregion=(0, 0, disp.width, disp.height))
        self._update_zoom_label()

    # zoom controls
    def _update_zoom_label(self):
        if self._fit and self._last_frame is None:
            self.zoom_label.configure(text="Fit")   # no frame yet, no real percent to show
        else:
            self.zoom_label.configure(
                text=f"{self._zoom_pct}%" + (" (fit)" if self._fit else ""))

    def _set_zoom(self, pct):
        # an explicit +/- or wheel zoom leaves fit mode and pins a concrete percent of native.
        self._fit = False
        self._zoom_pct = max(ZOOM_MIN, min(ZOOM_MAX, pct))
        if self._last_frame is not None:
            self._render_frame(self._last_frame, keep=False)
        else:
            self._update_zoom_label()

    def _zoom_in(self):
        self._set_zoom(self._zoom_pct + ZOOM_STEP)

    def _zoom_out(self):
        self._set_zoom(self._zoom_pct - ZOOM_STEP)

    def _zoom_reset(self):
        # back to fit: re-derive the fit percent from the current frame on the next render.
        self._fit = True
        if self._last_frame is not None:
            self._render_frame(self._last_frame, keep=False)
        else:
            self._update_zoom_label()

    def _on_wheel_pan(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _on_wheel_hpan(self, event):
        self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _on_wheel_zoom(self, event):
        self._zoom_in() if event.delta > 0 else self._zoom_out()
        return "break"

    def _save_frame(self):
        if self._last_frame is None:
            self._append_log("no frame to save yet")
            return
        out_dir = paths.debug_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(out_path), self._last_frame)
        self._append_log(f"saved frame: {out_path}")

    # maintenance actions
    def _open(self, path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # windows
        except Exception as e:
            self._append_log(f"could not open {path}: {e}")

    def _clear_cache(self):
        """delete the regenerable match-template cache (*.npy ncc/embed banks + *.npz ring template)
        and drop the in-memory thumbnails. the dedicated template dir holds only disposable files, so
        the sprites, index, labels, and trained model (all in the data dir proper) are untouched."""
        n = 0
        for pat in ("*.npy", "*.npz"):
            for p in paths.template_cache_dir().glob(pat):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        if self.app.app_state.library is not None:
            self.app.app_state.library.clear_thumbnail_cache()
        self._append_log(f"cleared cache: {n} template file(s) + in-memory thumbnails")

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
            from src import scraper   # deferred: see the note by the imports
            categories = sorted(set(scraper.PREFIXES.values()))
            # log one line per stage change (not per icon) so the log shows progress without flooding
            last = {"stage": None}

            def on_progress(stage, cur=None, tot=None):
                if stage != last["stage"]:
                    last["stage"] = stage
                    self.log(f"  {stage}" + (f" ({tot} items)" if tot else "…"))

            index, skipped = scraper.scrape(
                categories, scraper.DEFAULT_OUT, scraper.DEFAULT_INDEX, force=force,
                progress=on_progress)
            self.log(f"scrape done: {len(index)} icons indexed"
                     + (f", {len(skipped)} skipped" if skipped else ""))
            # the index changed: drop the template cache + thumbnails + the loaded library so they rebuild.
            for p in paths.template_cache_dir().glob("*.npy"):
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
