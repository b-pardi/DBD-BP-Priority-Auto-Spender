"""shared icon-library scrape runner with a small modal progress window.

the nav "Update icons" button and the first-run prompt both use this. the debug screen keeps its own
inline runner (it streams progress into its log textbox, and exposes the --force toggle). the scrape
runs on a worker thread; results are pumped back to the main thread with after(), since tk isn't
thread-safe.
"""

import threading
import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import paths, scraper


def invalidate_caches(app):
    """drop the regenerable ncc template cache and reload the in-memory library against the fresh
    index. the Library object is kept (reloaded in place) so widgets holding a reference to it pick
    up the new rows without being rebuilt."""
    for p in paths.cache_dir().glob("*.npy"):
        try:
            p.unlink()
        except OSError:
            pass
    lib = app.app_state.library
    if lib is not None:
        try:
            lib.reload()
        except Exception:
            app.app_state.library = None  # couldn't reload; force a fresh build on next use


def run_scrape(app, force=False, on_done=None):
    """pop a modal progress window and scrape the wiki icon library on a worker thread.
    on success, invalidates caches and calls on_done() on the main thread. returns the window."""
    win = ctk.CTkToplevel(app)
    win.title("Updating icon library")
    win.geometry("440x150")
    win.transient(app)
    win.protocol("WM_DELETE_WINDOW", lambda: None)  # no closing mid-scrape
    win.after(200, win.grab_set)  # CTkToplevel needs to be viewable before grabbing
    ctk.CTkLabel(
        win, justify="left",
        text="Fetching icons from deadbydaylight.wiki.gg…\nThis can take a few minutes.",
    ).pack(padx=16, pady=(18, 8), anchor="w")
    bar = ctk.CTkProgressBar(win, mode="indeterminate")
    bar.pack(fill="x", padx=16, pady=8)
    bar.start()

    result = {}

    def worker():
        try:
            cats = sorted(set(scraper.PREFIXES.values()))
            index, skipped = scraper.scrape(
                cats, scraper.DEFAULT_OUT, scraper.DEFAULT_INDEX, force=force)
            result["ok"] = (len(index), len(skipped))
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"

    def poll():
        if not result:
            win.after(150, poll)
            return
        bar.stop()
        try:
            win.grab_release()
        except Exception:
            pass
        win.destroy()
        if "err" in result:
            messagebox.showerror("scrape failed", result["err"])
            return
        n, skipped = result["ok"]
        invalidate_caches(app)
        if on_done:
            on_done()
        messagebox.showinfo(
            "icon library updated",
            f"Indexed {n} icons" + (f", {skipped} skipped." if skipped else "."))

    threading.Thread(target=worker, daemon=True).start()
    win.after(150, poll)
    return win
