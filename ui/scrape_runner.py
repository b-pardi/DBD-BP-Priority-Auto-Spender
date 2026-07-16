"""shared icon-library scrape runner with a small modal progress window.

the nav "Update icons" button and the first-run prompt both use this. the debug screen keeps its own
inline runner (it streams progress into its log textbox, and exposes the --force toggle). the scrape
runs on a worker thread; results are pumped back to the main thread with after(), since tk isn't
thread-safe.
"""

import threading
import tkinter as tk
import tkinter.messagebox as messagebox

import customtkinter as ctk

from src import detect, paths

ASSETS = paths.resource_path("ui/assets")  # bundled read-only asset, _MEIPASS/ui/assets when frozen

# src.scraper pulls in requests (~300ms), and it's only needed once the user actually starts a
# scrape -- which then runs for minutes -- so it's imported inside the worker, not at ui startup.


def invalidate_caches(app):
    """drop the regenerable template caches -- on disk AND detect's in-memory bank -- and reload the
    in-memory library against the fresh index. the Library object is kept (reloaded in place) so
    widgets holding a reference to it pick up the new rows without being rebuilt."""
    for pat in ("*.npy", "*.npz"):
        for p in paths.template_cache_dir().glob(pat):
            try:
                p.unlink()
            except OSError:
                pass
    # the disk files only matter to the NEXT process; this session's detect() would keep serving the
    # bank it already built (the 2026-07-16 stale-bank incident), so reset the in-memory one too.
    detect.reset_library_caches()
    lib = app.app_state.library
    if lib is not None:
        try:
            lib.reload()
        except Exception:
            app.app_state.library = None  # couldn't reload; force a fresh build on next use


def apply_window_icon(win):
    """give a child toplevel the app's icon instead of ctk's.

    ctk stamps its OWN logo on every CTkToplevel 200ms after it's created, and only skips that if
    iconbitmap() has already been called on the window (it checks a flag the method sets). so call
    ours right away: it both wins that race and trips the flag. the titlebar color and window
    background already match the main window without help, ctk applies the dark dwm titlebar in the
    toplevel's __init__ and theme paints CTkToplevel's fg_color.
    """
    ico = ASSETS / "icon.ico"
    try:
        if ico.exists():
            win.iconbitmap(str(ico))
            return
    except Exception:
        pass
    png = ASSETS / "icon.png"   # fallback if the .ico is missing or tk rejects it
    try:
        if png.exists():
            win._icon_img = tk.PhotoImage(file=str(png))  # keep a ref so it isn't gc'd
            win.iconphoto(False, win._icon_img)
    except Exception:
        pass


def bring_to_front(win):
    """raise a fresh CTkToplevel above the main window.

    ctk withdraws + deiconifies the toplevel during init (that's how it repaints the titlebar in the
    right color), and it can come back up *behind* the parent. a bare lift() is unreliable on windows
    when the parent holds focus, so flip topmost on and straight back off: that forces the raise
    without pinning the window there. deliberately no focus_force, it would pull focus off the entry
    inside an input dialog.
    """
    try:
        win.lift()
        win.attributes("-topmost", True)
        win.after(400, lambda: win.winfo_exists() and win.attributes("-topmost", False))
    except Exception:
        pass  # window already closed (a scrape that finished instantly); nothing to raise


def style_child_window(win):
    """the whole treatment for any toplevel we pop over the main window: our icon, not ctk's, and
    raised above the parent instead of behind it. the 220ms delay lands after ctk's own titlebar
    fiddling (a withdraw/deiconify at ~200ms) so the raise isn't immediately undone."""
    apply_window_icon(win)
    win.after(220, lambda: bring_to_front(win))


def run_scrape(app, force=False, on_done=None):
    """pop a modal progress window and scrape the wiki icon library on a worker thread.
    on success, invalidates caches and calls on_done() on the main thread. returns the window."""
    win = ctk.CTkToplevel(app)
    win.title("Updating icon library")
    win.geometry("440x160")
    win.transient(app)
    style_child_window(win)
    win.protocol("WM_DELETE_WINDOW", lambda: None)  # no closing mid-scrape
    win.after(200, win.grab_set)  # CTkToplevel needs to be viewable before grabbing
    ctk.CTkLabel(
        win, justify="left",
        text="Fetching icons from deadbydaylight.wiki.gg…",
    ).pack(padx=16, pady=(18, 4), anchor="w")
    bar = ctk.CTkProgressBar(win, mode="determinate")
    bar.set(0)
    bar.pack(fill="x", padx=16, pady=8)
    status = ctk.CTkLabel(win, justify="left", text="starting…", text_color="gray")
    status.pack(padx=16, pady=(0, 8), anchor="w")

    result = {}
    # latest progress, written by the worker thread's callback and read by poll() on the main
    # thread (tk isn't thread-safe, so the worker never touches widgets, it just updates this dict).
    prog = {"stage": "starting…", "cur": None, "tot": None}
    mode = {"m": "det"}  # track the bar's current mode so we only switch it when it changes

    def on_progress(stage, cur=None, tot=None):
        prog.update(stage=stage, cur=cur, tot=tot)

    def worker():
        try:
            from src import scraper   # deferred: see the note by the imports
            cats = sorted(set(scraper.PREFIXES.values()))
            index, skipped = scraper.scrape(
                cats, scraper.DEFAULT_OUT, scraper.DEFAULT_INDEX, force=force,
                progress=on_progress)
            result["ok"] = (len(index), len(skipped))
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"

    def poll():
        # reflect the latest progress each tick until the worker sets a result
        cur, tot = prog["cur"], prog["tot"]
        if tot:  # countable phase, show a real bar with the running count
            if mode["m"] != "det":
                bar.stop()
                bar.configure(mode="determinate")
                mode["m"] = "det"
            bar.set(min(cur / tot, 1.0))
            status.configure(text=f"{prog['stage']} ({cur}/{tot})")
        else:    # phase with no known total (up-front fetches, writing), just pulse
            if mode["m"] != "ind":
                bar.configure(mode="indeterminate")
                bar.start()
                mode["m"] = "ind"
            status.configure(text=f"{prog['stage']}…")

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
