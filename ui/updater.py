"""self-update: check github for a newer release and, with consent, install it in place.

split into three concerns so the ui stays thin:
  - check: hit the github releases api, decide if the newest tag is newer than src.version. pure
    network + parsing, safe to run on a worker thread (never touches tk).
  - check_async: run that on a thread and hand the result back on the main thread via app.after.
  - download_and_install: with the user already consented, stream the release zip down behind a
    modal progress window, stage it, then hand off to a tiny batch swapper that waits for us to
    exit, mirrors the new files over the install folder, and relaunches. the running .exe/.dlls are
    locked while we're alive, so the swap MUST happen from an outside process after we quit.

nothing here downloads without an explicit yes: check() only reads json, and download_and_install is
only ever called after a confirm dialog. self-install only applies to the frozen exe (onedir); from
source there's no bundle to swap, so install_supported() is false and the ui points at the releases
page instead.
"""

import os
import re
import subprocess
import sys
import threading
import zipfile
import tkinter.messagebox as messagebox
from pathlib import Path

import customtkinter as ctk

from src import paths
from src.version import __version__, REPO
from .scrape_runner import style_child_window   # our icon + raised above the parent, see there

# requests pulls in ~300ms and is only needed when the user actually checks/downloads, so it's
# imported inside the worker functions, never at ui startup.

_API = f"https://api.github.com/repos/{REPO}/releases"

# tacked onto every prompt raised because a newer release exists. the app and the icon library update
# separately: a release can carry a scraper fix or a new chapter, and neither reaches the user until
# they re-run the scrape, so say so wherever we tell them an update is waiting.
ICONS_NOTE = ("Note: You may need to Update Icons as well (the ⟳ Update icons button in the "
              "sidebar) to pick up new items and library fixes from this release.")


def current_version():
    return __version__


def install_supported():
    """true only for the frozen onedir exe, where there's an install folder we can swap.
    from source there's no bundle, so the ui offers the download page instead of a self-install."""
    return paths.is_frozen()


# version compare
def _parse(tag):
    """"v0.1.0-alpha" -> ((0,1,0), 1|0 release-flag, "alpha"); None if it isn't an X.Y.Z tag.
    the release-flag makes a plain X.Y.Z sort ABOVE any X.Y.Z-prerelease, and prerelease strings sort
    against each other lexically (good enough for alpha/beta/rc)."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:[-.]?(.*))?$", tag.strip().lstrip("vV"))
    if not m:
        return None
    nums = tuple(int(x) for x in m.group(1, 2, 3))
    pre = (m.group(4) or "").strip()
    return (nums, 0 if pre else 1, pre)


def is_newer(latest_tag, current=None):
    """is latest_tag a newer version than the running one? falls back to a plain != if either tag
    doesn't parse, so a weird tag still surfaces as "something changed" rather than silently never
    updating."""
    cur = current or current_version()
    lp, cp = _parse(latest_tag), _parse(cur)
    if lp is None or cp is None:
        return latest_tag.lstrip("vV") != cur.lstrip("vV")
    return lp > cp


# github query
def check(timeout=8):
    """query the releases list (prereleases included) and return the highest-VERSION non-draft as a
    dict, with a "newer" flag vs the running version. None if the repo has no usable release.
    raises on network/http errors so the caller can decide whether to surface them."""
    import requests

    r = requests.get(_API, headers={"Accept": "application/vnd.github+json"}, timeout=timeout)
    r.raise_for_status()

    # the api orders by creation date, which is NOT version order: publish a 0.1.1 hotfix on the old
    # line after 0.2.0 is out and the first entry is the *older* version, so taking the first result
    # would tell a 0.1.0 user about 0.1.1 and never mention 0.2.0 at all. pick the max by version.
    best = None
    for rel in r.json():
        if rel.get("draft"):
            continue
        tag = rel.get("tag_name") or ""
        if best is None or is_newer(tag, best.get("tag_name") or ""):
            best = rel
    if best is None:
        return None

    tag = best.get("tag_name") or ""
    zip_url = None
    for a in best.get("assets", []):
        if (a.get("name") or "").lower().endswith(".zip"):
            zip_url = a.get("browser_download_url")
            break
    return {
        "tag": tag,
        "name": best.get("name") or tag,
        "body": best.get("body") or "",
        "url": zip_url,                       # the onedir release zip, None if not attached
        "page": best.get("html_url"),         # the human releases page (dev fallback)
        "newer": is_newer(tag),
    }


def check_async(app, on_result, timeout=8):
    """run check() on a worker thread and deliver (info_or_None, err_or_None) to on_result on the
    main thread. tk isn't thread-safe, so the worker only marshals back through app.after."""

    def worker():
        try:
            info = check(timeout=timeout)
            app.after(0, lambda: on_result(info, None))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            app.after(0, lambda: on_result(None, msg))

    threading.Thread(target=worker, daemon=True).start()


# install
def download_and_install(app, info):
    """consented path: stream the release zip behind a modal progress bar, then apply + restart.
    only meaningful when frozen (install_supported)."""
    url = info.get("url")
    if not install_supported():
        # from source there's nothing to swap; send them to the releases page instead.
        if info.get("page") and messagebox.askyesno(
            "update available",
            f"A new version ({info['tag']}) is available, but self-install only works in the "
            f"packaged app.\n\n{ICONS_NOTE}\n\nOpen the download page?",
        ):
            import webbrowser
            webbrowser.open(info["page"])
        return
    if not url:
        messagebox.showerror(
            "update unavailable",
            "The new release has no downloadable package attached yet. Please update manually from "
            "the releases page.")
        return

    # staging lives under the user dir, NOT the install dir: the swapper mirrors the install dir with
    # /MIR, which would purge the very files it's copying from. fresh each time so a half-finished
    # prior attempt can't poison the extract.
    updir = paths.user_base() / "update"
    try:
        import shutil
        if updir.exists():
            shutil.rmtree(updir, ignore_errors=True)
        updir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        messagebox.showerror("update failed", f"Could not prepare a staging folder:\n{e}")
        return
    zip_path = updir / "release.zip"

    win = ctk.CTkToplevel(app)
    win.title("Updating")
    win.geometry("440x160")
    win.transient(app)
    style_child_window(win)
    win.protocol("WM_DELETE_WINDOW", lambda: None)  # no closing mid-download
    win.after(200, win.grab_set)
    ctk.CTkLabel(win, justify="left", text=f"Downloading {info['tag']}…").pack(
        padx=16, pady=(18, 4), anchor="w")
    bar = ctk.CTkProgressBar(win, mode="determinate")
    bar.set(0)
    bar.pack(fill="x", padx=16, pady=8)
    status = ctk.CTkLabel(win, justify="left", text="starting…", text_color="gray")
    status.pack(padx=16, pady=(0, 8), anchor="w")

    result = {}
    prog = {"got": 0, "tot": None}

    def worker():
        try:
            import requests
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                tot = int(r.headers.get("Content-Length") or 0)
                prog["tot"] = tot or None
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
                            prog["got"] += len(chunk)
            # extract, find the folder that actually holds the exe, then hand off to the swapper.
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(updir / "extracted")
            src_root = _find_app_root(updir / "extracted")
            if src_root is None:
                raise RuntimeError("could not find the app folder inside the release zip")
            result["src"] = src_root
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"

    def poll():
        got, tot = prog["got"], prog["tot"]
        if tot:
            bar.set(min(got / tot, 1.0))
            status.configure(text=f"{got // 1024 // 1024} / {tot // 1024 // 1024} MB")
        else:
            status.configure(text=f"{got // 1024 // 1024} MB…")
        if not result:
            win.after(150, poll)
            return
        try:
            win.grab_release()
        except Exception:
            pass
        win.destroy()
        if "err" in result:
            messagebox.showerror("update failed", result["err"])
            return
        _apply_and_restart(app, result["src"], updir)

    threading.Thread(target=worker, daemon=True).start()
    win.after(150, poll)


def _find_app_root(extract_dir):
    """the folder inside the extracted zip that holds the new build.

    matched by SHAPE (a top-level .exe sitting next to pyinstaller's _internal/), never by the
    running exe's filename. keying on sys.executable would mean the first release that ever renames
    the executable can't install itself over the previous one -- the old copy would go looking for
    its own name inside the new zip, not find it, and give up. that's a one-way trap: it only bites
    the version *after* the rename ships, when it's too late to fix. so we never depend on the name."""
    fallback = None
    for root, dirs, files in os.walk(extract_dir):
        if not any(f.lower().endswith(".exe") for f in files):
            continue
        if "_internal" in dirs:      # pyinstaller onedir: unambiguous, take it
            return Path(root)
        if fallback is None:
            fallback = Path(root)    # shallowest dir with an exe, in case the layout ever changes
    return fallback


def _payload_exe(src_root):
    """the exe to relaunch after the swap: whatever the NEW build calls itself.

    /MIR purges files the new build doesn't have, so after a rename the old executable is gone and
    relaunching sys.executable by name would just fail. prefer our own name when it's still there
    (the normal case), else take whatever exe the payload shipped."""
    names = sorted(p.name for p in src_root.iterdir() if p.suffix.lower() == ".exe")
    cur = os.path.basename(sys.executable)
    return cur if cur in names else (names[0] if names else cur)


def _apply_and_restart(app, src_root, updir):
    """write a batch swapper that waits for THIS process to exit, mirrors the new files over the
    install folder, relaunches the exe, then deletes itself + the staging dir. launched detached so
    it outlives us, then we quit (which unlocks the exe/dlls it needs to overwrite)."""
    install_dir = os.path.dirname(sys.executable)
    new_exe = os.path.join(install_dir, _payload_exe(src_root))   # post-swap name, may differ from ours
    old_exe = sys.executable
    log = updir / "update.log"
    bat = updir / "apply_update.bat"

    # only _internal gets /MIR (mirror + purge stale): pyinstaller owns every file in there. the
    # install dir itself gets /E (add + overwrite, never delete) -- if someone extracted the exe loose
    # into a folder holding their own files, /MIR aimed there would wipe them. and this swapper is the
    # one that installs the NEXT release, so a wipe bug here is only fixable after the fact.
    payload_internal = Path(src_root) / "_internal"
    if payload_internal.is_dir():
        copies = [
            (str(payload_internal), os.path.join(install_dir, "_internal"), "/MIR"),
            (str(src_root), install_dir, f'/E /XD "{payload_internal}"'),
        ]
    else:
        copies = [(str(src_root), install_dir, "/E")]   # unfamiliar layout: copy, but never purge

    # /R:/W: retry while a file is briefly still locked as we tear down. robocopy exit codes 0-7 are
    # degrees of success, >= 8 means a file genuinely failed (locked, or no write perms in eg Program
    # Files). cmd's `if errorlevel 8` is a >= test, so RC lands on 8 if any leg failed.
    copy_lines = "".join(
        f'robocopy "{s}" "{d}" {flags} /R:5 /W:2 /NP >> "{log}" 2>&1\r\n'
        "if errorlevel 8 set RC=8\r\n"
        for s, d, flags in copies
    )

    # the wait loop tries to open our own exe for append, which fails while we're still running, and
    # retries via ping. it deliberately uses NO console builtins: this script runs windowless, and
    # `tasklist | find` (pipe) and `timeout` (wants a console/stdin) both hang or die there.
    # on failure we still relaunch, but KEEP the staging dir so update.log survives for a bug report.
    # the (goto)+rmdir idiom lets cmd release the running .bat so the staging folder can delete itself.
    script = (
        "@echo off\r\n"
        "setlocal\r\n"
        ":wait\r\n"
        "ping 127.0.0.1 -n 2 >nul\r\n"
        f'2>nul (>>"{old_exe}" call ) || goto wait\r\n'
        "set RC=0\r\n"
        f"{copy_lines}"
        f'if exist "{new_exe}" (start "" "{new_exe}") else (start "" "{old_exe}")\r\n'
        "if %RC% GEQ 8 exit /b %RC%\r\n"
        f'(goto) 2>nul & rmdir /s /q "{updir}"\r\n'
    )
    # write_bytes, NOT write_text: text mode rewrites every \n to os.linesep, so our \r\n becomes
    # \r\r\n, cmd reads the label as ":wait\r", `goto wait` finds nothing, and the script aborts
    # before copying anything -- silently, since it has no console. encode as oem, the codepage cmd
    # reads a .bat in, so a non-ascii windows username in these paths survives.
    bat.write_bytes(script.encode("oem", errors="replace"))

    messagebox.showinfo(
        "installing update",
        "The app will now close and reopen to finish updating. This takes a few seconds, you don't "
        f"need to do anything.\n\n{ICONS_NOTE}")

    try:
        # CREATE_NO_WINDOW alone: it still gives cmd a (windowless) console, which console tools need.
        # DETACHED_PROCESS would leave it with NO console -- and it silently overrides CREATE_NO_WINDOW
        # when both are passed. the swapper outlives us either way; windows doesn't kill children.
        # std handles are DEVNULL because a windowed exe's own handles are invalid to inherit.
        subprocess.Popen(
            ["cmd", "/c", str(bat)],
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        messagebox.showerror("update failed", f"Could not launch the updater:\n{e}")
        return

    # quit cleanly so the swapper can overwrite the (now unlocked) exe. _on_close stops the loop.
    on_close = getattr(app, "_on_close", None)
    if callable(on_close):
        on_close()
    else:
        app.destroy()
