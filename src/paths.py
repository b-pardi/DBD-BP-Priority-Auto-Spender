"""frozen-aware path resolution shared by the ui and the detection/spend modules.

one resolver so the ui, the `python -m src.spender` cli, and the eventual pyinstaller exe all agree
on where the config, caches, and debug output live. the rule is a split:
  dev (running from source): everything stays in the repo tree, exactly as before the ui existed.
  frozen (.exe): writable state moves to %APPDATA%/dbdbp since the bundle is read-only/temp.
read-only bundled assets (icons, the index) always resolve via resource_path, which knows about
pyinstaller's _MEIPASS unpack dir. lives in src/ (not ui/) because spender/detect need it and src
must not depend on the ui package; ui imports it via `from src import paths`.
"""

import os
import sys
from pathlib import Path

APP_DIR_NAME = "dbdbp"  # the per-user folder under %APPDATA% for the frozen exe

# repo root is one level above src/ (src/paths.py -> src -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def is_frozen():
    """true when running from a pyinstaller bundle (the .exe), false from source."""
    return getattr(sys, "frozen", False)


def resource_path(rel=""):
    """absolute path to a read-only bundled asset (data/icons, icons_index.json, default config).
    uses pyinstaller's _MEIPASS unpack dir when frozen, else the repo root, so detect/scraper
    resolve data/ the same way whether run from source or the exe."""
    base = Path(getattr(sys, "_MEIPASS", _REPO_ROOT)) if is_frozen() else _REPO_ROOT
    return (base / rel) if rel else base


def user_base():
    """writable per-user base dir for state the app writes (config, caches, debug output).
    frozen -> %APPDATA%/dbdbp (the bundle is read-only); dev -> the repo root, so source runs keep
    writing into the repo tree exactly as before."""
    if is_frozen():
        appdata = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(appdata) / APP_DIR_NAME
    return _REPO_ROOT


def config_path():
    """the single priority+settings config file.
    dev: repo config/priority.json; exe: %APPDATA%/dbdbp/config/priority.json."""
    return user_base() / "config" / "priority.json"


def cache_dir():
    """writable cache dir (ncc template .npy, ui thumbnails).
    dev: repo data/ (where the ncc cache already lives); exe: %APPDATA%/dbdbp/cache."""
    return user_base() / ("data" if not is_frozen() else "cache")


def debug_dir():
    """writable debug-output dir (detect's saved overlays / glyph crops).
    dev: repo .tmp/; exe: %APPDATA%/dbdbp/debug."""
    return user_base() / (".tmp" if not is_frozen() else "debug")


def ensure_user_dirs():
    """create the writable dirs and, on first frozen run, seed the default config from the bundle.
    returns the resolved config path. a no-op in dev where the repo dirs already exist."""
    for d in (config_path().parent, cache_dir(), debug_dir()):
        d.mkdir(parents=True, exist_ok=True)
    cfg = config_path()
    if not cfg.exists():
        default = resource_path("config/priority.json")
        if default.exists():
            cfg.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
    return cfg
