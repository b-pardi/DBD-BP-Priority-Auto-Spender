"""frozen-aware path resolution shared by the ui and the detection/spend modules.

one resolver so the ui, the `python -m src.spender` cli, and the eventual pyinstaller exe all agree
on where the config, caches, and debug output live. the rule is a split:
  dev (running from source): everything stays in the repo tree, exactly as before the ui existed.
  frozen (.exe): writable state moves to %APPDATA%/dbdbp-pas since the bundle is read-only/temp.
read-only bundled assets (icons, the index) always resolve via resource_path, which knows about
pyinstaller's _MEIPASS unpack dir. lives in src/ (not ui/) because spender/detect need it and src
must not depend on the ui package; ui imports it via `from src import paths`.
"""

import os
import sys
from pathlib import Path

APP_DIR_NAME = "dbdbp-pas"   # the per-user folder under %APPDATA% for the frozen exe
LEGACY_APP_DIR_NAME = "dbdbp"  # what <=0.1.0-alpha used; migrated once, see user_base

# repo root is one level above src/ (src/paths.py -> src -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _appdata():
    return Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))


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
    frozen -> %APPDATA%/dbdbp-pas (the bundle is read-only); dev -> the repo root, so source runs
    keep writing into the repo tree exactly as before.

    the folder was called "dbdbp" up to 0.1.0-alpha and is renamed once, in place, the first time a
    newer build runs. that folder is not scratch: it holds the user's config + priority profiles and
    the ~700MB icon library they waited minutes to scrape, so a plain rename of the constant would
    have silently reset every existing install. if the rename can't happen (another instance holding
    a handle, a permissions problem) we keep using the old folder rather than stranding them -- a
    stale name is harmless, a lost library is not."""
    if not is_frozen():
        return _REPO_ROOT
    new = _appdata() / APP_DIR_NAME
    if new.exists():
        return new                      # already migrated (or a fresh install): one stat, then done
    old = _appdata() / LEGACY_APP_DIR_NAME
    if old.exists():
        try:
            old.rename(new)
        except OSError:
            return old
    return new


def config_path():
    """the single priority+settings config file.
    dev: repo config/priority.json; exe: %APPDATA%/dbdbp-pas/config/priority.json."""
    return user_base() / "config" / "priority.json"


def cache_dir():
    """writable app-data root: the scraped index + sprites live here, and the disposable match
    caches live in the template_cache_dir() subfolder under it.
    dev: repo data/ (where the index/sprites already live); exe: %APPDATA%/dbdbp-pas/cache."""
    return user_base() / ("data" if not is_frozen() else "cache")


def template_cache_dir():
    """the ONLY-disposable cache: regenerable match templates (ncc .npy, cnn embed bank, ring
    template, eval renderbanks). kept in its own subfolder so the debug "cache" button can open/clear
    it without touching the sprites, the index, labels, or the trained model that share the data dir.
    dev: repo data/cache; exe: %APPDATA%/dbdbp-pas/cache/templates. everything here rebuilds on demand."""
    return cache_dir() / ("cache" if not is_frozen() else "templates")


def debug_dir():
    """writable debug-output dir (detect's saved overlays / glyph crops).
    dev: repo .tmp/; exe: %APPDATA%/dbdbp-pas/debug."""
    return user_base() / (".tmp" if not is_frozen() else "debug")


def ensure_user_dirs():
    """create the writable dirs and, on first frozen run, seed the default config from the bundle.
    returns the resolved config path. a no-op in dev where the repo dirs already exist."""
    # usr/ holds the evolving rarity-HSV anchors detect rewrites per web (detect.USR_HSV)
    for d in (config_path().parent, cache_dir(), template_cache_dir(), debug_dir(),
              user_base() / "usr"):
        d.mkdir(parents=True, exist_ok=True)
    # one-time migration: earlier builds wrote the match caches straight into the cache-dir root
    # (mixed in with the sprites + index). relocate any that are still there into the templates
    # subfolder so the writable data dir holds only keep-able assets. glob isn't recursive, so this
    # never re-scoops the files it just moved.
    tdir = template_cache_dir()
    for legacy in list(cache_dir().glob("*.npy")) + list(cache_dir().glob("*.npz")):
        try:
            legacy.replace(tdir / legacy.name)
        except OSError:
            pass
    cfg = config_path()
    if not cfg.exists():
        default = resource_path("config/priority.json")
        if default.exists():
            cfg.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
    return cfg
