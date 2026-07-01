"""thin config layer for the ui.

loads/saves the single priority+settings file through the same spender serializer the cli uses,
resolving the path via src.paths so the ui, the cli, and the exe never disagree on which file is
authoritative. all schema, validation, and v1->v2 migration logic stays in spender; this module is
just the path-resolution + first-run seeding glue.
"""

from src import paths, spender

DEFAULT_PROFILE = "Default"


def load():
    """load the config from the resolved path, seeding the default on first (frozen) run.
    lets exceptions propagate so the ui can surface them: ValueError on a malformed rule, and
    FileNotFoundError if the file is genuinely missing (the repo ships one, so in dev that is a
    real error to show, not swallow)."""
    cfg_path = paths.ensure_user_dirs()
    cfg = spender.load_config(cfg_path)
    ensure_profiles(cfg)
    return cfg


def ensure_profiles(cfg):
    """guarantee cfg has a `profiles` dict + `active_profile`, and keep top-level `priorities`
    mirroring the active profile.

    profiles are a ui-only concept (named priority lists for survivor / killer / per-killer). the
    engine and cli only ever read `priorities`, so we point that at the active profile and they stay
    oblivious. an older file with just a flat `priorities` is seeded into a single 'Default' profile.
    settings (hotkeys, matcher, dry-run, ...) stay global, outside profiles, by design."""
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        cfg["profiles"] = {DEFAULT_PROFILE: cfg.get("priorities", [])}
        cfg["active_profile"] = DEFAULT_PROFILE
    else:
        active = cfg.get("active_profile")
        if active not in profiles:
            active = next(iter(profiles))
        cfg["active_profile"] = active
        cfg["priorities"] = profiles[active]
    return cfg


def save(cfg):
    """validate + write the config to the resolved path, returning the path written.
    raises ValueError before writing if any rule is malformed, so we never persist a broken file."""
    return spender.save_config(cfg, paths.config_path())


def config_file():
    """the resolved config path, for display in the ui."""
    return paths.config_path()
