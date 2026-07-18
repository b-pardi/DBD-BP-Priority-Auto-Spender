"""headless build/health self-test: does this build actually work end to end?

one importable engine (run_all) surfaced two ways: a "Test build" button on the run screen and a
"Self-test" group in the debug view, both when debugging is on. also runnable headless as
`python -m src.selftest` (exit code non-zero on any failure) so it can gate a release from source.

two tiers of checks:
  environment  the things that break when you FREEZE a build and nothing else: writable dirs under
               %APPDATA%, the config seeding, the wiki icon index, tesseract's dlls + traineddata,
               the onnx model bundling, the match-bank build. all fixture-free, so they run anywhere.
  functional   detection + ocr on a handful of curated, ground-truthed frames bundled in
               data/selftest (q90 jpgs), plus a bounded offline run through the real spender loop.
               skipped with a note if the fixtures or icon library aren't present.

imports only src (never ui) so it stays usable from the cli and inside the frozen exe.
"""

import io
import threading
import time
import traceback
from collections import namedtuple
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import detect, ocr, ocr_runtime, paths, spender, version
from .resolution import Resolution

# a check outcome. status is one of pass|warn|fail|skip; detail is a one-line human string.
#   pass  the thing works.
#   warn  not broken by the build, but the machine isn't fully set up (e.g. no icon library yet).
#   fail  genuinely broken; a release should not ship and a bug report should include it.
#   skip  couldn't run this check here (missing prerequisite it can't be blamed for).
Result = namedtuple("Result", "name status detail")

# curated ground-truth fixtures (data/selftest, bundled). coords are in the web_bbox crop, matching
# eval_matchers.GOLD; levels self-encode in the filename; the reads were confirmed on the q90 jpgs.
GOLD = {
    "web-005122.jpg": {(373, 327): "banquetMedKit", (757, 632): "banquetToolbox",
                       (289, 638): "banquetFlashlight", (470, 639): "10thAnniversary"},
    "web-005135.jpg": {(610, 396): "banquetFlashlight", (471, 474): "10thAnniversary"},
}
LEVELS = {"web-L1.jpg": 1, "web-L30.jpg": 30, "web-L50.jpg": 50}
PRESTIGE_READY = "web-prestige-ready.jpg"      # prestige star showing; read_prestige_level reads a number
PRESTIGE_CONFIRM = "web-prestige-confirm.jpg"  # post-click screen; find_ok_button locates the OK button
# a real 2560x1440 (16:9) grab: the aspect that broke the edge-anchored top-bar reads (fraction crops
# read only the trailing bp digits + a bogus 4, wrong level, prestige 0). ground truth off the frame.
ASPECT_FIXTURE = "web-16x9.jpg"
ASPECT_TRUTH = {"bp": 414162, "level": 50, "prestige": 100}


def fixtures_dir():
    """the bundled curated-fixture dir (resource_path, so _MEIPASS when frozen)."""
    return paths.resource_path("data/selftest")


# ------------------------------------------------------------------- environment tier

def check_version():
    v = version.__version__
    if not isinstance(v, str) or not v.strip():
        return "fail", "version.__version__ is empty"
    # the updater compares this against a github tag, so it must at least start with a number.
    head = v.lstrip("v").split(".")[0]
    if not head.isdigit():
        return "fail", f"version {v!r} doesn't start with a number (updater compares it to a tag)"
    return "pass", v


def check_paths():
    """the freeze-critical one: ensure_user_dirs must create every writable dir and seed the config.
    a frozen bundle is read-only, so anything that writes into it instead of %APPDATA% breaks here."""
    cfg_path = paths.ensure_user_dirs()
    dirs = {
        "config": cfg_path.parent, "cache": paths.cache_dir(),
        "templates": paths.template_cache_dir(), "debug": paths.debug_dir(),
        "usr": paths.user_base() / "usr",
    }
    bad = []
    for name, d in dirs.items():
        if not d.is_dir():
            bad.append(f"{name} missing")
            continue
        try:                                     # prove it's actually writable, don't just stat it
            probe = d / ".selftest_write"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            bad.append(f"{name} not writable ({e.__class__.__name__})")
    if bad:
        return "fail", "; ".join(bad)
    seeded = "config seeded" if cfg_path.is_file() else "config absent (will seed on first load)"
    return "pass", f"{len(dirs)} dirs writable, {seeded}"


def check_config():
    """load the config through the real serializer and round-trip it to a scratch file, so a schema
    or migration bug shows up before it corrupts the user's actual config (we never write theirs)."""
    cfg = spender.load_config(paths.config_path())
    tmp = paths.debug_dir() / ".selftest_config.json"
    spender.save_config(cfg, tmp)
    reloaded = spender.load_config(tmp)
    try:
        tmp.unlink()
    except OSError:
        pass
    n0, n1 = len(cfg.get("priorities", [])), len(reloaded.get("priorities", []))
    if n0 != n1:
        return "fail", f"round-trip changed tier count {n0} -> {n1}"
    return "pass", f"loads + round-trips ({n0} tiers, matcher={cfg.get('matcher', 'cnn')})"


def check_library():
    rows = detect.load_rows()
    if not rows:
        return "warn", "icon library empty — run Update icons (scraper) before matching"
    return "pass", f"{len(rows)} icons indexed"


def check_sprites():
    """a sample of index rows must resolve to a sprite on disk (the scrape actually downloaded them)."""
    rows = detect.load_rows()
    if not rows:
        return "skip", "no icon library yet"
    base = Path(detect.DEFAULT_INDEX).parent
    sample = rows[:: max(1, len(rows) // 20)]     # ~20 evenly-spaced rows
    missing = [r.get("file") for r in sample if not (base / (r.get("file") or "")).is_file()]
    if missing:
        return "fail", f"{len(missing)}/{len(sample)} sampled sprites missing (e.g. {missing[0]})"
    return "pass", f"{len(sample)} sampled sprites present"


def check_tesseract():
    """import tesserocr with its dlls wired (the #1 frozen breakage) and actually OCR a synthetic
    number, so we catch a broken traineddata path too, not just a clean import."""
    tess = ocr_runtime.get_tesserocr()
    # big white "42" on black via cv2 (scalable, no font-file dependency), like the game text we read.
    img = np.zeros((100, 220), np.uint8)
    cv2.putText(img, "42", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6, cv2.LINE_AA)
    api = ocr._api(tess.PSM.SINGLE_LINE, whitelist="0123456789")
    api.SetImage(Image.fromarray(img))
    txt = (api.GetUTF8Text() or "").strip()
    if "42" not in txt:
        return "fail", f"tesseract loaded but misread synthetic '42' as {txt!r}"
    return "pass", f"dlls + traineddata ok (read {txt!r})"


def check_cnn_model():
    """the onnx encoder must be bundled and loadable by cv2.dnn, and a forward pass must return a
    128-d embedding. absent isn't fatal (detect degrades to ncc), so that's a warn."""
    net = detect.load_cnn_model()
    if net is None:
        return "warn", "no onnx model bundled — matcher degrades to ncc (weaker)"
    blob = detect._cnn_blob(np.zeros((96, 96, 3), np.uint8))
    net.setInput(blob)
    out = net.forward().reshape(-1)
    if out.shape[0] != 128:
        return "fail", f"encoder output is {out.shape[0]}-d, expected 128"
    return "pass", "onnx loads + forwards a 128-d embedding"


def check_match_bank():
    """building the match bank exercises the cache-write path (read-only cache dir is a frozen risk)
    and confirms every row gets a template/embedding aligned to it."""
    rows = detect.load_rows()
    if not rows:
        return "skip", "no icon library yet"
    if detect.load_cnn_model() is not None:
        bank = detect.load_cnn_bank(rows)
        kind, n = "cnn embed", bank.shape[0]
    else:
        T, _ = detect.load_ncc_templates(rows)
        kind, n = "ncc", T.shape[0]
    if n != len(rows):
        return "fail", f"{kind} bank has {n} rows, expected {len(rows)}"
    return "pass", f"{kind} bank built ({n} rows)"


def check_bank_integrity():
    """stale-bank guard: sampled bank rows must match a fresh embed of their own sprite.
    catches a bank built against different rows/model/sprites even when shape and mtime look fine."""
    rows = detect.load_rows()
    if not rows:
        return "skip", "no icon library yet"
    net = detect.load_cnn_model()
    if net is None:
        return "skip", "no onnx model (ncc fallback has no embed bank)"
    bank = detect.load_cnn_bank(rows)
    base = Path(detect.DEFAULT_INDEX).parent
    worst, worst_key, n = 1.0, None, 0
    for i in range(0, len(rows), max(1, len(rows) // 8)):     # ~8 evenly-spaced rows
        p = base / (rows[i].get("file") or "")
        if not p.is_file():
            continue                                          # check_sprites owns missing files
        cos = float(bank[i] @ detect._cnn_embed(net, detect._sprite_glyph_color(p)))
        n += 1
        if cos < worst:
            worst, worst_key = cos, rows[i]["key"]
    if n == 0:
        return "skip", "no sampled sprites on disk"
    if worst < 0.999:
        return "fail", (f"bank row for {worst_key!r} diverges from a fresh embed (cos {worst:.3f})"
                        " — stale or misaligned bank cache, refresh the icon library")
    return "pass", f"{n} sampled bank rows match fresh embeds (min cos {worst:.4f})"


def check_library_hygiene():
    """duplicate-identity guard: no two matchable anchors should embed nearly identically.
    twin rows sit at cosine 0.95+, closest legit pair sits ~0.74, so 0.95 splits them cleanly.
    a warn means the library grew a duplicate the demote rules don't cover yet."""
    rows = detect.load_rows()
    if not rows:
        return "skip", "no icon library yet"
    net = detect.load_cnn_model()
    if net is None:
        return "skip", "no onnx model (anchor similarity needs the embed bank)"
    keep = [i for i, r in enumerate(rows) if detect.is_matchable(r)]
    B = detect.load_cnn_bank(rows)[keep]
    S = B @ B.T
    np.fill_diagonal(S, -2.0)
    i, j = np.unravel_index(int(np.argmax(S)), S.shape)
    top = float(S[i, j])
    detail = (f"{len(keep)}/{len(rows)} rows matchable; closest anchor pair "
              f"{rows[keep[i]]['key']} <-> {rows[keep[j]]['key']} at cos {top:.3f}")
    if top > 0.95:
        return "warn", detail + " — duplicate rows? refresh the icon library"
    return "pass", detail


def check_resolution_anchoring():
    """fixture-free guard on the fix for the 16:9 bp bug: the top-bar reads must be EDGE-anchored, not
    width fractions. across two widths at one height, bp must stay a fixed px from the RIGHT edge and
    level/prestige a fixed px from the LEFT, with every crop in bounds. a revert to width fractions
    (the bug) drifts the offsets and fails here without needing a capture."""
    base = Resolution(w=3440, h=1440)
    wide = Resolution(w=2560, h=1440)   # same height, 16:9: the aspect that broke the reads
    bad = []
    for r in (base, wide):
        b = r.bp_region_px()
        if not (0 <= b['x0'] < b['x1'] <= r.w and 0 <= b['y0'] < b['y1'] <= r.h):
            bad.append(f"bp crop out of bounds at w={r.w}")
    if (base.w - base.bp_region_px()['x1']) != (wide.w - wide.bp_region_px()['x1']):
        bad.append("bp drifts with width (not right-anchored)")
    for name, meth in (("level", "level_region_px"), ("prestige", "prestige_crest_region_px")):
        b0, b1 = getattr(base, meth)(), getattr(wide, meth)()
        if (b0['x0'], b0['x1']) != (b1['x0'], b1['x1']):
            bad.append(f"{name} drifts with width (not left-anchored)")
        if not (0 <= b1['x0'] < b1['x1'] <= wide.w):
            bad.append(f"{name} crop out of bounds at 16:9")
    if bad:
        return "fail", "; ".join(bad)
    return "pass", "bp right-anchored, level/prestige left-anchored across 3440/2560 @1440"


# ------------------------------------------------------------------- functional tier

def _have_fixtures():
    d = fixtures_dir()
    return d.is_dir() and (d / "web-005122.jpg").is_file()


def check_detection():
    """run the real cnn detect pipeline on the curated gold frames and confirm each known node still
    matches its expected library key. the core detection+matching signal."""
    if not detect.load_rows():
        return "skip", "no icon library yet"
    if not _have_fixtures():
        return "skip", "curated fixtures not bundled (dev-only from source)"
    have_model = detect.load_cnn_model() is not None
    web = Resolution().web_bbox_fallback_px()
    correct = total = 0
    misses = []
    for fx, exp in GOLD.items():
        frame = cv2.imread(str(fixtures_dir() / fx))
        if frame is None:
            return "fail", f"couldn't read fixture {fx}"
        frame = frame[web["y0"]:web["yf"], web["x0"]:web["xf"]]
        nodes = detect.detect(frame, matcher="cnn")
        for (x, y), want in exp.items():
            total += 1
            nd = min(nodes, key=lambda n: (n["x"] - x) ** 2 + (n["y"] - y) ** 2)
            m = nd.get("match")
            key = m.get("key") if isinstance(m, dict) else None
            if key == want:
                correct += 1
            else:
                misses.append(f"{want}->{key}")
    if correct == total:
        return "pass", f"{correct}/{total} gold nodes matched"
    detail = f"{correct}/{total} gold nodes matched ({', '.join(misses)})"
    # under the ncc fallback a miss is expected-ish (weaker matcher), so don't fail the build for it.
    return ("fail" if have_model else "warn"), detail


def check_ocr_levels():
    if not _have_fixtures():
        return "skip", "curated fixtures not bundled (dev-only from source)"
    ok = []
    bad = []
    for fx, want in LEVELS.items():
        img = cv2.imread(str(fixtures_dir() / fx))
        got = ocr.read_bloodweb_level(img, resolution=Resolution.from_frame(img))
        (ok if got == want else bad).append(f"{fx.split('.')[0]}:{got}!={want}" if got != want else fx)
    if bad:
        return "fail", f"{len(ok)}/{len(LEVELS)} level reads correct; wrong: {', '.join(bad)}"
    return "pass", f"{len(ok)}/{len(LEVELS)} bloodweb level reads correct"


def check_ocr_prestige():
    """the prestige-flow reads: the star's prestige number reads (not None), and the OK button on the
    post-click screen is located. both gate the auto-prestige feature."""
    if not _have_fixtures():
        return "skip", "curated fixtures not bundled (dev-only from source)"
    img = cv2.imread(str(fixtures_dir() / PRESTIGE_READY))
    p = ocr.read_prestige_level(img, resolution=Resolution.from_frame(img))
    img2 = cv2.imread(str(fixtures_dir() / PRESTIGE_CONFIRM))
    ok = ocr.find_ok_button(img2, resolution=Resolution.from_frame(img2))
    if p is None:
        return "fail", "couldn't read the prestige number on the ready screen"
    if ok is None:
        return "fail", "couldn't locate the OK button on the confirm screen"
    return "pass", f"prestige read ({p}), OK button located @ {ok}"


def check_ocr_bp():
    if not _have_fixtures():
        return "skip", "curated fixtures not bundled (dev-only from source)"
    img = cv2.imread(str(fixtures_dir() / "web-005122.jpg"))
    bp = ocr.read_bp(img, resolution=Resolution.from_frame(img))
    if not isinstance(bp, int) or bp <= 0:
        return "fail", f"bloodpoint read returned {bp!r} (expected a positive int)"
    return "pass", f"bloodpoint read = {bp:,}"


def check_ocr_aspect():
    """the regression test for the 16:9 bp-misread: bp/level/prestige off a real 2560x1440 grab vs
    ground truth. before the edge-anchored fix this read only the trailing bp digits (+ a bogus 4), a
    wrong level, and prestige 0; complements check_resolution_anchoring (which only proves the crops
    are edge-anchored, not that they land on the digits)."""
    if not _have_fixtures():
        return "skip", "curated fixtures not bundled (dev-only from source)"
    img = cv2.imread(str(fixtures_dir() / ASPECT_FIXTURE))
    if img is None:
        return "skip", f"{ASPECT_FIXTURE} not bundled"
    res = Resolution.from_frame(img)
    got = {"bp": ocr.read_bp(img, res), "level": ocr.read_bloodweb_level(img, res),
           "prestige": ocr.read_prestige_level(img, res)}
    bad = [f"{k} {got[k]}!={v}" for k, v in ASPECT_TRUTH.items() if got[k] != v]
    if bad:
        return "fail", f"16:9 (2560x1440) reads wrong: {', '.join(bad)}"
    return "pass", f"16:9 reads correct: bp={got['bp']:,} level={got['level']} prestige={got['prestige']}"


def check_sim_run():
    """drive the REAL spender loop over the offline simulator: a forced node the priorities pick,
    zeroed timing, dry-run (no clicks), bounded to a few levels. proves source -> decide -> buy ->
    consume -> advance works end to end with no game and no screen access."""
    rows = detect.load_rows()
    if not rows:
        return "skip", "no icon library yet"
    # a real row that presents a node, forced onto every web, with an item rule that picks it.
    forceable = next((r for r in rows if r.get("category") in _sim_node_categories()), None)
    if forceable is None:
        return "skip", "no node-shaped rows in the library"
    key, name = forceable["key"], forceable["name"]

    src = spender.sim_source(rows, seed=0, n=12, force=(key,))
    levels = {"n": 0}
    switch = spender.Switch()                 # NOT armed: no global hotkey registration for a self-test
    switch.reset()
    switch.toggle()                           # idle -> running
    orig_advance = src.advance

    def advance():                            # stop after a few levels so the infinite sim terminates
        levels["n"] += 1
        if levels["n"] >= 3:
            switch.kill()
        orig_advance()
    src.advance = advance

    config = {
        "priorities": [{"rules": [{"type": "item", "name": name}], "ordered": False}],
        "settle_s": 0, "advance_s": 0, "entity_settle_s": 0,   # rip through, no real waits
    }
    buf = io.StringIO()
    err = {"exc": None}

    def worker():
        try:
            with redirect_stdout(buf):        # swallow the loop's [dry-run] prints, count them after
                spender.run(src, config, switch, rows, click=False, debug=False)
        except Exception as e:                # noqa: BLE001 - report any loop crash as the failure
            err["exc"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=8.0)
    switch.kill()                             # backstop in case it didn't terminate on its own
    if t.is_alive():
        return "fail", "sim run didn't terminate within 8s"
    if err["exc"] is not None:
        return "fail", f"loop raised {type(err['exc']).__name__}: {err['exc']}"
    buys = sum(1 for line in buf.getvalue().splitlines() if "dry-run" in line.lower())
    if levels["n"] < 1:
        return "fail", "loop never completed a level"
    return "pass", f"ran {levels['n']} levels, {buys} dry-run buys, no errors"


def _sim_node_categories():
    from .sim import CATEGORY_SHAPE
    return set(CATEGORY_SHAPE)


# ------------------------------------------------------------------- driver

# (label, fn) in run order. environment first (fast, fixture-free), then functional.
CHECKS = [
    ("version", check_version),
    ("writable dirs", check_paths),
    ("config round-trip", check_config),
    ("icon library", check_library),
    ("sprites", check_sprites),
    ("tesseract ocr", check_tesseract),
    ("cnn model", check_cnn_model),
    ("match bank", check_match_bank),
    ("bank integrity", check_bank_integrity),
    ("library hygiene", check_library_hygiene),
    ("resolution anchoring", check_resolution_anchoring),
    ("detection (gold)", check_detection),
    ("ocr: levels", check_ocr_levels),
    ("ocr: prestige", check_ocr_prestige),
    ("ocr: bloodpoints", check_ocr_bp),
    ("ocr: 16:9 reads", check_ocr_aspect),
    ("offline sim run", check_sim_run),
]


def run_all(progress=None):
    """run every check in order, returning [Result]. progress, if given, is called with each Result
    as it completes so a ui can stream the run (each check can take a moment to warm caches)."""
    results = []
    for name, fn in CHECKS:
        try:
            out = fn()
            res = out if isinstance(out, Result) else Result(name, out[0], out[1])
        except Exception as e:  # noqa: BLE001 - a crashing check is itself a failure to report
            res = Result(name, "fail", f"{type(e).__name__}: {e}")
            if progress is None:                 # headless: keep the traceback for the console
                traceback.print_exc()
        results.append(res)
        if progress is not None:
            progress(res)
    return results


def summary(results):
    """(passed, warned, failed, skipped) counts over a result list."""
    c = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for r in results:
        c[r.status] = c.get(r.status, 0) + 1
    return c["pass"], c["warn"], c["fail"], c["skip"]


def format_line(res):
    """one aligned console/log line for a Result."""
    tag = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}[res.status]
    return f"[{tag}] {res.name:20s} {res.detail}"


def main():
    print(f"dbdbp-pas self-test  (version {version.__version__}, "
          f"{'frozen' if paths.is_frozen() else 'source'})\n")
    results = run_all(progress=lambda r: print(format_line(r)))
    p, w, f, s = summary(results)
    print(f"\n{p} passed, {w} warnings, {f} failed, {s} skipped")
    return 1 if f else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
