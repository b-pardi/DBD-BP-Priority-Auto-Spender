# dbd bloodweb priority auto-spender

windows/python tool that watches the dead by daylight bloodweb on screen and auto-buys
nodes in the order you care about. you give it a priority list (specific items like
"very rare flashlight", or whole categories like "any offering"); it scans the current
level, buys the highest-priority nodes available, and when none of your priorities are
present it clicks the center auto-spend to finish the level and move on.

> notice: dbd runs easy anti-cheat in menus too. automated input is a ban-risk you're choosing to take on. dry-run mode (no input sent) is the default for testing.

## Status

early build. done and upcoming:

- [x] scaffold + requirements
- [x] scraper: pull all icons + metadata from deadbydaylight.wiki.gg
- [ ] detect: localize nodes, read rarity, identify icon (offline, on fixtures)
- [ ] priority: rank detected nodes, pick the next buy or the auto-spend fallback
- [ ] spender: live capture/click loop, ocr stop-threshold, kill switch
- [ ] ui: user interface for defining priorities and general program settings

## setup

uses a conda env named `dbdbp-env`.

```
conda activate dbdbp-env
pip install -r requirements.txt
```

the bp-counter ocr also needs the tesseract binary installed system-wide:
https://github.com/UB-Mannheim/tesseract/wiki (pytesseract just wraps it).

## layout

```
config/priority.json      your priority list (stand-in for the future selection ui)
data/icons/               scraped reference icons, one subfolder per category
data/icons_index.json     per-icon metadata: key, name, category, rarity, file, phash
src/scraper.py            pull icons + metadata from the wiki
src/capture.py            grab the bloodweb screen region via mss
src/detect.py             localize nodes, read rarity, identify icon
src/priority.py           load config, rank nodes, pick next buy / fallback
src/input_control.py      mouse move + click via pydirectinput
src/ocr.py                read the bp counter for the stop threshold
src/spender.py            main loop + dry-run + global-hotkey kill switch
tests/fixtures/           saved bloodweb screenshots for offline detection tests
```

## usage (in progress)

```
# build the icon library (run once, re-run after a dbd content patch)
python -m src.scraper
```
