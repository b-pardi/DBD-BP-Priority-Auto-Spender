# dbd bloodweb priority auto-spender

windows/python tool that watches the dead by daylight bloodweb on screen and auto-buys
nodes in the order you care about. you give it a priority list (specific items like
"very rare flashlight", or whole categories like "any offering"); it scans the current
level, buys the highest-priority nodes available, and when none of your priorities are
present it clicks the center auto-spend to finish the level and move on.

> notice: dbd runs easy anti-cheat in menus too. automated input is a ban-risk you're choosing to take on. dry-run mode (no input sent) is the default for testing.
>
> note: run dbd in borderless (windowed fullscreen), not exclusive fullscreen. exclusive fullscreen returns black screen-captures, can swallow the synthesized clicks, and blocks the global kill-switch hotkey. borderless looks identical and makes capture, clicking, and the f7/f8 hotkeys all work.

## Status

#### Pre-Alpha (earliest possible working version tuned for one setup)

- [x] scaffold + requirements
- [x] scraper: pull all icons + metadata from deadbydaylight.wiki.gg
- [x] detect: localize nodes, read rarity, identify icon (offline, on fixtures)
- [x] match: match nodes with p-hashing/NCC
    - [x] OCR fall back method to detect difficult nodes
- [x] priority: rank detected nodes, pick the next buy or the auto-spend fallback
- [x] spender: live capture/click loop, ocr stop-threshold, kill switch
- [ ] ui: user interface for defining priorities and general program settings

---

#### Alpha (ready for small test group)
- [ ] make robust across resolutions
- [ ] detect: switch to custom trained CNN
    - [ ] data labelling util
    - [ ] train model
    - [ ] integrate model
- [ ] node detection/extraction v2
- [ ] auto determine bbox for bloodweb frame crop.
- [ ] (if needed) HSV calibration/fine tuning option for color masking

## setup

uses a conda env named `dbdbp-env`.

1. Create Virtual Environment:
- Python venv: `python -m pip venv dbdbp-env`
OR
- Anaconda: `conda activate dbdbp-env`

2. Install requirements: `pip install -r requirements.txt`

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

# test node detection on a screenshot of the dbd bloodweb (note detect.py has crop bounds that may need adjustment for now)
python -m src.detect detect path/to/screenshot.png

# view gallery image of all nodes detected and their predictions
python -m src.detect glpyhs path/to/screenshot.png

# run detections on simulated bloodweb levels (i.e. a 'perfect' bloodweb node detection)
python -m src.spender --sim

# EARLY VERSION MAY NOT WORK PROPERLY
# run the auto spender as functionally intended
# begins listening for start key (default F7) to start scanning and clicking nodes,
# and kill key (default F8) to cut the program
python -m src.spender --live 

# to run the spender without clicking anything as a test run
python -m src.spender --dry-run # note dry run is the default
```

## How It Works

### Detecting nodes on screen in the bloodweb

### Matching nodes with wiki references

### Priority Selection

### Screen Capturing / Input Control