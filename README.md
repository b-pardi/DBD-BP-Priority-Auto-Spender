# Dead By Daylight BloodWeb Priority Auto-Spender 
## DBD-BW-PAS

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
- [x] ui: user interface for defining priorities and general program settings
    - [x] add debugging view in ui

---

#### Alpha (ready for small test group)
- [x] make robust across resolutions
- [x] detect: switch to custom trained CNN
    - [x] data labelling util
    - [x] synthetic data data generation
    - [x] train/tune/integrate model
- [x] node detection/extraction v2
- [ ] bloodweb prestige screen detection
- [x] auto determine bbox for bloodweb frame crop.
- [x] fix non-breaking UI bugs
- [ ] add within tier priority selection
- [ ] core functionality test code on fixtures
- [ ] flush out documentation
    - [ ] add instructions tab to ui
- [ ] update default settings and setup profile templates
- [ ] bundle an exe

#### Beta (some smart nice to have features)
- [ ] Entity aware node selection
- [ ] bloodweb prestige/level spending cap
- [ ] bloodpoint spending limit
- [ ] validate abilities in other bloodweb menu backgrounds
- [ ] implement the synthetic node background images into the scraper
- [ ] update ui button to auto update the software in the ui
- [ ] auto pause when detecting bloodweb no longer visible
- [ ] ui drag and drop priority elements

## Setup

uses a conda env named `dbdbp`

Anaconda recommended:
`conda env create -f environment.yml`
`conda activate dbdbp`

One can use a python venv with version 3.11 and download all the pkgs in `environment.yml`, using special care when it comes to [pytesseroc](https://pypi.org/project/tesserocr/)

Note in either case, the Tesseract binary is _not_ required to be installed system wide, pytesseroc is cool and nice and comes with the tesseract C bins (hence the special attention to installing it for your specific windows platform).

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

**Ensure all CMDs are run from the project root dir (`<path/to/dbd_bp_prioritized_auto_spender/>`) and that the conda environment is activated (`conda activate dbdbp`)**
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

# to run the ui from python and not the exe
python -m ui
```

## How It Works

### Detecting nodes on screen in the bloodweb

- Grabs the screen, then auto crops down to just the bloodweb using a couple bits of OCR'd anchor text so UI buttons and other screen junk can't get mistaken for nodes.
- Binarizes the crop and finds node shaped blobs, then figures out each one's rarity from the color of its disk and its item/perk/addon type from the socket outline.
- Turns out every bloodweb is secretly the same fixed 30 slot ring layout no matter what's in it, so detection now fits that lattice to the frame and snaps candidates onto it. This single fact would have saved an embarrassing number of weekends fighting almost circular blobs that were actually just shadows.
- The center node gets found separately by its red glow, since it's the auto spend fallback and never something we buy on purpose.

### Matching nodes with wiki references 

- Crops out just the little icon glyph sitting inside each node, blurry JPEG artifacts and all.
- The real matcher is a small CNN that learns to squish that ugly extracted glyph and the clean wiki sprite into the same neighborhood of embedding space, then it's just nearest neighbor lookup against a cached bank of every icon's embedding.
- Classical approaches (pHash, plain NCC, masked NCC) all got tried first and all capped out around the mid 50s to low 60s percent on real screenshots, which is a nice way of saying they weren't good enough and something smarter was unavoidable.
- Training that CNN needed a pile of labeled examples nobody was going to hand label, so there's a synthetic glyph generator that renders fake nodes and puts them through the exact same crop and degrade pipeline the live detector uses, just to manufacture something realistic enough to learn from. Genuinely one of the more soul crushing parts of this whole project.
- When the matcher still isn't confident enough, it falls back to hovering the node and OCR'ing the tooltip that pops up. Slower, but basically never wrong.

#### Getting the wiki references

- Scrapes deadbydaylight.wiki.gg for every icon plus its name, rarity, category, and description.
- Also works out whether an icon is currently obtainable and whether it belongs to the Survivor or Killer side, so matching can skip comparing against icons that could never show up in the web you're actually looking at.
- Rerun it after a content patch and the whole library just refreshes itself, no manual relabeling required.

### Priority Selection

- You write out tiers of rules in a JSON config, from specific items at a given rarity down to whole categories like "any offering".
- Each scan walks the list top to bottom and buys the first match it finds among the detected nodes, one buy per scan since DBD auto pathfinds to whatever you click anyway.
- Once nothing on the list is present anymore it just hits the center auto spend and lets the level finish itself out.

### Screen Capturing / Input Control

- MSS grabs the frames, mostly because it's a lot faster than PIL on Windows and speed matters when you're polling a game window.
- pydirectinput drives the mouse so the game sees ordinary input rather than something obviously synthetic.
- The cursor gets parked off in a corner between actions so it isn't just sitting on a node accidentally triggering a tooltip every time a scan runs.
- A global hotkey kill switch cuts everything immediately, because sooner or later something will misbehave and you'll want a way out that doesn't involve alt tabbing in a panic.

### Program Interface

idk nothing special here it's a customtkinter python ui that I couldn't be fucked to make so claude did most and I just fixed its stuff.

- Lets you build and reorder your priority tiers, tune all the detection and timing knobs, and pick a matcher without touching the JSON by hand.
- A debug screen shows you what the detector's actually seeing frame by frame, with a zoom and a save button for whenever it inevitably sees something wrong and you need to know why.