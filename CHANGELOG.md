## Changelog

### v0.3.3-beta
- fixed bug where distance right nodes not detected due to poor bloodweb crop bounds
- fixed bug where model kept classifying focus lens as saboteur
    - some scraper icon rows had duplicates and some had rarity none
    - led to retrained model on fixed glyph dataset
- brand new part now more consistently recognized
    - white part of glyph was much larger and hsv anchors fail
    - added an anchor match ranking to test candidate anchors closest to glyph
- fixed glyph extraction mis-crop that was the deeper focus lens -> saboteur cause
    - the addon '+' marker survived the brightness cut at the plate edge, stretched the glyph bbox and shoved faint icons off-center
    - real match accuracy 91.4% -> 94.8% (98.6% on independent labels) before any retrain
- synth training renders now mimic real nodes much closer:
    - semi-transparent art
    - plate brightness wobble
    - game-scale uncropped icons
    - slight art shift/zoom
    - (fix) '+' marker drawn on top
- new model diagnostics
- added a matcher veto to auto default to ocr for glyphs that have very similar pairs (e.g. murky reagent and clear reagent)
- removed a leftover duplicate icon row and fixed scraper so it don't do that shit again

### v0.3.2-beta
- fixed stale cache issue leading to over reliance on ocr fallback

### v0.3.1-beta
- CRITICAL: fixed ocr bloodpoint read bug on 16x9 monitors that prevented users from spending anything as the spender said there were insufficient funds

### v0.3.0-beta
- fixed wiki scraper to grab new chapter items handling weird wiki quirks with those
    - note also implicitly fixes `Toothy Torte` only showing as `10th anniversary`
- fixed text norm for perks with '&' or accents
- retrained model with fixed scraped icons and a few tweaks:
    - added embedding mining
    - added cnn fallback to ocr confidence threshold with a margin threshold
- detection of grabbed nodes (by entity and user/spender grabbed separately)
- added entity race mode, that uses priority list, but will prioritize nodes soon to be taken by the entity
- added default/template profiles for users to start with
- fixed the wiki's "Visceral" rarity tier reading as no rarity (iri add-ons like Gel Dressings showed gray/'any rarity'); maps to ultra rare
- scraping/index fixes now propagate into saved priority profiles
- spender now reads your bloodpoints every live run and auto-stops when it can't afford anything more
- start/pause hotkey (f7) now also STARTS a run when idle, so you can set up, tab into the game, and start from there
- new tunable detection knobs in settings: presence threshold, matcher rescue min score / margin
- library tooltips now show the item's actual gameplay/effect text (with per-tier numbers) under the lead sentence (~93% of the library, the rest keep the lead sentence)
- added a self-test suite for some of the core functionality and weakpoints of the pipeline
- a metric fuck ton of ui changes:
    - fixed some ui widget bboxes not being aligned properly
    - profiles group by survivor/killer in the picker via a per-profile side tag, so the list isn't stressful to look at
    - switching profiles no longer flags unsaved changes
    - killer/survivor library filter expanded to each individual killer
    - event / n/a reveal filters split into two checkboxes (event shown by default, n/a hidden)
    - draggable divider between the library and priority panes
    - ✕ button in the search box to clear it
    - drag-and-drop now shows a bright insertion bar exactly where the item will land
    - run buttons show their hotkeys, debug Save frame writes annotated + raw pngs, follow-on-github link on the run tab
    - update prompts now note that you may need to Update Icons as well
    - instructions: dialed-back anti-cheat note, full settings-knob documentation (settings tab points at it), save-your-profile and first-levels-warm-up notes
    - buttons and dropdowns sitting on an oxblood panel were the exact same color as it, so they had no visible edge and the tinted ones next to them (profile delete, the dropdown arrows) read as stray oversized blocks; controls are now a lighter step of the same red. affected the profile bar, the tier list's add-tier and rule builder, the debug maintenance panel and the settings save button
    - the profile bar and the rule builder gave their controls no vertical padding, so each bar was exactly as tall as the widgets in it and they sagged a pixel or two out the bottom of it; both bars now inset their contents
    - added a restore all defaults button to settings in ui
    - fixed up some instructions and settings tooltips
- added an FAQ to repo


### v0.2.0-beta
- Bloodweb prestige screen detection/automation
- bloodpoint spend thresholding
- bw prestige level spend cap
- massive responsiveness improvements to ui
- ui instructions tab
- ui theme changes
- misc ui bug fixes fully flushing it out further (instructions tab, proper color design, accessibility options, bug fixes, etc.).
- check for updates/auto update feature
- reduced package size from 860MB -> 340MB (301MB -> 135MB zipped)

### v0.1.0-alpha
- literally just spent bloodpoints idk really just a proof of concept see the readme for things it can do