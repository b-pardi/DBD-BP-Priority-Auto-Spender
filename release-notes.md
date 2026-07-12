## v0.2.0-beta

Flushed out details from the alpha build of the DBD Bloodweb Priority Auto-Spender as a standalone Windows app. No Python install needed.

**To run:**
1. Download the zip folder below.
2. Extract the folder `dbdbp` anywhere on your computer. **DO NOT** Extract _only_ the exe, keep the .exe file and _internal folder together when running
3. Run `dbdbp.exe`
4. View the 'Instructions' tab in the UI to get started

- **See [README.md](https://github.com/b-pardi/DBD-BP-Priority-Auto-Spender/blob/main/README.md) for more details**

> notice: dbd runs easy anti-cheat in menus too. automated input is a ban-risk you're choosing to take on. dry-run mode (no input sent) is the default for testing.
>
> note: run dbd in borderless (windowed fullscreen), not exclusive fullscreen. exclusive fullscreen returns black screen-captures, can swallow the synthesized clicks, and blocks the global kill-switch hotkey. borderless looks identical and makes capture, clicking, and the f7/f8 hotkeys all work.

## Changes

- Bloodweb prestige screen detection/automation
- bloodpoint spend thresholding
- bw prestige level spend cap
- massive responsiveness improvements to ui
- ui instructions tab
- ui theme changes
- misc ui bug fixes fully flushing it out further (instructions tab, proper color design, accessibility options, bug fixes, etc.).
- check for updates/auto update feature