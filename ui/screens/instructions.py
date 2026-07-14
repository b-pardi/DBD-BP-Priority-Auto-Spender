"""instructions / help screen: a scrollable, formatted walkthrough of the whole app.

read-only, no config touched. it's just typography: a CTkScrollableFrame holding numbered step
sections, callout boxes (danger / warn / tip / notice), a hotkey table, a troubleshooting +
debug-reporting block, and a clickable github-issues link at the bottom. the boxes above step 1
run anti-cheat note -> display-mode warn -> attribution, most-to-least actionable.

wrapping labels don't know their width until the frame is laid out, so every long label is registered
in self._wrap_labels and its wraplength is refreshed from the scrollable frame's width on <Configure>.
every label here is height=1 + anchor="nw" so it hugs its text and tops out where the reader expects:
a CTkLabel otherwise reserves ~28px of canvas per line and centers the text inside it, which is what
left the bullet dots floating above their copy.
"""

import webbrowser

import customtkinter as ctk

from src import spender
from .. import theme

# callout tints: a hue wash over the panel tone, so a box reads by its strip rather than by shouting.
# each one has to sit DARKER than theme.BLOOD, because that oxblood is the field these boxes lie on
# (CTkFrame.top_fg_color) -- a callout lighter than the page would float instead of recess, and the
# numbered step cards next to them are darker for exactly the same reason.
CALLOUTS = {
    "danger": {"bg": "#251316", "accent": theme.DANGER, "icon": "⚠"},
    "warn":   {"bg": "#221c11", "accent": theme.ACCENT_BRIGHT, "icon": "!"},
    "tip":    {"bg": "#161c1f", "accent": "#7d8c94", "icon": "★"},   # fog: quieter than the accent
    "notice": {"bg": "#1e1a20", "accent": "#7a6f86", "icon": "©"},   # attribution, dimmest of all
}

GITHUB_ISSUES_URL = "https://github.com/b-pardi/DBD-BP-Priority-Auto-Spender/issues"

# a slightly larger heading than FONT_TITLE for the page title, and a section-header size.
FONT_PAGE = ("Segoe UI", 24, "bold")
FONT_SECTION = ("Segoe UI", 16, "bold")
FONT_LEAD = ("Segoe UI", 13)
FONT_CALLOUT_TITLE = ("Segoe UI", 13, "bold")  # compact so a callout hugs its text

CONTENT_MAX = 820  # cap the readable column so lines don't run edge-to-edge on a wide window


class InstructionsScreen(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._wrap_labels = []  # (label, side_padding) refreshed on resize
        self._last_wrap_w = None   # last width actually applied, so a no-op resize does nothing
        self._pending_w = None     # width awaiting the debounced reflow
        self._reflow_job = None     # pending after() id, so a resize burst collapses to one reflow

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)

        # a centered column of fixed max width, so the text stays comfortably readable while the
        # window itself can be any width.
        self.col = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.col.grid(row=0, column=0, sticky="n", pady=theme.PAD)
        self.col.grid_columnconfigure(0, weight=1)

        self._build()
        # add="+", NOT a plain bind: CTkScrollableFrame binds its OWN <Configure> on this same inner
        # frame to keep the canvas scrollregion synced to the content bbox (ctk_scrollable_frame.py).
        # a plain bind here replaces that handler, so the scrollregion freezes and the scrollbar stops
        # tracking the content. chaining with add="+" runs both.
        self.scroll.bind("<Configure>", self._on_resize, add="+")

    # layout helpers
    def _on_resize(self, event):
        # keep the reading column capped and reflow every wrapping label to the column width, but do
        # it OFF the event via after(): reconfiguring col + wraplengths itself emits <Configure>s, so
        # running the reflow synchronously here re-fires the handler on its own output. worse, any
        # update_idletasks during startup (e.g. a sibling scrollbar's draw) pumps that synchronously
        # into an infinite loop that hangs the app before the window shows. deferring means the burst
        # of construction-time Configures only ever reschedules a single reflow, which then runs once
        # in the mainloop and converges (the width guard stops it re-triggering).
        width = min(event.width - 2 * theme.PAD, CONTENT_MAX)
        width = max(width, 320)
        if width == self._last_wrap_w:
            return
        self._pending_w = width
        if self._reflow_job is not None:
            self.after_cancel(self._reflow_job)
        self._reflow_job = self.after(40, self._reflow)

    def _reflow(self):
        self._reflow_job = None
        width = self._pending_w
        if width is None or width == self._last_wrap_w:
            return
        self._last_wrap_w = width
        self.col.configure(width=width)
        for label, pad in self._wrap_labels:
            label.configure(wraplength=width - pad)

    def _para(self, parent, text, pad=2 * theme.PAD, font=theme.FONT_BODY, color=None):
        """a left-justified wrapping paragraph, registered for reflow.

        height=1 so the label hugs its text. left alone a CTkLabel reserves ~28px of canvas even for
        one line, which quietly padded out every paragraph on the page; it still grows to fit however
        many lines the text wraps to."""
        kw = {"font": font, "justify": "left", "anchor": "nw", "height": 1,
              "wraplength": CONTENT_MAX}
        if color is not None:
            kw["text_color"] = color
        lbl = ctk.CTkLabel(parent, text=text, **kw)
        lbl.pack(fill="x", anchor="w", pady=(0, theme.PAD))
        self._wrap_labels.append((lbl, pad))
        return lbl

    def _bullet(self, parent, text):
        """a hanging-indent bullet row (dot column + wrapping text).

        both labels are height=1 + anchor="nw" and both are packed anchor="n", which is what makes the
        dot sit on the *first line* of the text. left alone a CTkLabel is ~28px tall and centers its
        text in that box, so the dot (anchored n, at the top of its own box) floated above a
        single-line bullet, and well above the vertically-centered block of a wrapped one."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", anchor="w", pady=(0, 4))
        ctk.CTkLabel(row, text="•", font=theme.FONT_BODY, width=16, height=1, anchor="nw").pack(
            side="left", anchor="n")
        lbl = ctk.CTkLabel(row, text=text, font=theme.FONT_BODY, justify="left", anchor="nw",
                           height=1, wraplength=CONTENT_MAX)
        lbl.pack(side="left", fill="x", expand=True, anchor="n")
        # dot column (16) + the two frame pads + a little breathing room
        self._wrap_labels.append((lbl, 16 + 2 * theme.PAD + 8))

    def _section(self, number, title):
        """a numbered step card, returns the inner body frame to pack content into."""
        card = ctk.CTkFrame(self.col)
        card.pack(fill="x", pady=(0, theme.PAD))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=theme.PAD, pady=(theme.PAD, 4))
        if number is not None:
            ctk.CTkLabel(head, text=str(number), font=theme.FONT_TITLE, width=34, height=34,
                         fg_color=theme.ACCENT, corner_radius=17,
                         text_color=theme.BONE).pack(side="left", padx=(0, theme.PAD))
        ctk.CTkLabel(head, text=title, font=FONT_SECTION, anchor="w").pack(
            side="left", fill="x", expand=True)
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=theme.PAD + (46 if number is not None else 0),
                  pady=(0, theme.PAD))
        return body

    def _callout(self, kind, title, text):
        """a tinted note box with a colored accent strip: danger / warn / tip / attribution."""
        style = CALLOUTS[kind]
        wrap = ctk.CTkFrame(self.col, fg_color=style["bg"], corner_radius=8)
        wrap.pack(fill="x", pady=(0, theme.PAD))
        # height=1 on the strip is load-bearing, not a nicety. it has no pack/grid children of its
        # own (a CTkFrame's canvas is place'd), so tk never sizes it from its contents and it keeps
        # its *requested* height -- and ctk's default request is 200x200. that alone was propping
        # every callout open to 229px (dpi-scaled) around ~60px of text. asking for 1px and letting
        # fill="y" stretch it lets the box hug its copy, like the numbered step cards do.
        strip = ctk.CTkFrame(wrap, width=theme.ACCENT_W, height=1, fg_color=style["accent"],
                             corner_radius=0)
        strip.pack(side="left", fill="y")
        inner = ctk.CTkFrame(wrap, fg_color="transparent")
        # theme.PAD top/bottom == the step cards' internal padding, so a callout and a step read as
        # the same kind of block.
        inner.pack(side="left", fill="x", expand=True, padx=theme.PAD, pady=theme.PAD)
        # height=1 lets each CTkLabel shrink to its text instead of reserving the default 28px canvas
        # floor; the frame still grows to fit wrapped lines, so the box hugs the copy either way.
        ctk.CTkLabel(inner, text=f"{style['icon']}  {title}", font=FONT_CALLOUT_TITLE,
                     anchor="w", height=1).pack(fill="x", anchor="w", pady=(0, 2))
        lbl = ctk.CTkLabel(inner, text=text, font=theme.FONT_BODY, justify="left", anchor="nw",
                           height=1, wraplength=CONTENT_MAX)
        lbl.pack(fill="x", anchor="w")
        self._wrap_labels.append((lbl, theme.ACCENT_W + 3 * theme.PAD))

    def _subhead(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=theme.FONT_TITLE, anchor="w").pack(
            fill="x", anchor="w", pady=(theme.PAD, 4))

    # content
    def _build(self):
        cfg = self.app.app_state.config or {}
        start_key = cfg.get("start_key", spender.START_KEY)
        kill_key = cfg.get("kill_key", spender.KILL_KEY)

        # page header
        header = ctk.CTkFrame(self.col, fg_color="transparent")
        header.pack(fill="x", pady=(0, theme.PAD))
        ctk.CTkLabel(header, text="How to use dbdbp-pas", font=FONT_PAGE, anchor="w").pack(
            fill="x", anchor="w")
        self._para(
            header,
            "This tool watches the Dead by Daylight bloodweb on your screen and buys nodes in the "
            "order you care about. You give it a priority list, it scans each level, buys your "
            "highest-priority nodes, and once nothing on your list is left it clicks the center to "
            "auto-spend the rest and move on. This page walks through setting it up, from a first "
            "launch to a live run, plus how to report a problem if something looks wrong.",
            font=FONT_LEAD, color=theme.ASH)

        # the anti-cheat note first, before any steps; attribution goes last of the three, since
        # it's a notice rather than something the reader has to act on. deliberately measured, not
        # scary: menu-only bloodweb automation has a long community track record.
        self._callout(
            "warn", "A note on anti-cheat",
            "Dead by Daylight runs Easy Anti-Cheat, and this tool works by sending input in the "
            "menus, so use it at your own discretion. Community tools that automate the bloodweb "
            "the same way have been around for years with sizeable user bases and no known ban "
            "waves, but past record isn't a guarantee. Dry-run mode sends no clicks and is the "
            "default, so you can try everything safely before anything ever touches the game.")
        self._callout(
            "warn", "Run the game in borderless windowed mode",
            "Set Dead by Daylight to Borderless (windowed fullscreen), not exclusive fullscreen. "
            "Exclusive fullscreen gives back black screen captures, can swallow the clicks, and "
            "blocks the panic hotkey. Borderless looks identical and makes capture, clicking, and "
            f"the {start_key}/{kill_key} hotkeys all work.")
        self._callout(
            "notice", "Attribution",
            "Icon art, names, and descriptions come from deadbydaylight.wiki.gg and are used for "
            "matching and shown here. This is an unofficial fan tool, not affiliated with Behaviour "
            "Interactive (but they could hire me though). Game assets © Behaviour Interactive.")

        # step 1: icon library
        b = self._section(1, "Get the icon library")
        self._para(
            b,
            "The app needs Dead by Daylight's icons before it can recognize or show anything. On a "
            "fresh install there's no library yet, so it offers to fetch it on first launch. Let "
            "that run once. It pulls every icon and its details from the community wiki and takes a "
            "few minutes, with a progress bar.")
        self._para(
            b,
            "If you skipped the prompt, or a game patch added new content, use the ⟳ Update "
            "icons button at the bottom of the left nav rail to fetch or refresh it anytime. After "
            "the first fetch everything is cached, so later startups are fast.")

        # step 2: priorities
        b = self._section(2, "Build your priority list")
        self._para(
            b,
            "Open the Priorities tab. The left side is the full icon library with search and "
            "filters, the role dropdown narrows to survivor "
            "gear or to one specific killer's bloodweb, and the event / n/a checkboxes reveal "
            "event-only and never-buyable glyphs (miscellaneous wiki garbage). The right side is your priority list, arranged "
            "as tiers stacked from most wanted at the top to least wanted at the bottom. "
            "If 'within tier' is set to 'Ordered', the ordering of glyphs within each tier will be respected by the matcher, "
            "if set to random, it won't be."
            "note: you can drag the divider between the panes to resize them.")
        self._subhead(b, "How buying decides")
        self._para(
            b,
            "Each scan reads the top tier first and buys the first matching node it finds (in order within tier is set to 'Ordered'), "
            "then the next tier, and so on. It's one buy per scan on purpose, since the game auto-paths to "
            "whatever you click. When none of your tiers match anything left on the web, it clicks "
            "the center to auto-spend the remainder and advance.")
        self._subhead(b, "Adding rules")
        self._bullet(b, "Add an item two ways: click a tier to select it, then the glyph of your choice to add it into the selected "
                        "tier. Alternatively, drag a glyph from the library into whichever tier you want and release.")
        self._bullet(b, "Use the group cards at the bottom right to add a whole category, like "
                        "'any perks'”' or 'any ultra rare addon'.")
        self._bullet(b, "Use the template dropdown to drop in a pre-made priority profile and salt to taste")
        self._bullet(b, "A tier can be set to ordered, so within that one tier it prefers the "
                        "rule listed earliest when there are multiple matches within a single tier.")
        self._subhead(b, "Profiles")
        self._para(
            b,
            "Profiles are named priority lists. Keep one for your survivor grind and one per "
            "killer, then switch between them from the Profile picker. Tag each profile with the "
            "side dropdown next to it and the picker groups your survivor and killer lists "
            "separately, so a long list stays navigable.")
        self._callout(
            "tip", "Save your profile",
            "When you finish assembling or editing a priority list, hit Save. A star on the button "
            "(Save *) means there are unsaved changes, and Revert throws them away. Runs read the "
            "saved config, so an unsaved list isn't what a run will use. Simply switching profiles "
            "doesn't need a save; the app remembers your pick on its own.")

        # step 3: settings
        b = self._section(3, "Check your settings")
        self._para(
            b,
            "Open the Settings tab to set the details of how a run behaves. Most defaults are "
            "fine; here's what each knob actually does, group by group.")
        self._subhead(b, "Display & accessibility")
        self._bullet(b, "Text size scales every font and control in the app.")
        self._bullet(b, "Enable debugging adds the Debug view to the nav rail (see the reporting "
                        "section at the bottom of this page).")
        self._bullet(b, "Show tooltips toggles the hover descriptions on library cards and placed "
                        "rules.")
        self._subhead(b, "Hotkeys")
        self._bullet(b, f"The start key (default {start_key}) starts a run, pauses it, and resumes "
                        "it, and works with the game focused, so you can start from inside DBD. "
                        f"The panic key (default {kill_key}) always stops. Click a button and press "
                        "a new key to rebind; restart the app for a rebind to re-arm the global "
                        "hotkeys.")
        self._subhead(b, "Detection & matching")
        self._bullet(b, "Matching method decides how a detected node's icon is identified. cnn "
                        "(the learned matcher) is the default and the most accurate; ncc, "
                        "ncc_masked, and phash are classical fallbacks for experiments.")
        self._bullet(b, "Binarization method and node detection control how node circles are found "
                        "in the frame. Leave them alone unless the Debug view shows missed or "
                        "invented nodes.")
        self._bullet(b, "Presence threshold: after the main pass, empty web slots are re-checked "
                        "with a learned 'does a node sit here' score, and slots above this floor "
                        "are recovered as missed nodes. Lower it if real nodes go undetected, "
                        "raise it if phantom nodes appear.")
        self._bullet(b, "Matcher rescue min score and margin: a weak icon match normally falls "
                        "back to hovering the node and reading its tooltip with OCR (sure, but "
                        "slow). A match that scores at least the min AND beats its runner-up by "
                        "the margin is trusted without OCR. Raise the margin if wrong items get "
                        "bought; lower it if runs spend too long hovering.")
        self._subhead(b, "Match pool")
        self._bullet(b, "Narrow match pool to priority sources compares each node only against "
                        "icons that can appear in the webs your priorities imply, so a survivor "
                        "list never scores another killer's add-ons. Recommended on.")
        self._bullet(b, "Only compare against the priority list is the strict version: anything "
                        "not literally in your list reads as unknown. Fastest, but the log tells "
                        "you less about the rest of the web.")
        self._bullet(b, "Skip nodes when a weak match's OCR read fails: by default such a node "
                        "falls back to its best icon guess; flip this on to skip it instead.")
        self._subhead(b, "Spend order")
        self._bullet(b, "Race the entity breaks ties within a tier toward the node nearest the "
                        "entity, since whatever it eats is gone for good. Opt-in, because it "
                        "changes which node an equal-priority tie picks.")
        self._subhead(b, "Timing (seconds)")
        self._bullet(b, "Post-buy settle wait: pause after each buy before the next decision, "
                        "covering the buy animation. Raise it on a laggy machine.")
        self._bullet(b, "Entity smoke wait: extra pause before node states are re-read after a "
                        "buy, so the entity's smoke has finished rendering. 0 is fine too; a "
                        "missed read self-corrects a buy later.")
        self._bullet(b, "OCR tooltip wait: how long the game's name tooltip gets to fade in "
                        "before it's read. Raise it if the log shows failed reads.")
        self._bullet(b, "Level transition wait: pause after the center auto-spend (and after a "
                        "prestige) while the web fills and the next level renders. Raise it if "
                        "scans start mid-transition.")
        self._subhead(b, "Stops & prestige")
        self._bullet(b, "Auto-prestige at level 50 clicks the prestige star once the web is spent "
                        "(costing 20k bloodpoints) and carries on. The prestige animation wait is "
                        "the pause between that click and looking for the rewards screen's OK "
                        "button; raise it if the run sits through the animation and misses OK.")
        self._bullet(b, "Stop at bloodpoints remaining / prestige level / bloodweb level end a run "
                        "at a floor or a goal; 0 turns a stop off. Separately, every live run "
                        "stops on its own when it reads that your bloodpoints can't cover the "
                        "next node it wants.")
        self._para(
            b,
            "Your settings, priority profiles, icon cache, and debug output all live under "
            "%APPDATA%\\dbdbp-pas, separate from the app folder, so moving or replacing the app "
            "keeps everything. Deleting that folder resets you to a fresh install.")

        # step 4: dry run
        b = self._section(4, "Test with a dry run first")
        self._para(
            b,
            "Open the Run tab. Before letting it touch the game, prove your list does what you "
            "expect. Two safe ways, neither sends a single click:")
        self._bullet(b, "Use simulator generates fake bloodweb levels with no game needed, so you "
                        "can watch which nodes your priorities pick. This mode is always dry.")
        self._bullet(b, "Dry run (no clicks) runs against the real game on screen and logs every "
                        "buy it would make, without actually clicking. Open the game to a bloodweb, "
                        "start it, and read the log.")
        self._para(
            b,
            "Watch the log pane. Each decision is written out so you can confirm it's picking the "
            "right nodes in the right order before you hand it real input.")

        # step 5: go live
        b = self._section(5, "Go live")
        self._para(
            b,
            "When the dry run looks right, uncheck Dry run and Use simulator, then Start. Only a "
            "live, non-simulator, non-dry run actually clicks in the game. The on-screen buttons "
            f"mirror the global hotkeys, so you can also tab into the game first and press "
            f"{start_key} from there — it starts the run just like the Start button.")
        self._callout(
            "tip", "Stopping fast",
            f"Press {kill_key} (or Stop) to cut the run before you queue into a match. The cursor "
            "parks in a corner between actions so it isn't left hovering a node.")
        self._callout(
            "tip", "A short warm-up",
            "The detector builds a small template of what a node looks like on your machine from "
            "the first webs it sees, so the first level or two of a fresh install can miss a small "
            "percentage of items. It converges after a couple of levels, and the center auto-spend "
            "still buys anything a scan missed.")

        # hotkeys table
        b = self._section(None, "Hotkey reference")
        self._hotkey_table(b, start_key, kill_key)

        # troubleshooting
        b = self._section(None, "If something looks wrong")
        self._bullet(b, "Nothing gets detected. Make sure the game is in borderless windowed mode "
                        "and the bloodweb is actually on screen and unobstructed.")
        self._bullet(b, "It buys the wrong things. Recheck your tier order in Priorities, and "
                        "confirm you saved and are on the profile you think you are.")
        self._bullet(b, "Lots of failed reads in the log. Raise the OCR tooltip wait and the "
                        "post-buy settle wait in Settings.")
        self._bullet(b, "The library looks empty or out of date. Run ⟳ Update icons from the "
                        "nav rail.")
        self._bullet(b, "Windows SmartScreen or antivirus flags the app. It ships input-injection "
                        "libraries and isn't code-signed, so that's expected. Allow it if you trust "
                        "this build.")

        # debug + reporting (the part they specifically asked to end on)
        self._debug_section(start_key, kill_key)

        # footer: point users at the github issue tracker with a clickable link.
        foot = ctk.CTkFrame(self.col, fg_color="transparent")
        foot.pack(fill="x", anchor="w", pady=(theme.PAD, 0))
        self._para(
            foot,
            "Found a bug, or have a request? Please open an issue on GitHub so it can be tracked "
            "and looked at:")
        self._link(foot, GITHUB_ISSUES_URL, GITHUB_ISSUES_URL)

    def _link(self, parent, text, url):
        """a clickable, underlined link label that opens `url` in the default browser."""
        link = ctk.CTkLabel(parent, text=text, font=("Segoe UI", 12, "underline"),
                            text_color=theme.ACCENT_BRIGHT, anchor="w", cursor="hand2")
        link.pack(anchor="w", pady=(0, theme.PAD))
        link.bind("<Button-1>", lambda e: webbrowser.open(url))

    def _hotkey_table(self, parent, start_key, kill_key):
        rows = [
            ("Key", "Action"),
            (start_key, "Start the run, or pause and resume it"),
            (kill_key, "Panic stop, cuts the run"),
        ]
        table = ctk.CTkFrame(parent, fg_color="transparent")
        table.pack(fill="x", anchor="w")
        table.grid_columnconfigure(1, weight=1)
        for i, (key, action) in enumerate(rows):
            is_head = i == 0
            font = theme.FONT_TITLE if is_head else theme.FONT_BODY
            key_lbl = ctk.CTkLabel(
                table, text=key, font=(theme.FONT_SMALL if not is_head else theme.FONT_TITLE),
                width=90, anchor="w",
                fg_color=(None if is_head else theme.ACCENT),
                corner_radius=(0 if is_head else 6),
                text_color=(theme.BONE if not is_head else None))
            key_lbl.grid(row=i, column=0, sticky="w", padx=(0, theme.PAD), pady=4)
            ctk.CTkLabel(table, text=action, font=font, anchor="w").grid(
                row=i, column=1, sticky="w", pady=4)

    def _debug_section(self, start_key, kill_key):
        card = ctk.CTkFrame(self.col, border_width=2, border_color=theme.ACCENT)
        card.pack(fill="x", pady=(theme.PAD, theme.PAD))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=theme.PAD, pady=(theme.PAD, 4))
        ctk.CTkLabel(head, text="\U0001f41b", font=theme.FONT_TITLE).pack(
            side="left", padx=(0, theme.PAD))
        ctk.CTkLabel(head, text="Debug mode and reporting a problem", font=FONT_SECTION,
                     anchor="w").pack(side="left", fill="x", expand=True)
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=theme.PAD, pady=(0, theme.PAD))

        self._para(
            body,
            "If a run does the wrong thing, debug mode shows you exactly what the detector sees, and "
            "gives you the files to attach to a bug report. Here's the flow to capture a good one.")

        self._subhead(body, "Turn on debug mode")
        self._para(
            body,
            "Flip Enable debugging on the Settings tab, or the Debugging checkbox on the Run tab. A "
            "new Debug entry appears in the left nav rail. Turning it off hides that view again.")

        self._subhead(body, "What the Debug view shows")
        self._bullet(body, "Detector view. The live captured frame with the detector's boxes drawn "
                          "on top, so you can see which nodes it found, what it thinks each one is, "
                          "and which it missed. Zoom and pan with the buttons or the mouse wheel.")
        self._bullet(body, "OCR readout. The prestige, bloodweb level, and bloodpoint values it read "
                          "off the screen, so you can check them against what the game shows.")
        self._bullet(body, "Save frame. Writes the current frame to the debug folder as PNGs, both "
                          "the annotated overlay and the raw capture behind it (the raw one is "
                          "what detection can be re-run against when debugging a report).")
        self._bullet(body, "Maintenance. Open or clear the regenerable cache and the debug-output "
                          "folder, and re-run the icon scraper. Clearing the cache is always safe, "
                          "it rebuilds on demand.")

        self._subhead(body, "Capturing a report")
        self._numbered(body, [
            "Turn on debugging, then reproduce the problem with a run (a dry run is fine, and "
            "safest).",
            "When you see the bad behavior, open the Debug view and click Save frame to grab the "
            "annotated image.",
            "Note the matching method, detection method, and any timing values you changed in "
            "Settings, plus your game resolution.",
            "Click Open debug folder in Maintenance and collect the saved frame(s). Copy the log "
            "text from the Run tab too.",
        ])
        self._para(
            body,
            "The debug output and your config both live under %APPDATA%\\dbdbp-pas. Attach the saved "
            "frame, the log text, and a short note on what you expected versus what happened. That "
            "combination is usually enough to pin down what went wrong.")

    def _numbered(self, parent, items):
        for i, text in enumerate(items, start=1):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", anchor="w", pady=(0, 4))
            ctk.CTkLabel(row, text=str(i), font=theme.FONT_SMALL, width=22, height=22,
                         fg_color=theme.ACCENT, corner_radius=11,
                         text_color=theme.BONE).pack(side="left", anchor="n", padx=(0, theme.PAD))
            # height=1 + anchor="nw" + packed anchor="n", so the text tops out level with the badge
            # instead of centering itself in a box the badge doesn't share (see _bullet).
            lbl = ctk.CTkLabel(row, text=text, font=theme.FONT_BODY, justify="left", anchor="nw",
                               height=1, wraplength=CONTENT_MAX)
            lbl.pack(side="left", fill="x", expand=True, anchor="n", pady=(3, 0))
            self._wrap_labels.append((lbl, 22 + 3 * theme.PAD + 8))
