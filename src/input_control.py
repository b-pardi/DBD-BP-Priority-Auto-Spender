"""mouse move + click into dbd.

pydirectinput over pyautogui: dbd ignores pyautogui's synthesized events, while pydirectinput sends hardware-level input (SendInput) the game actually picks up.
a buy might be a tap or a press-and-hold, so click_node presses for hold_s seconds and we tune that live.

coords here are absolute screen coords with origin at the primary monitor's top-left, which is also pydirectinput's absolute-coord origin (it normalizes against the primary monitor size only), so it reliably targets the display where the bloodweb lives.
map detection's frame coords to screen coords first via capture.frame_to_screen.
"""

import time
import pydirectinput

# we own all the timing explicitly below (moves, press duration, settles), so drop pydirectinput's implicit per-call pause to avoid double-waiting.
# failsafe is left on, so slamming the cursor into a screen corner aborts, a handy manual complement to the keyboard kill switch.
pydirectinput.PAUSE = 0.02


def get_position():
    """current cursor position as (x, y) screen ints.
    pydirectinput.position is a plain GetCursorPos wrapper without the failsafe decorator, so this is safe to call even with the cursor in a screen corner."""
    return pydirectinput.position()


def move_to(x, y, pause=0.02):
    """move the cursor to screen (x, y).
    the brief pause lets dbd register the hover before any press, since it highlights a node on hover and an instant press can miss that.
    usable on its own as a safe live test, to watch the cursor land on each target without buying anything."""
    pydirectinput.moveTo(int(round(x)), int(round(y)))
    if pause:
        time.sleep(pause)


def click_node(x, y, hold_s=0.03, settle=0.0):
    """move to screen (x, y) and buy the node by pressing for hold_s seconds.
    a plain tap is just a short press, and if dbd needs a real press-and-hold to purchase (the fill animation) you raise hold_s and confirm live.
    settle optionally waits after the release, though the loop usually owns the post-buy rescan delay."""
    move_to(x, y)
    pydirectinput.mouseDown()
    time.sleep(hold_s)
    pydirectinput.mouseUp()
    if settle:
        time.sleep(settle)
