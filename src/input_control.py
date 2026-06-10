"""mouse move + click into dbd.

pydirectinput over pyautogui: dbd ignores pyautogui's synthesized events, while
pydirectinput sends scancode/hardware-level input the game actually picks up. whether
a buy is a click or a click-and-hold gets confirmed live.
"""

# TODO: click_node(x, y) -> move to (x, y) then click (or click-and-hold)


def click_node(x, y):
    raise NotImplementedError
