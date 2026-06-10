"""main loop: capture -> detect -> decide -> click -> rescan, with a kill switch.

re-scans between buys because the web changes as you spend (and as the entity eats
nodes). dry-run mode logs what it WOULD click and sends no input, for safe testing.
stop = always-on global hotkey kill switch, plus the optional bp-threshold via ocr.
"""

# TODO: wire capture + detect + priority + input + ocr into the loop.
# TODO: register the keyboard kill-switch hotkey.
# TODO: dry-run path that logs intended clicks and sends nothing.


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
