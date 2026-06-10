"""read the bp counter to enforce the optional stop threshold.

pytesseract over the fixed-font digit region. needs the tesseract binary installed
system-wide (see README). only used for the optional bp stop condition; the always-on
hotkey kill switch is the primary stop.
"""

# TODO: read_bp(frame) -> int bloodpoints, or None if unreadable


def read_bp(frame):
    raise NotImplementedError
