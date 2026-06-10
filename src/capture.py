"""grab the bloodweb region off-screen.

mss because it's the fastest screen grab on windows, well ahead of PIL.ImageGrab.
returns a bgr ndarray (h, w, 3) so frames drop straight into opencv.
"""

# TODO: implement with mss. grab full screen or a calibrated region,
# convert bgra -> bgr, return ndarray shape (h, w, 3).


def grab_bloodweb():
    """return the current bloodweb frame as a bgr ndarray (h, w, 3)."""
    raise NotImplementedError
