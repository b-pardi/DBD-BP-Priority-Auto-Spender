"""grab the bloodweb region off-screen.

mss because it's the fastest screen grab on windows, well ahead of PIL.ImageGrab.
returns a bgr ndarray (h, w, 3) so frames drop straight into opencv.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mss
import keyboard
import cv2

# mss binds its GDI context to the creating thread (BitBlt fails cross-thread),
# so keep one instance per thread instead of sharing or recreating every grab.
_tls = threading.local()
def get_sct():
    if getattr(_tls, "sct", None) is None:
        _tls.sct = mss.MSS()
    return _tls.sct

def grab_bloodweb(region=None):
    """return the current bloodweb frame as a bgr ndarray (h, w, 3)."""
    sct = get_sct()
    region = sct.primary_monitor if region is None else region # whole (main) monitor if no region given
    ss = sct.grab(region) # (h,w,(bgra))
    ss = np.asarray(ss)[:,:,:3] # (h,w,(bgr))
    return ss


def grab_with_region(region=None):
    """like grab_bloodweb but also return the region dict, so callers can map frame coords back to screen coords for clicking.
    returns (frame_bgr (h, w, 3), region), where region has keys left, top, width, height in mss virtual-screen coords."""
    sct = get_sct()
    region = sct.primary_monitor if region is None else region
    ss = np.asarray(sct.grab(region))[:, :, :3]
    return ss, region


def frame_to_screen(x, y, region, crop_origin=(0, 0)):
    """map a detection frame coord to an absolute screen coord for input_control.
    region is what grab_with_region returned, crop_origin is the (x0, y0) offset when detect ran on a sub-crop ((0, 0) on a full grab).
    screen origin is the primary monitor's top-left, matching pydirectinput's coord space.
    usually identity for a full primary-monitor grab, so it mainly exists for a future sub-region or cropped detect."""
    cx, cy = crop_origin
    return int(round(x + cx + region["left"])), int(round(y + cy + region["top"]))


if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        if keyboard.is_pressed('esc'):
            break
        if keyboard.is_pressed('f9'):
            frame = grab_bloodweb()
            fp = out_dir / f"web-{datetime.now():%H%M%S}.png"
            print(frame.shape, type(fp))
            cv2.imwrite(str(fp), frame)
            time.sleep(0.1) # sleep to not take mult screen caps with one press