"""grab the bloodweb region off-screen.

mss because it's the fastest screen grab on windows, well ahead of PIL.ImageGrab.
returns a bgr ndarray (h, w, 3) so frames drop straight into opencv.
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mss
import keyboard
import cv2

# global screen capture inst
# reinitializing new screencap obj every time used is inefficient
_sct = None
def get_sct():
    global _sct
    if _sct is None:
        _sct = mss.MSS()
    return _sct

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
    region is what grab_with_region returned, and crop_origin is the (x0, y0) offset if detect ran on a sub-crop of the frame (it is (0, 0) when detecting on the full grab).
    the screen origin is the primary monitor's top-left, which matches pydirectinput's coord space.
    this is usually identity for a full primary-monitor grab, so it mainly exists so a future sub-region or cropped detect still clicks the right pixel."""
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
            print(frame.shape)
            cv2.imwrite(fp, frame)
            time.sleep(0.1) # sleep to not take mult screen caps with one press