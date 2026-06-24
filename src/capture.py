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