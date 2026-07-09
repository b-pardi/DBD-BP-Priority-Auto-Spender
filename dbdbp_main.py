"""frozen exe entry point, imports ui as a package so its relative imports resolve.

pyinstaller runs this script directly (see dbdbp.spec), and a directly-run script has no
package context, which would break `python -m ui`'s relative imports. importing ui.__main__
here keeps ui a proper package so `from .app import App` still works in the bundle.
"""

import os
import sys

# windowed pyinstaller builds run under runw.exe with no console, so sys.stdout/stderr are None.
# anything that writes to them (the scraper's tqdm bar, stray print) then dies on
# 'NoneType has no attribute write', and tqdm's held lock hangs the next scrape, so point the
# streams at a sink before importing anything that might print.
if sys.stdout is None or sys.stderr is None:
    _sink = open(os.devnull, "w")
    sys.stdout = sys.stdout or _sink
    sys.stderr = sys.stderr or _sink

from ui.__main__ import main

if __name__ == "__main__":
    main()
