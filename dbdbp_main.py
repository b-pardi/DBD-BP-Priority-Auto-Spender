"""frozen exe entry point, imports ui as a package so its relative imports resolve.

pyinstaller runs this script directly (see dbdbp.spec), and a directly-run script has no
package context, which would break `python -m ui`'s relative imports. importing ui.__main__
here keeps ui a proper package so `from .app import App` still works in the bundle.
"""

from ui.__main__ import main

if __name__ == "__main__":
    main()
