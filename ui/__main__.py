"""entry point: `python -m ui` (and the eventual exe entry)."""

from .app import App


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
