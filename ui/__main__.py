"""entry point: `python -m ui` (and the eventual exe entry)."""

from src import paths
from .app import App


def main():
    paths.ensure_user_dirs()  # create writable dirs and seed the default config before anything reads them
    App().mainloop()


if __name__ == "__main__":
    main()
