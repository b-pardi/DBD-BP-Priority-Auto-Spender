# ui assets

`icon.ico` / `icon.png` are a **placeholder**: a plain black square. Replace them with the real
logo when you have one (keep the same filenames, or update `_set_window_icon` in `ui/app.py`).

- `icon.ico` is what Windows uses for the title-bar + taskbar icon (`iconbitmap`). Keep it a real
  multi-size `.ico` (16/32/48/256 px) so it stays crisp at every size.
- `icon.png` is a square fallback used via `iconphoto`.

When packaging with pyinstaller, bundle this folder (e.g. `--add-data "ui/assets;ui/assets"`) and
pass `--icon ui/assets/icon.ico` for the exe's own icon.
