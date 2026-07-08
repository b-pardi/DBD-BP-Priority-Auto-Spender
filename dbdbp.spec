# dbdbp.spec, pyinstaller onedir build for the dbd bloodweb auto-spender.
# build from the activated dbdbp conda env (the build needs its Library/bin dlls + tessdata):
#   conda activate dbdbp
#   pip install pyinstaller
#   pyinstaller dbdbp.spec
# onedir (a folder), not onefile: onefile re-unpacks tens of MB to a temp dir every launch.
# nothing icon-related ships, the wiki library + ncc/cnn caches build into %APPDATA%/dbdbp on first run.

import os
import sys
import glob
import shutil
from PyInstaller.utils.hooks import collect_data_files

BIN = os.path.join(sys.prefix, "Library", "bin")            # conda dll home
SHARE_TESS = os.path.join(sys.prefix, "share", "tessdata")  # eng.traineddata lives here

# tesserocr loads its dll chain at runtime via os.add_dll_directory(_MEIPASS) (src/ocr_runtime.py),
# so pyinstaller's link analysis can miss them, bundle the chain explicitly at the bundle root.
# glob by pattern so a version string in a filename (leptonica-1.78.0.dll) still matches.
_dll_pats = [
    "tesseract*.dll", "leptonica*.dll", "tiff*.dll", "libtiff*.dll", "libjpeg*.dll",
    "libpng*.dll", "zlib*.dll", "deflate.dll", "libdeflate.dll", "libsharpyuv*.dll",
    "openjp2*.dll", "liblzma*.dll", "lzma*.dll", "zstd*.dll", "libwebp*.dll", "webp*.dll",
    "iconv*.dll", "archive*.dll",
]
# libtiff asks for libdeflate.dll but conda only ships deflate.dll (same lib, renamed), stage the
# alias at build time so the runtime copy in ocr_runtime never has to write into a read-only install.
_alias = os.path.join(BIN, "libdeflate.dll")
if not os.path.exists(_alias) and os.path.exists(os.path.join(BIN, "deflate.dll")):
    shutil.copyfile(os.path.join(BIN, "deflate.dll"), _alias)

binaries = []
for pat in _dll_pats:
    for f in glob.glob(os.path.join(BIN, pat)):
        binaries.append((f, "."))  # '.' == bundle root == _MEIPASS, where ocr_runtime adds the dll dir

# read-only bundled assets, all resolved through src.paths.resource_path when frozen (_MEIPASS/<rel>).
datas = []
datas += [("config/priority.json", "config")]                  # seed, ensure_user_dirs copies it to %APPDATA%
datas += [("data/models/glyph_encoder.onnx", "data/models")]   # cnn matcher weights (detect.CNN_ONNX)
datas += [("ui/assets/icon.ico", "ui/assets"), ("ui/assets/icon.png", "ui/assets")]
datas += collect_data_files("customtkinter")                   # ctk loads theme json at runtime
if os.path.isfile(os.path.join(SHARE_TESS, "eng.traineddata")):
    datas += [(os.path.join(SHARE_TESS, "eng.traineddata"), "tessdata")]  # _tessdata() probes _MEIPASS/tessdata

# tesserocr is loaded lazily inside a function, keyboard/pydirectinput/mss are pip pure-python,
# list them so the graph never drops one. cv2/PIL/customtkinter are found via imports but harmless here.
hiddenimports = [
    "tesserocr", "keyboard", "pydirectinput", "mss",
    "cv2", "PIL", "imagehash", "customtkinter",
]

# matplotlib is dev-only debug draw (src.detect._plt), never on the ui/spend path, drop it.
# torch is training-only (tools/glyph_cnn), at runtime the onnx runs via cv2.dnn, no torch needed.
excludes = ["matplotlib", "torch", "torchvision"]

a = Analysis(
    ["dbdbp_main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dbdbp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # gui app, no console window (flip to True to see stdout while debugging a build)
    icon="ui/assets/icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="dbdbp",
)
