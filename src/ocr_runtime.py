"""import tesserocr with its dependent dlls resolved, in dev and in a frozen exe.

no hardcoded paths: the dll dir derives from sys.prefix (conda dev env) or the
pyinstaller bundle dir (frozen), so nobody has to point at a username or env name.
fixes two windows gotchas, see get_tesserocr.
"""
import os
import sys
import shutil

_tess = None  # cache so the dll wiring only runs once


def _dll_dir():
    """where tesserocr's dependent dlls live for the current run."""
    if getattr(sys, "frozen", False):  # pyinstaller bundle
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.join(sys.prefix, "Library", "bin")  # conda env layout


def get_tesserocr():
    """import and return the tesserocr module, wiring up its dlls first.

    stupid windows problems this fixes:
    1. tesserocr.pyd's dlls (tesseract52, leptonica, tiff, ...) sit in the env's
       Library/bin, which python 3.8+ no longer searches via PATH, so we add it.
    2. conda's libtiff links the old name libdeflate.dll but only ships deflate.dll;
       same lib, different filename, so we make the alias once.
    """
    global _tess
    if _tess is None:
        d = _dll_dir()
        # libtiff asks for libdeflate.dll; conda only ships deflate.dll. alias it.
        alias = os.path.join(d, "libdeflate.dll")
        real = os.path.join(d, "deflate.dll")
        if not os.path.exists(alias) and os.path.exists(real):
            try:
                shutil.copyfile(real, alias)
            except OSError:
                pass  # read-only env; the import below will surface the real error
        if hasattr(os, "add_dll_directory"):  # py3.8+ ignores PATH for a .pyd's deps
            os.add_dll_directory(d)
        import tesserocr
        _tess = tesserocr
    return _tess
