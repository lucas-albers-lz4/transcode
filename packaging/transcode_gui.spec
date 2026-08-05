# PyInstaller spec for the HEVC transcoder GUI (ffmpeg is NOT bundled).
# Build from repo root:
#   uv pip install -r requirements-gui.txt
#   pyinstaller packaging/transcode_gui.spec --noconfirm

import sys
from pathlib import Path

import customtkinter
from PyInstaller.utils.hooks import collect_all

block_cipher = None
root = Path(SPECPATH).parent
ct_path = Path(customtkinter.__file__).parent

tk_datas, tk_binaries, tk_hiddenimports = collect_all("tkinter")


def _tcl_tk_shared_libs():
    """uv/standalone CPython keeps Tcl/Tk .so under base_prefix/lib (off ld path)."""
    libdir = Path(sys.base_prefix) / "lib"
    bins = []
    for pattern in ("libtcl*.so*", "libtk*.so*", "libtcl9tk*.so*"):
        for path in sorted(libdir.glob(pattern)):
            if path.is_file() and not path.is_symlink():
                bins.append((str(path.resolve()), "."))
            elif path.is_symlink() and path.resolve().is_file():
                bins.append((str(path.resolve()), "."))
    # Deduplicate by destination basename
    seen: set[str] = set()
    unique = []
    for src, dest in bins:
        name = Path(src).name
        if name in seen:
            continue
        seen.add(name)
        unique.append((src, dest))
    return unique


a = Analysis(
    [str(root / "transcode_gui.py")],
    pathex=[str(root)],
    binaries=list(tk_binaries) + _tcl_tk_shared_libs(),
    datas=[
        (str(ct_path / "assets"), "customtkinter/assets"),
        *tk_datas,
    ],
    hiddenimports=[
        "customtkinter",
        "encode_profiles",
        "workflow",
        "scan_media",
        "convert_media",
        "media_analysis",
        "analyze_space",
        "analyze_errors",
        "ffmpeg_utils",
        "psutil",
        "gui.app",
        "gui.step_folders",
        "gui.step_convert",
        "gui.workers",
        "gui.log_redirect",
        "gui.ffmpeg_gate",
        "gui.theme",
        *tk_hiddenimports,
    ],
    hookspath=[str(root / "packaging" / "pyinstaller_hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="transcode_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="transcode_gui",
)
