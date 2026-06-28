# PyInstaller spec for the HEVC transcoder GUI (ffmpeg is NOT bundled).
# Build from repo root:
#   uv pip install -r requirements-gui.txt
#   pyinstaller packaging/transcode_gui.spec --noconfirm

import customtkinter
from pathlib import Path

block_cipher = None
root = Path(SPECPATH).parent
ct_path = Path(customtkinter.__file__).parent

a = Analysis(
    [str(root / "transcode_gui.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(ct_path / "assets"), "customtkinter/assets")],
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
