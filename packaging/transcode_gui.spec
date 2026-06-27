# PyInstaller spec for the HEVC transcoder GUI (ffmpeg is NOT bundled).
# Build from repo root:
#   pip install -r requirements-gui.txt
#   pyinstaller packaging/transcode_gui.spec

from pathlib import Path

block_cipher = None
root = Path(SPECPATH).parent.parent

a = Analysis(
    [str(root / "transcode_gui.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "customtkinter",
        "encode_profiles",
        "workflow",
        "gui.app",
        "gui.step_folders",
        "gui.step_convert",
        "gui.workers",
        "gui.log_redirect",
        "gui.ffmpeg_gate",
    ],
    hookspath=[],
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
