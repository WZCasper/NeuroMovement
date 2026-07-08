# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec-файл для NeuroMovement.

Собирает автономную портативную папку с исполняемым файлом (режим
onedir — быстрее стартует и проще для диагностики, чем однофайловый exe).

Запуск (обычно вызывается через build.py, но можно и напрямую):
    pyinstaller neuromovement.spec --noconfirm
"""
import sys
from pathlib import Path

block_cipher = None

project_root = Path(SPECPATH)

datas = [
    (str(project_root / "weights"), "weights"),
]

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cv2",
    "mss",
    "ultralytics",
    "ultralytics.engine",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.segment",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "torch",
    "torchvision",
]

if sys.platform == "win32":
    hiddenimports += ["win32gui", "win32con", "win32ui", "win32api"]

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuroMovement",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name="NeuroMovement",
)
