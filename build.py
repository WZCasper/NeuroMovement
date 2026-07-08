"""Автоматизация сборки портативного Windows-приложения NeuroMovement.

Использование:
    python build.py

Требует установленный PyInstaller (см. requirements-dev.txt) и файл весов
модели в weights/yolov8n-seg.pt (см. weights/README.md).

Собранный .exe будет работать только на той ОС, на которой запущена эта
сборка (PyInstaller не выполняет кросс-компиляцию). Для получения
Windows-сборки без собственного Windows-компьютера используйте GitHub
Actions — см. .github/workflows/build.yml, который автоматически собирает
Windows-артефакт на официальном windows-latest runner при создании релиза.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_FILE = PROJECT_ROOT / "neuromovement.spec"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
WEIGHTS_FILE = PROJECT_ROOT / "weights" / "yolov8n-seg.pt"


def check_prerequisites() -> None:
    if not WEIGHTS_FILE.exists():
        print(
            f"ОШИБКА: файл весов не найден: {WEIGHTS_FILE}\n"
            "Запустите 'python scripts/fetch_weights.py' перед сборкой."
        )
        sys.exit(1)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("ОШИБКА: PyInstaller не установлен. Выполните: pip install -r requirements-dev.txt")
        sys.exit(1)


def clean_previous_build() -> None:
    for directory in (DIST_DIR, BUILD_DIR):
        if directory.exists():
            shutil.rmtree(directory)


def run_pyinstaller() -> None:
    command = [sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"]
    print("Запуск:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("ОШИБКА: сборка PyInstaller завершилась с ошибкой (см. вывод выше).")
        sys.exit(result.returncode)


def main() -> None:
    check_prerequisites()
    clean_previous_build()
    run_pyinstaller()

    output_dir = DIST_DIR / "NeuroMovement"
    exe_name = "NeuroMovement.exe" if sys.platform == "win32" else "NeuroMovement"
    print(f"\nГотово. Портативная сборка находится в: {output_dir}")
    print(f"Исполняемый файл: {output_dir / exe_name}")
    print("Скопируйте всю папку NeuroMovement/ на целевой компьютер — она самодостаточна.")


if __name__ == "__main__":
    main()
