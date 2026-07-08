"""Разовая загрузка официальной предобученной модели yolov8n-seg.pt в папку
weights/ проекта.

ВАЖНО: этот скрипт НИКОГДА не вызывается самим приложением NeuroMovement.
Он предназначен для однократного запуска человеком при подготовке проекта
или перед сборкой .exe — единственный момент, когда программе вообще
нужен доступ в интернет. Само приложение (main.py) после этого работает
полностью офлайн и даже аппаратно блокирует сетевые вызовы из Python
(см. neuromovement/network_guard.py).

Запуск:
    python scripts/fetch_weights.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "yolov8n-seg.pt"
MODEL_NAME = "yolov8n-seg.pt"


def main() -> None:
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if WEIGHTS_FILE.exists():
        print(f"Файл уже существует: {WEIGHTS_FILE}")
        print("Если хотите перезагрузить заново — удалите файл и запустите скрипт снова.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ОШИБКА: пакет ultralytics не установлен.")
        print("Сначала выполните: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Загрузка официальной модели {MODEL_NAME} с серверов Ultralytics (GitHub Releases)...")
    print("Лицензия модели: AGPL-3.0 (https://ultralytics.com/license)")

    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        print(f"ОШИБКА при загрузке модели: {exc}")
        print(
            "Проверьте подключение к интернету. Если проблема повторяется, "
            "скачайте файл вручную со страницы релизов "
            "https://github.com/ultralytics/assets/releases и поместите его "
            f"как {WEIGHTS_FILE}"
        )
        sys.exit(1)

    downloaded_path = Path(getattr(model, "ckpt_path", "") or "")
    if not downloaded_path.exists():
        candidate = Path.cwd() / MODEL_NAME
        if candidate.exists():
            downloaded_path = candidate

    if not downloaded_path.exists():
        print("ОШИБКА: модель загрузилась, но не удалось определить путь к файлу.")
        sys.exit(1)

    shutil.copy(downloaded_path, WEIGHTS_FILE)
    size_mb = WEIGHTS_FILE.stat().st_size / (1024 * 1024)
    print(f"Готово: {WEIGHTS_FILE} ({size_mb:.1f} МБ)")
    print("Теперь приложение можно запускать полностью офлайн: python main.py")


if __name__ == "__main__":
    main()
