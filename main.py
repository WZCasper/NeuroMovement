"""Точка входа приложения NeuroMovement.

Блокировка сети включается САМОЙ ПЕРВОЙ строкой исполняемого кода —
раньше любого импорта, который потенциально мог бы обратиться к сети
(PySide6 в этом смысле безопасен, но detector/ultralytics/torch — нет).
"""
from __future__ import annotations

import sys

from neuromovement.network_guard import enable_offline_guard

enable_offline_guard()

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from neuromovement.config import Settings, get_weights_path  # noqa: E402
from neuromovement.ui.main_window import MainWindow  # noqa: E402
from neuromovement.ui.theme import DARK_THEME_QSS  # noqa: E402
from neuromovement.vision.detector import SegmentationDetector  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroMovement")
    app.setStyleSheet(DARK_THEME_QSS)

    settings = Settings.load()

    try:
        detector = SegmentationDetector(
            weights_path=get_weights_path(),
            device=settings.device_preference,
            confidence=settings.confidence_threshold,
        )
    except Exception as exc:
        QMessageBox.critical(None, "NeuroMovement — ошибка запуска", str(exc))
        return 1

    window = MainWindow(settings, detector)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
