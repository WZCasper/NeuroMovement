"""Главное окно приложения NeuroMovement.

Видео отображается абсолютно чистым: единственное, что рисуется поверх
кадра — это тревожная визуализация из vision/overlay.py, и только в
момент реального обнаружения движения. Никакие статусы/FPS/подсказки не
выводятся на сам кадр — они живут в статус-баре, вне зоны видео.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QStatusBar, QWidget

from neuromovement.config import Settings
from neuromovement.pipeline import CaptureWorker
from neuromovement.ui.settings_panel import SettingsPanel
from neuromovement.ui.video_widget import VideoWidget
from neuromovement.vision.detector import SegmentationDetector

_STATUS_LABELS = {
    "connected": "Подключено",
    "reconnecting": "Переподключение…",
    "stopped": "Остановлено",
}


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, detector: SegmentationDetector) -> None:
        super().__init__()
        self.settings = settings
        self.detector = detector
        self.worker: CaptureWorker | None = None
        self._frame_count = 0

        self.setWindowTitle("NeuroMovement")
        self.resize(1280, 800)

        self.video_widget = VideoWidget()
        self.settings_panel = SettingsPanel(self.settings)
        self.settings_panel.start_requested.connect(self.start_monitoring)
        self.settings_panel.stop_requested.connect(self.stop_monitoring)
        self.settings_panel.settings_changed.connect(self._on_settings_changed)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video_widget, 1)
        layout.addWidget(self.settings_panel, 0)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Остановлено")
        self.fps_label = QLabel("FPS: —")
        self.device_label = QLabel(self._device_text())
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.fps_label)
        self.status_bar.addPermanentWidget(self.device_label)

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

    def _device_text(self) -> str:
        device = getattr(self.detector, "device", "cpu")
        return f"Устройство: {device.upper()}"

    def start_monitoring(self) -> None:
        if self.worker is not None:
            return
        self.worker = CaptureWorker(self.settings, self.detector)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()
        self.settings_panel.set_running(True)

    def stop_monitoring(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(3000)
        self.worker = None
        self.settings_panel.set_running(False)
        self.video_widget.show_no_signal("Остановлено")
        self.status_label.setText("Остановлено")

    def _on_frame_ready(self, frame, _has_detection: bool) -> None:
        self._frame_count += 1
        self.video_widget.show_frame(frame)

    def _on_status_changed(self, status: str) -> None:
        self.status_label.setText(_STATUS_LABELS.get(status, status))

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"Ошибка: {message}")

    def _on_settings_changed(self) -> None:
        self.settings.save()

    def _update_fps(self) -> None:
        self.fps_label.setText(f"FPS: {self._frame_count}")
        self._frame_count = 0

    def closeEvent(self, event) -> None:  # noqa: N802 (сигнатура Qt)
        self.stop_monitoring()
        super().closeEvent(event)
