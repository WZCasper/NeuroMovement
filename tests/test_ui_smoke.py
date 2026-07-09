"""Безголовый (offscreen) интеграционный тест интерфейса.

Требует переменную окружения QT_QPA_PLATFORM=offscreen (задаётся в
conftest.py). Не проверяет визуальный вид (это невозможно без реального
экрана), но проверяет, что окно, панель настроек и конвейер захвата
реально конструируются и работают вместе без исключений — включая полный
цикл запуск -> получение кадров -> остановка."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from neuromovement.config import Settings
from neuromovement.ui.main_window import MainWindow


class _StubDetector:
    """Дублёр SegmentationDetector для UI-теста — не требует PyTorch/YOLO."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.confidence = 0.35

    def set_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    def detect(self, frame_bgr: np.ndarray):
        return []


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("ui_video") / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    del writer
    time.sleep(0.2)
    return str(path)


def test_main_window_constructs_without_error(qapp):
    settings = Settings()
    window = MainWindow(settings, _StubDetector())
    assert window.windowTitle() == "NeuroMovement"
    assert window.video_widget is not None
    assert window.settings_panel is not None
    window.close()


def test_settings_panel_widgets_present(qapp):
    settings = Settings()
    window = MainWindow(settings, _StubDetector())
    panel = window.settings_panel
    assert panel.rtsp_edit.text() == settings.rtsp_url
    assert panel.blink_checkbox.isChecked() == settings.blink_enabled
    assert panel.thickness_slider.value() == settings.contour_thickness
    window.close()


def test_full_start_stop_cycle_receives_frames(qapp, synthetic_video):
    settings = Settings(source_type="rtsp", rtsp_url=synthetic_video, motion_gate_enabled=False)
    window = MainWindow(settings, _StubDetector())

    window.start_monitoring()
    assert window.worker is not None

    deadline = time.monotonic() + 15.0
    while window._frame_count == 0 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.05)

    assert window._frame_count > 0, "не получено ни одного кадра за 15 секунд"

    window.stop_monitoring()
    assert window.worker is None
    window.close()


def test_settings_changes_persist_to_disk(qapp, tmp_path):
    settings_path = tmp_path / "settings.json"
    settings = Settings()
    window = MainWindow(settings, _StubDetector())

    window.settings_panel.thickness_slider.setValue(7)
    settings.save(settings_path)

    reloaded = Settings.load(settings_path)
    assert reloaded.contour_thickness == 7
    window.close()
