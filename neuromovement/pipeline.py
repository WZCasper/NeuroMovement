"""Оркестрация конвейера обработки видео.

process_frame() — чистая функция без побочных эффектов на переданные
объекты состояния (кроме их собственного внутреннего состояния — CLAHE,
фоновая модель движения), что делает её тестируемой без Qt и без камеры.

CaptureWorker — фоновый QThread, который в цикле читает кадры из источника
и прогоняет их через process_frame, публикуя результат в GUI-поток через
Qt-сигнал (Qt автоматически использует QueuedConnection для сигналов между
разными потоками, что делает передачу кадра в GUI потокобезопасной).
"""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from neuromovement.capture import FrameSource, create_source
from neuromovement.config import Settings
from neuromovement.vision.detector import Detection, SegmentationDetector
from neuromovement.vision.motion_gate import MotionGate
from neuromovement.vision.night_vision import NightVisionProcessor
from neuromovement.vision.overlay import OverlayStyle, render_overlay


def process_frame(
    frame_bgr: np.ndarray,
    *,
    night_vision: NightVisionProcessor,
    motion_gate: MotionGate,
    detector: SegmentationDetector,
    settings: Settings,
    now: float | None = None,
) -> tuple[np.ndarray, list[Detection], bool]:
    """Обрабатывает один кадр: улучшение видимости -> (опционально) фильтр
    движения -> сегментация -> тревожная визуализация.

    Возвращает (готовый_кадр, список_обнаружений, было_ли_движение)."""
    if now is None:
        now = time.monotonic()

    detector.set_confidence(settings.confidence_threshold)
    enhanced = night_vision.enhance(frame_bgr)

    if settings.motion_gate_enabled:
        motion_present, _regions = motion_gate.detect(enhanced)
    else:
        motion_present = True

    if motion_present:
        detections = detector.detect(enhanced)
        if settings.target_classes:
            detections = [d for d in detections if d.class_id in settings.target_classes]
    else:
        detections = []  # нет движения -> кадр должен остаться абсолютно чистым

    style = OverlayStyle(
        color_bgr=settings.highlight_color_bgr(),
        mask_alpha=settings.mask_alpha,
        contour_thickness=settings.contour_thickness,
        blink_enabled=settings.blink_enabled,
        blink_frequency_hz=settings.blink_frequency_hz,
    )
    output = render_overlay(enhanced, detections, style, now=now)
    return output, detections, motion_present


class CaptureWorker(QThread):
    """Фоновый поток: непрерывно захватывает кадры и прогоняет их через
    конвейер обработки, не блокируя интерфейс."""

    frame_ready = Signal(np.ndarray, bool)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, settings: Settings, detector: SegmentationDetector, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.detector = detector
        self._running = False
        self._night_vision = NightVisionProcessor()
        self._motion_gate = MotionGate(min_area=settings.motion_min_area)
        self._source: FrameSource | None = None

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        try:
            self._source = create_source(
                self.settings.source_type,
                rtsp_url=self.settings.rtsp_url,
                monitor_index=self.settings.monitor_index,
                window_title=self.settings.window_title,
            )
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self._running = False
            return

        self.status_changed.emit("connected")
        empty_reads_streak = 0

        while self._running:
            ok, frame = self._source.read()
            if not ok or frame is None:
                empty_reads_streak += 1
                if empty_reads_streak % 30 == 0:
                    self.status_changed.emit("reconnecting")
                self.msleep(50)
                continue

            if empty_reads_streak > 0:
                self.status_changed.emit("connected")
            empty_reads_streak = 0

            try:
                processed, detections, _motion = process_frame(
                    frame,
                    night_vision=self._night_vision,
                    motion_gate=self._motion_gate,
                    detector=self.detector,
                    settings=self.settings,
                )
            except Exception as exc:
                self.error_occurred.emit(f"Ошибка обработки кадра: {exc}")
                continue

            self.frame_ready.emit(processed, len(detections) > 0)

        if self._source is not None:
            self._source.release()
        self.status_changed.emit("stopped")
