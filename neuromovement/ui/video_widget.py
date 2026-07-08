"""Виджет отображения кадров видео.

Все преобразования BGR->RGB и создание QImage сосредоточены здесь, в одном
месте, чтобы весь остальной код (night_vision, detector, overlay) мог
работать в нативном для OpenCV порядке каналов BGR без риска путаницы.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #000000; color: #6e7681;")
        self._current_pixmap: QPixmap | None = None
        self.setText("Нет сигнала")

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()  # .copy() отделяет буфер QImage от numpy-массива,
        # который будет перезаписан следующим кадром
        self._current_pixmap = QPixmap.fromImage(qimage)
        self._update_display()

    def show_no_signal(self, message: str = "Нет сигнала") -> None:
        self._current_pixmap = None
        self.setText(message)

    def resizeEvent(self, event) -> None:  # noqa: N802 (сигнатура Qt)
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self) -> None:
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
