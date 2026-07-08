"""Дешёвый детектор грубого движения на основе разницы кадров.

Используется как быстрый предварительный фильтр перед дорогим инференсом
YOLO: если в кадре нет заметного изменения по сравнению со скользящим
фоном, сегментация не запускается вовсе. Это экономит ресурсы CPU/GPU
при 24/7-мониторинге статичной сцены и одновременно реализует требование
"определения области, в которой происходит движение" независимо от того,
распознал ли YOLO конкретный объект.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, min_area: int = 900, blur_kernel: int = 21, threshold: int = 25) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.threshold = threshold
        self._background: np.ndarray | None = None

    def reset(self) -> None:
        self._background = None

    def detect(self, frame_bgr: np.ndarray) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """Возвращает (есть_ли_движение, список_прямоугольников_движения)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        if self._background is None or self._background.shape != gray.shape:
            self._background = gray.astype("float32")
            return False, []

        cv2.accumulateWeighted(gray, self._background, 0.05)
        background_u8 = cv2.convertScaleAbs(self._background)

        diff = cv2.absdiff(gray, background_u8)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[tuple[int, int, int, int]] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, x + w, y + h))

        return (len(regions) > 0), regions
