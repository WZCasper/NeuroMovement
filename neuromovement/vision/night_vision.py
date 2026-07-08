"""Улучшение видимости в темноте на основе CLAHE.

CLAHE (Contrast Limited Adaptive Histogram Equalization) применяется только
к каналу яркости (L) в цветовом пространстве LAB, а не к каждому каналу
BGR по отдельности — это устраняет искажение цветов, которое возникает при
наивном применении CLAHE прямо к RGB/BGR-каналам.
"""
from __future__ import annotations

import cv2
import numpy as np


class NightVisionProcessor:
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def set_clip_limit(self, clip_limit: float) -> None:
        self.clip_limit = clip_limit
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Возвращает новый кадр с вытянутым контрастом. Входной массив не изменяется."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def estimate_brightness(frame_bgr: np.ndarray) -> float:
        """Средняя яркость кадра (0-255). Полезно для авто-подстройки clip_limit."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
