"""Отрисовка тревожной визуализации поверх кадра.

Три слоя, все на базе OpenCV:
1. Полупрозрачная цветная заливка силуэта (cv2.addWeighted, alpha ~0.5).
2. Контур силуэта заданной толщины (cv2.findContours + cv2.drawContours).
3. Пульсирующий bounding box вокруг объекта (cv2.rectangle), который
   переключается между полностью видимым и полностью скрытым состоянием
   с частотой blink_frequency_hz (по умолчанию 2-3 раза в секунду).

Когда объектов не обнаружено, render_overlay возвращает исходный кадр
без каких-либо изменений и без копирования — кадр должен оставаться
абсолютно чистым в отсутствие движения.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from neuromovement.vision.detector import Detection


@dataclass
class OverlayStyle:
    color_bgr: tuple[int, int, int]
    mask_alpha: float
    contour_thickness: int
    blink_enabled: bool
    blink_frequency_hz: float


def blink_is_visible(style: OverlayStyle, now: float) -> bool:
    """Определяет фазу пульсации рамки в момент времени now (секунды).
    Чистая функция от времени — не зависит от номера кадра, поэтому
    частота пульсации не "плывёт" при переменном FPS."""
    if not style.blink_enabled:
        return True
    period = 1.0 / max(style.blink_frequency_hz, 0.1)
    phase = (now % period) / period
    return phase < 0.5


def render_overlay(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    style: OverlayStyle,
    now: float | None = None,
) -> np.ndarray:
    if not detections:
        return frame_bgr

    if now is None:
        now = time.monotonic()

    output = frame_bgr.copy()

    combined_mask: np.ndarray | None = None
    for det in detections:
        if det.mask is None:
            continue
        if combined_mask is None:
            combined_mask = det.mask.copy()
        else:
            combined_mask = cv2.bitwise_or(combined_mask, det.mask)

    if combined_mask is not None and combined_mask.any():
        mask_bool = combined_mask.astype(bool)
        overlay = output.copy()
        overlay[mask_bool] = style.color_bgr
        output = cv2.addWeighted(overlay, style.mask_alpha, output, 1 - style.mask_alpha, 0)

        contours, _ = cv2.findContours(
            combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, style.color_bgr, style.contour_thickness)

    if blink_is_visible(style, now):
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(output, (x1, y1), (x2, y2), style.color_bgr, style.contour_thickness)

    return output
