from __future__ import annotations

import numpy as np

from neuromovement.vision.detector import Detection
from neuromovement.vision.overlay import OverlayStyle, blink_is_visible, render_overlay


def _blank_frame(height: int = 100, width: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _style(**overrides) -> OverlayStyle:
    base = dict(
        color_bgr=(0, 0, 255),
        mask_alpha=0.5,
        contour_thickness=2,
        blink_enabled=True,
        blink_frequency_hz=2.5,
    )
    base.update(overrides)
    return OverlayStyle(**base)


def test_no_detections_returns_same_object_unmodified():
    frame = _blank_frame()
    result = render_overlay(frame, [], _style())
    assert result is frame  # без копирования, кадр остаётся "чистым"


def test_input_frame_not_mutated_when_detections_present():
    frame = _blank_frame()
    original = frame.copy()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1
    det = Detection(box=(30, 30, 70, 70), confidence=0.9, class_id=0, class_name="person", mask=mask)

    render_overlay(frame, [det], _style(), now=0.0)
    assert np.array_equal(frame, original)


def test_mask_fill_blends_color_with_alpha():
    frame = _blank_frame()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    det = Detection(box=(40, 40, 60, 60), confidence=0.9, class_id=0, class_name="person", mask=mask)

    style = _style(color_bgr=(0, 0, 255), mask_alpha=0.5, blink_enabled=False)
    result = render_overlay(frame, [det], style, now=0.0)

    inside = result[50, 50]
    outside = result[5, 5]
    assert np.array_equal(outside, [0, 0, 0])  # вне маски кадр остался чёрным
    assert inside[2] > inside[0]  # внутри маски выражен красный канал (BGR: индекс 2 = R)
    assert 100 < inside[2] < 200  # ~alpha 0.5 от 255, не полностью залито


def test_blink_hides_box_in_off_phase():
    frame = _blank_frame()
    det = Detection(box=(10, 10, 90, 90), confidence=0.9, class_id=0, class_name="person", mask=None)
    style = _style(blink_enabled=True, blink_frequency_hz=1.0)  # период 1с, видим первые 0.5с

    visible = render_overlay(frame, [det], style, now=0.1)
    hidden = render_overlay(frame, [det], style, now=0.6)

    assert not np.array_equal(visible, frame)  # рамка нарисована
    assert np.array_equal(hidden, frame)  # рамка скрыта — кадр чист


def test_blink_disabled_always_visible():
    style = _style(blink_enabled=False, blink_frequency_hz=2.5)
    assert blink_is_visible(style, 0.0) is True
    assert blink_is_visible(style, 123.456) is True


def test_blink_phase_matches_frequency():
    style = _style(blink_enabled=True, blink_frequency_hz=2.0)  # период 0.5с
    assert blink_is_visible(style, 0.0) is True
    assert blink_is_visible(style, 0.24) is True
    assert blink_is_visible(style, 0.26) is False
    assert blink_is_visible(style, 0.49) is False
    assert blink_is_visible(style, 0.51) is True  # следующий цикл
