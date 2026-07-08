from __future__ import annotations

import numpy as np

from neuromovement.vision.motion_gate import MotionGate


def _still_frame(height: int = 240, width: int = 320, value: int = 60) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _frame_with_block(height: int, width: int, box: tuple[int, int, int, int], value: int = 220) -> np.ndarray:
    frame = _still_frame(height, width)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = value
    return frame


def test_first_frame_never_reports_motion():
    gate = MotionGate()
    motion, regions = gate.detect(_still_frame())
    assert motion is False
    assert regions == []


def test_static_scene_reports_no_motion_after_warmup():
    gate = MotionGate()
    frame = _still_frame()
    for _ in range(5):
        motion, regions = gate.detect(frame)
    assert motion is False
    assert regions == []


def test_moving_block_is_detected():
    gate = MotionGate(min_area=200)
    height, width = 240, 320
    background = _still_frame(height, width)
    for _ in range(5):
        gate.detect(background)

    moving = _frame_with_block(height, width, (100, 80, 180, 160))
    motion, regions = gate.detect(moving)
    assert motion is True
    assert len(regions) >= 1
    x1, y1, x2, y2 = regions[0]
    assert x2 > x1 and y2 > y1


def test_tiny_noise_below_min_area_is_ignored():
    gate = MotionGate(min_area=5000)
    height, width = 240, 320
    background = _still_frame(height, width)
    for _ in range(5):
        gate.detect(background)

    slightly_changed = _frame_with_block(height, width, (100, 100, 110, 110))  # 10x10 = 100 px^2
    motion, regions = gate.detect(slightly_changed)
    assert motion is False
    assert regions == []


def test_reset_clears_background():
    gate = MotionGate()
    frame = _still_frame()
    gate.detect(frame)
    gate.reset()
    motion, regions = gate.detect(frame)
    assert motion is False  # снова "первый кадр" после reset
