from __future__ import annotations

import numpy as np

from neuromovement.vision.night_vision import NightVisionProcessor


def _dark_frame(height: int = 240, width: int = 320) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.integers(5, 25, size=(height, width, 3), dtype=np.uint8)  # тёмный кадр
    base[100:140, 100:180] = rng.integers(15, 35, size=(40, 80, 3), dtype=np.uint8)  # чуть светлее "силуэт"
    return base


def test_enhance_preserves_shape_and_dtype():
    frame = _dark_frame()
    processor = NightVisionProcessor()
    enhanced = processor.enhance(frame)
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == frame.dtype


def test_enhance_does_not_mutate_input():
    frame = _dark_frame()
    original = frame.copy()
    processor = NightVisionProcessor()
    processor.enhance(frame)
    assert np.array_equal(frame, original)


def test_enhance_increases_contrast_on_dark_frame():
    frame = _dark_frame()
    processor = NightVisionProcessor(clip_limit=4.0)
    enhanced = processor.enhance(frame)
    assert np.std(enhanced) > np.std(frame)


def test_set_clip_limit_changes_behaviour():
    frame = _dark_frame()
    low = NightVisionProcessor(clip_limit=1.0).enhance(frame)
    proc = NightVisionProcessor(clip_limit=1.0)
    proc.set_clip_limit(8.0)
    high = proc.enhance(frame)
    assert not np.array_equal(low, high)


def test_estimate_brightness_range():
    frame = _dark_frame()
    brightness = NightVisionProcessor.estimate_brightness(frame)
    assert 0.0 <= brightness <= 255.0
