from __future__ import annotations

import numpy as np

from neuromovement.config import Settings
from neuromovement.pipeline import process_frame
from neuromovement.vision.detector import Detection
from neuromovement.vision.motion_gate import MotionGate
from neuromovement.vision.night_vision import NightVisionProcessor


class _StubDetector:
    """Тестовый дублёр SegmentationDetector: не требует PyTorch/YOLO,
    позволяет проверить логику pipeline.process_frame изолированно."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.confidence = 0.35
        self._detections = detections or []
        self.calls = 0

    def set_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        self.calls += 1
        return list(self._detections)


def _frame(height: int = 120, width: int = 160) -> np.ndarray:
    return np.full((height, width, 3), 50, dtype=np.uint8)


def test_process_frame_without_motion_gate_always_detects():
    settings = Settings(motion_gate_enabled=False)
    detector = _StubDetector(detections=[])
    output, detections, motion = process_frame(
        _frame(),
        night_vision=NightVisionProcessor(),
        motion_gate=MotionGate(),
        detector=detector,
        settings=settings,
        now=0.0,
    )
    assert motion is True
    assert detector.calls == 1
    assert output.shape == _frame().shape


def test_process_frame_with_motion_gate_skips_detection_when_static():
    settings = Settings(motion_gate_enabled=True)
    detector = _StubDetector(detections=[])
    motion_gate = MotionGate()
    frame = _frame()

    # прогреваем фон, чтобы кадр стал "статичным"
    for _ in range(5):
        process_frame(
            frame, night_vision=NightVisionProcessor(), motion_gate=motion_gate,
            detector=detector, settings=settings, now=0.0,
        )

    calls_before = detector.calls
    output, detections, motion = process_frame(
        frame, night_vision=NightVisionProcessor(), motion_gate=motion_gate,
        detector=detector, settings=settings, now=0.0,
    )
    assert motion is False
    assert detections == []
    assert detector.calls == calls_before  # детектор не вызывался повторно на статичном кадре


def test_process_frame_filters_by_target_classes():
    person = Detection(box=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person", mask=None)
    car = Detection(box=(20, 20, 40, 40), confidence=0.9, class_id=2, class_name="car", mask=None)
    settings = Settings(motion_gate_enabled=False, target_classes=[0])
    detector = _StubDetector(detections=[person, car])

    _output, detections, _motion = process_frame(
        _frame(), night_vision=NightVisionProcessor(), motion_gate=MotionGate(),
        detector=detector, settings=settings, now=0.0,
    )
    assert len(detections) == 1
    assert detections[0].class_name == "person"


def test_process_frame_propagates_confidence_setting_to_detector():
    settings = Settings(motion_gate_enabled=False, confidence_threshold=0.77)
    detector = _StubDetector()
    process_frame(
        _frame(), night_vision=NightVisionProcessor(), motion_gate=MotionGate(),
        detector=detector, settings=settings, now=0.0,
    )
    assert detector.confidence == 0.77


def test_process_frame_output_has_alarm_colors_when_detection_present():
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[40:80, 40:100] = 1
    det = Detection(box=(40, 40, 100, 80), confidence=0.9, class_id=0, class_name="person", mask=mask)
    settings = Settings(motion_gate_enabled=False, highlight_color_name="red", blink_enabled=False)
    detector = _StubDetector(detections=[det])

    output, detections, _motion = process_frame(
        _frame(), night_vision=NightVisionProcessor(), motion_gate=MotionGate(),
        detector=detector, settings=settings, now=0.0,
    )
    pixel_in_mask = output[60, 70]
    assert pixel_in_mask[2] > pixel_in_mask[0]  # красный канал доминирует (BGR)
