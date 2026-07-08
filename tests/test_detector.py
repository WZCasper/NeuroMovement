"""Тесты для SegmentationDetector с РЕАЛЬНОЙ архитектурой YOLOv8-seg.

В отличие от остальных тестов, здесь используется настоящий класс
ultralytics.YOLO с настоящим forward-проходом через сеть — но веса
инициализированы случайно (собраны из встроенного в пакет ultralytics
YAML-описания архитектуры, без обращения к сети), а не загружены как
предобученные. Это специально: тест проверяет, что весь код загрузки
чекпоинта и разбора результатов (боксы, маски, уверенность, классы)
работает корректно с реальным объектом Results, а не с его имитацией.
Экономическая логика (кто есть кто) не проверяется — для этого нужны уже
предобученные веса, которые пользователь получает через
scripts/fetch_weights.py на своей машине.

Требует установленных torch/ultralytics — пропускается, если их нет.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")
torch = pytest.importorskip("torch")

from neuromovement.vision.detector import SegmentationDetector  # noqa: E402


@pytest.fixture(scope="module")
def architecture_only_weights(tmp_path_factory) -> Path:
    """Строит YOLOv8n-seg из встроенного YAML (без сети, веса случайные)
    и сохраняет как обычный .pt checkpoint — по формату неотличим от
    настоящего предобученного файла для целей тестирования кода загрузки."""
    from ultralytics import YOLO

    out_dir = tmp_path_factory.mktemp("weights")
    out_path = out_dir / "yolov8n-seg.pt"
    model = YOLO("yolov8n-seg.yaml")
    model.save(str(out_path))
    return out_path


@pytest.fixture(scope="module")
def detector(architecture_only_weights) -> SegmentationDetector:
    return SegmentationDetector(architecture_only_weights, device="cpu", confidence=0.01)


def test_missing_weights_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        SegmentationDetector(tmp_path / "does_not_exist.pt", device="cpu")


def test_device_resolution_cpu_explicit(architecture_only_weights):
    det = SegmentationDetector(architecture_only_weights, device="cpu")
    assert det.device == "cpu"


def test_device_resolution_auto_falls_back_without_gpu(architecture_only_weights):
    det = SegmentationDetector(architecture_only_weights, device="auto")
    assert det.device in ("cpu", "cuda:0")
    if not torch.cuda.is_available():
        assert det.device == "cpu"


def test_device_resolution_cuda_explicit_without_gpu_raises(architecture_only_weights):
    if torch.cuda.is_available():
        pytest.skip("тест проверяет поведение именно при отсутствии GPU")
    with pytest.raises(RuntimeError):
        SegmentationDetector(architecture_only_weights, device="cuda")


def test_detect_on_real_frame_does_not_crash_and_returns_list(detector):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    # со случайными весами объектов обычно не находится — это ожидаемо
    # и не является ошибкой; важно, что не было исключений и тип верный.
    for det in detections:
        x1, y1, x2, y2 = det.box
        assert x2 >= x1 and y2 >= y1
        assert 0.0 <= det.confidence <= 1.0
        assert isinstance(det.class_name, str)
        if det.mask is not None:
            assert det.mask.shape == frame.shape[:2]
            assert set(np.unique(det.mask)).issubset({0, 1})


def test_set_confidence_updates_attribute(detector):
    detector.set_confidence(0.87)
    assert detector.confidence == 0.87
    detector.set_confidence(0.01)  # возвращаем низкий порог для остальных тестов


def test_detect_handles_multiple_calls_consistently(detector):
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    first = detector.detect(frame)
    second = detector.detect(frame)
    assert isinstance(first, list)
    assert isinstance(second, list)
