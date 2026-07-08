"""Обёртка над YOLOv8-seg для локального, полностью оффлайн инференса.

Модель загружается ИСКЛЮЧИТЕЛЬНО по локальному пути к файлу весов.
Библиотека ultralytics не пытается скачивать файл, если он уже
физически существует по переданному пути — сетевой запрос в её
внутренней логике происходит только тогда, когда путь не найден на
диске и совпадает с одним из имён предустановленных моделей. Кроме
этого, network_guard.enable_offline_guard() (см. соответствующий модуль)
дополнительно блокирует любые попытки сетевых обращений на уровне
Python "для перестраховки" (defense in depth).

Тяжёлые импорты (torch, ultralytics) выполняются лениво — внутри
__init__ — чтобы простой импорт этого модуля (например, ради класса
Detection в overlay.py) не требовал наличия PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 в пикселях исходного кадра
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None  # бинарная маска (uint8, 0/1) размером с кадр, либо None


class SegmentationDetector:
    def __init__(self, weights_path: Path, device: str = "auto", confidence: float = 0.35) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Файл весов модели не найден: {weights_path}\n"
                "Поместите yolov8n-seg.pt в папку weights/ рядом с приложением "
                "(см. weights/README.md или запустите scripts/fetch_weights.py)."
            )

        import torch
        from ultralytics import YOLO
        from ultralytics import settings as yolo_settings

        # Доп. слой защиты сверх network_guard: явно отключаем встроенную
        # телеметрию/облачную синхронизацию Ultralytics на уровне её
        # собственного файла настроек (~/.config/Ultralytics/settings.json
        # в Linux, %APPDATA%/Ultralytics/settings.json в Windows). Даже если
        # бы этот файл был установлен в "true" на чужой машине, конкретно
        # сетевые вызовы всё равно физически заблокированы network_guard'ом.
        yolo_settings.update(
            {
                "sync": False,
                "hub": False,
                "clearml": False,
                "comet": False,
                "dvc": False,
                "mlflow": False,
                "neptune": False,
                "raytune": False,
                "wandb": False,
            }
        )

        self.device = self._resolve_device(device, torch)
        self.confidence = confidence
        self._model = YOLO(str(weights_path))
        self._model.to(self.device)

    @staticmethod
    def _resolve_device(preference: str, torch_module) -> str:
        if preference == "cpu":
            return "cpu"
        if preference == "cuda":
            if torch_module.cuda.is_available():
                return "cuda:0"
            raise RuntimeError("Запрошен режим CUDA, но совместимая NVIDIA GPU не найдена.")
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"

    def set_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            source=frame_bgr,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        masks_data = None
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()  # (N, Hm, Wm), значения 0..1

        frame_h, frame_w = frame_bgr.shape[:2]

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            mask_resized = None
            if masks_data is not None and i < len(masks_data):
                resized = cv2.resize(masks_data[i], (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = (resized > 0.5).astype(np.uint8)
            detections.append(
                Detection(
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(confidences[i]),
                    class_id=int(class_ids[i]),
                    class_name=str(names.get(int(class_ids[i]), "object")),
                    mask=mask_resized,
                )
            )
        return detections
