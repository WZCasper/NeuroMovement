"""Настройки приложения: структура данных и хранение в JSON-файле рядом с программой.

JSON выбран вместо системного реестра Windows (QSettings), чтобы приложение
оставалось полностью портативным: настройки лежат в одной папке со сборкой
и переносятся вместе с ней на другой компьютер.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceType = Literal["rtsp", "monitor", "window"]
DevicePreference = Literal["auto", "cpu", "cuda"]

# Цветовые пресеты хранятся в RGB (естественный порядок для Qt/цветовых пикеров).
# Перевод в BGR для OpenCV выполняется методом Settings.highlight_color_bgr().
HIGHLIGHT_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (57, 255, 20),  # кислотно-зелёный (electric lime)
    "yellow": (255, 255, 0),
}

COCO_PERSON_CLASS_ID = 0


def get_app_dir() -> Path:
    """Каталог приложения: рядом с .exe при сборке PyInstaller, иначе — корень репозитория."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_settings_path() -> Path:
    return get_app_dir() / "neuromovement_settings.json"


def get_weights_path() -> Path:
    return get_app_dir() / "weights" / "yolov8n-seg.pt"


@dataclass
class Settings:
    # --- Источник видео ---
    source_type: SourceType = "rtsp"
    rtsp_url: str = "rtsp://192.168.1.100:554/stream1"
    monitor_index: int = 1
    window_title: str = ""

    # --- Визуализация тревоги ---
    highlight_color_name: str = "red"
    blink_enabled: bool = True
    blink_frequency_hz: float = 2.5
    contour_thickness: int = 3
    mask_alpha: float = 0.5

    # --- Детекция ---
    confidence_threshold: float = 0.35
    target_classes: list[int] = field(default_factory=lambda: [COCO_PERSON_CLASS_ID])
    device_preference: DevicePreference = "auto"

    # --- Оптимизация ---
    motion_gate_enabled: bool = True
    motion_min_area: int = 900

    def highlight_color_rgb(self) -> tuple[int, int, int]:
        return HIGHLIGHT_COLORS.get(self.highlight_color_name, HIGHLIGHT_COLORS["red"])

    def highlight_color_bgr(self) -> tuple[int, int, int]:
        r, g, b = self.highlight_color_rgb()
        return (b, g, r)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or get_settings_path()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return cls()
            known_fields = set(cls.__dataclass_fields__.keys())
            filtered = {key: value for key, value in raw.items() if key in known_fields}
            try:
                return cls(**filtered)
            except TypeError:
                return cls()
        return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or get_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
