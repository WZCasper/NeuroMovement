from __future__ import annotations

import sys

from neuromovement import config
from neuromovement.config import Settings


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(mask_alpha=0.66, highlight_color_name="green")
    s.save(path)
    loaded = Settings.load(path)
    assert loaded.mask_alpha == 0.66
    assert loaded.highlight_color_name == "green"


def test_weights_path_uses_meipass_when_frozen(monkeypatch, tmp_path):
    """Регрессия: в PyInstaller 6.x onedir-сборках datas распаковываются в
    подпапку _internal/, на которую указывает sys._MEIPASS, а НЕ в папку
    рядом с .exe напрямую. get_weights_path() обязан использовать именно
    _MEIPASS, иначе собранное приложение не находит веса (это реальный
    баг, воспроизведённый на настоящей сборке, а не гипотетический)."""
    fake_exe_dir = tmp_path / "AppFolder"
    fake_internal_dir = fake_exe_dir / "_internal"
    fake_internal_dir.mkdir(parents=True)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe_dir / "NeuroMovement.exe"), raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_internal_dir), raising=False)

    weights_path = config.get_weights_path()
    assert weights_path == fake_internal_dir / "weights" / "yolov8n-seg.pt"
    # settings при этом обязаны остаться ЗАПИСЫВАЕМЫМИ, рядом с .exe,
    # а не внутри _internal (иначе настройки не переживут переустановку,
    # а в onefile-режиме _MEIPASS вообще может быть недоступен на запись)
    assert config.get_settings_path() == fake_exe_dir / "neuromovement_settings.json"


def test_weights_path_dev_mode_unaffected():
    """В режиме разработки (не frozen) поведение не должно было измениться."""
    assert getattr(sys, "frozen", False) is False
    path = config.get_weights_path()
    assert path.name == "yolov8n-seg.pt"
    assert path.parent.name == "weights"
