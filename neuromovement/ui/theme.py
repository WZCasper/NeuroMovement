"""QSS-тема приложения: тёмный, спокойный интерфейс, который не отвлекает
и не бликует при просмотре в тёмной комнате. Тревожные цвета (красный/
зелёный/жёлтый) используются ТОЛЬКО в самой визуализации обнаружения
и не пересекаются с цветами интерфейса."""
from __future__ import annotations

DARK_THEME_QSS = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #8b949e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px 12px;
    color: #c9d1d9;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}

QPushButton:pressed {
    background-color: #161b22;
}

QLineEdit, QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 6px;
    color: #c9d1d9;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #58a6ff;
}

QComboBox::drop-down {
    border: none;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #58a6ff;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}

QCheckBox, QRadioButton {
    spacing: 8px;
    color: #c9d1d9;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
}

QStatusBar {
    background-color: #010409;
    color: #8b949e;
    border-top: 1px solid #30363d;
}

QLabel {
    color: #c9d1d9;
}
"""
