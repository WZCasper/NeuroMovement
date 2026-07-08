"""Панель настроек: выбор источника видео и параметры тревожной
визуализации (цвет, пульсация, толщина обводки и дополнительные
параметры точности/производительности)."""
from __future__ import annotations

import platform

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from neuromovement.capture import list_monitor_count, list_window_titles
from neuromovement.config import Settings

_COLOR_LABELS: dict[str, str] = {
    "red": "Красный",
    "green": "Кислотно-зелёный",
    "yellow": "Жёлтый",
}


class SettingsPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setFixedWidth(320)
        self._running = False

        root = QVBoxLayout(self)
        root.addWidget(self._build_source_group())
        root.addWidget(self._build_alarm_group())
        root.addWidget(self._build_advanced_group())

        self.toggle_button = QPushButton("Запустить")
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.clicked.connect(self._on_toggle_clicked)
        root.addWidget(self.toggle_button)
        root.addStretch(1)

    # ---------------------------------------------------------------- Источник
    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Источник видео")
        layout = QVBoxLayout(group)

        self.source_combo = QComboBox()
        self.source_combo.addItem("RTSP-камера", "rtsp")
        self.source_combo.addItem("Монитор", "monitor")
        if platform.system() == "Windows":
            self.source_combo.addItem("Окно приложения", "window")
        index_map = {"rtsp": 0, "monitor": 1, "window": 2}
        start_index = index_map.get(self.settings.source_type, 0)
        self.source_combo.setCurrentIndex(min(start_index, self.source_combo.count() - 1))
        self.source_combo.currentIndexChanged.connect(self._on_source_type_changed)
        layout.addWidget(self.source_combo)

        self.rtsp_edit = QLineEdit(self.settings.rtsp_url)
        self.rtsp_edit.setPlaceholderText("rtsp://192.168.1.100:554/stream1")
        self.rtsp_edit.editingFinished.connect(self._on_rtsp_changed)
        layout.addWidget(self.rtsp_edit)

        self.monitor_combo = QComboBox()
        for i in range(1, list_monitor_count() + 1):
            self.monitor_combo.addItem(f"Монитор {i}", i)
        if self.monitor_combo.count() == 0:
            self.monitor_combo.addItem("Монитор 1", 1)
        self.monitor_combo.currentIndexChanged.connect(self._on_monitor_changed)
        layout.addWidget(self.monitor_combo)

        self.window_combo = QComboBox()
        self.window_refresh_button = QPushButton("Обновить список окон")
        self.window_refresh_button.clicked.connect(self._refresh_windows)
        layout.addWidget(self.window_combo)
        layout.addWidget(self.window_refresh_button)
        self.window_combo.currentTextChanged.connect(self._on_window_changed)

        self._apply_source_visibility()
        return group

    def _refresh_windows(self) -> None:
        self.window_combo.clear()
        for title in list_window_titles():
            self.window_combo.addItem(title)

    def _apply_source_visibility(self) -> None:
        source_type = self.source_combo.currentData()
        self.rtsp_edit.setVisible(source_type == "rtsp")
        self.monitor_combo.setVisible(source_type == "monitor")
        is_window = source_type == "window"
        self.window_combo.setVisible(is_window)
        self.window_refresh_button.setVisible(is_window)

    def _on_source_type_changed(self) -> None:
        self.settings.source_type = self.source_combo.currentData()
        self._apply_source_visibility()
        self.settings_changed.emit()

    def _on_rtsp_changed(self) -> None:
        self.settings.rtsp_url = self.rtsp_edit.text().strip()
        self.settings_changed.emit()

    def _on_monitor_changed(self) -> None:
        self.settings.monitor_index = self.monitor_combo.currentData() or 1
        self.settings_changed.emit()

    def _on_window_changed(self, text: str) -> None:
        self.settings.window_title = text
        self.settings_changed.emit()

    # ------------------------------------------------------------------ Тревога
    def _build_alarm_group(self) -> QGroupBox:
        group = QGroupBox("Визуализация тревоги")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Цвет подсветки"))
        self.color_group = QButtonGroup(self)
        for key, label in _COLOR_LABELS.items():
            radio = QRadioButton(label)
            radio.setChecked(self.settings.highlight_color_name == key)
            radio.toggled.connect(lambda checked, k=key: self._on_color_changed(k, checked))
            self.color_group.addButton(radio)
            layout.addWidget(radio)

        self.blink_checkbox = QCheckBox("Пульсация рамки")
        self.blink_checkbox.setChecked(self.settings.blink_enabled)
        self.blink_checkbox.toggled.connect(self._on_blink_toggled)
        layout.addWidget(self.blink_checkbox)

        form = QFormLayout()
        self.thickness_slider = _make_slider(1, 10, self.settings.contour_thickness, self._on_thickness_changed)
        form.addRow("Толщина обводки", self.thickness_slider)
        layout.addLayout(form)

        return group

    def _on_color_changed(self, key: str, checked: bool) -> None:
        if checked:
            self.settings.highlight_color_name = key
            self.settings_changed.emit()

    def _on_blink_toggled(self, checked: bool) -> None:
        self.settings.blink_enabled = checked
        self.settings_changed.emit()

    def _on_thickness_changed(self, value: int) -> None:
        self.settings.contour_thickness = value
        self.settings_changed.emit()

    # --------------------------------------------------------------- Дополнительно
    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Дополнительно")
        form = QFormLayout(group)

        self.alpha_slider = _make_slider(
            10, 90, int(self.settings.mask_alpha * 100), self._on_alpha_changed
        )
        form.addRow("Прозрачность заливки", self.alpha_slider)

        self.confidence_slider = _make_slider(
            10, 90, int(self.settings.confidence_threshold * 100), self._on_confidence_changed
        )
        form.addRow("Порог уверенности", self.confidence_slider)

        self.motion_gate_checkbox = QCheckBox("Экономия ресурсов (реагировать только на движение)")
        self.motion_gate_checkbox.setChecked(self.settings.motion_gate_enabled)
        self.motion_gate_checkbox.toggled.connect(self._on_motion_gate_toggled)
        form.addRow(self.motion_gate_checkbox)

        return group

    def _on_alpha_changed(self, value: int) -> None:
        self.settings.mask_alpha = value / 100.0
        self.settings_changed.emit()

    def _on_confidence_changed(self, value: int) -> None:
        self.settings.confidence_threshold = value / 100.0
        self.settings_changed.emit()

    def _on_motion_gate_toggled(self, checked: bool) -> None:
        self.settings.motion_gate_enabled = checked
        self.settings_changed.emit()

    # -------------------------------------------------------------------- Общее
    def _on_toggle_clicked(self) -> None:
        if self._running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.toggle_button.setText("Остановить" if running else "Запустить")
        for widget in (
            self.source_combo,
            self.rtsp_edit,
            self.monitor_combo,
            self.window_combo,
            self.window_refresh_button,
        ):
            widget.setEnabled(not running)


def _make_slider(minimum: int, maximum: int, value: int, callback) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    slider.valueChanged.connect(callback)
    return slider
