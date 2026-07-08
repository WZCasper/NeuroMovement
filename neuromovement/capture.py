"""Источники видеопотока: RTSP-камера, захват целого монитора, захват
области конкретного открытого окна.

Все источники реализуют единый интерфейс FrameSource.read() -> (ok, frame),
что позволяет остальному конвейеру (pipeline.py) не зависеть от конкретного
типа источника.
"""
from __future__ import annotations

import platform
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def is_available(self) -> bool:
        return True


class RTSPSource(FrameSource):
    """Захват видео с IP-камеры по RTSP с автоматическим переподключением
    при обрыве связи. Также используется для локальных видеофайлов
    (тот же cv2.VideoCapture) — что позволяет тестировать логику конвейера
    без реальной камеры."""

    def __init__(self, url: str, reconnect_delay: float = 2.0, max_consecutive_failures: int = 15) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_consecutive_failures = max_consecutive_failures
        self._capture: cv2.VideoCapture | None = None
        self._failures = 0
        self._last_reconnect_attempt = 0.0
        self._open()

    def _open(self) -> bool:
        if self._capture is not None:
            self._capture.release()
        self._capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._last_reconnect_attempt = time.monotonic()
        opened = self._capture.isOpened()
        if opened:
            self._failures = 0
        return opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None or not self._capture.isOpened():
            if time.monotonic() - self._last_reconnect_attempt >= self.reconnect_delay:
                self._open()
            return False, None

        ok, frame = self._capture.read()
        if not ok:
            self._failures += 1
            if self._failures >= self.max_consecutive_failures:
                if time.monotonic() - self._last_reconnect_attempt >= self.reconnect_delay:
                    self._open()
            return False, None

        self._failures = 0
        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.isOpened()


class MonitorSource(FrameSource):
    """Захват изображения целого монитора через mss."""

    def __init__(self, monitor_index: int = 1) -> None:
        import mss

        self._sct = mss.MSS()
        monitors = self._sct.monitors
        if monitor_index < 1 or monitor_index >= len(monitors):
            monitor_index = 1 if len(monitors) > 1 else 0
        self.monitor_index = monitor_index
        self._monitor = monitors[self.monitor_index]

    def read(self) -> tuple[bool, np.ndarray | None]:
        shot = self._sct.grab(self._monitor)
        frame = np.array(shot)[:, :, :3]  # mss отдаёт BGRA -> берём BGR
        return True, frame

    def release(self) -> None:
        self._sct.close()


class WindowSource(FrameSource):
    """Захват экранной области, соответствующей конкретному открытому окну
    (по подстроке заголовка). Работает только в Windows: границы окна
    определяются через win32gui, а сами пиксели захватываются через mss.
    Окно должно быть видимо на экране (не свёрнуто и не перекрыто целиком)."""

    def __init__(self, window_title_substring: str) -> None:
        if platform.system() != "Windows":
            raise RuntimeError("Захват конкретного окна поддерживается только в Windows.")
        import mss

        self.window_title_substring = window_title_substring.lower()
        self._sct = mss.MSS()
        self._hwnd: int | None = None
        self._find_window()

    def _find_window(self) -> bool:
        import win32gui

        found: list[int] = []

        def _callback(hwnd: int, _extra: object) -> bool:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and self.window_title_substring in title.lower():
                    found.append(hwnd)
            return True

        win32gui.EnumWindows(_callback, None)
        self._hwnd = found[0] if found else None
        return self._hwnd is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        import win32gui

        if self._hwnd is None or not win32gui.IsWindow(self._hwnd):
            if not self._find_window():
                return False, None

        try:
            left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        except Exception:
            self._hwnd = None
            return False, None

        if right <= left or bottom <= top:
            return False, None

        region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        shot = self._sct.grab(region)
        frame = np.array(shot)[:, :, :3]
        return True, frame

    def release(self) -> None:
        self._sct.close()

    def is_available(self) -> bool:
        if self._hwnd is None:
            return False
        import win32gui

        return win32gui.IsWindow(self._hwnd)


def list_window_titles() -> list[str]:
    """Список заголовков видимых окон для выбора в UI. Только Windows."""
    if platform.system() != "Windows":
        return []
    import win32gui

    titles: list[str] = []

    def _callback(hwnd: int, _extra: object) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)
        return True

    win32gui.EnumWindows(_callback, None)
    return titles


def list_monitor_count() -> int:
    """Возвращает количество физических мониторов. При любой ошибке
    инициализации захвата экрана (нет дисплея, нет прав и т.п.) безопасно
    возвращает 1, чтобы UI не падал при построении списка источников."""
    try:
        import mss

        with mss.MSS() as sct:
            return max(len(sct.monitors) - 1, 1)
    except Exception:
        return 1


def create_source(
    source_type: str, *, rtsp_url: str = "", monitor_index: int = 1, window_title: str = ""
) -> FrameSource:
    if source_type == "rtsp":
        return RTSPSource(rtsp_url)
    if source_type == "monitor":
        return MonitorSource(monitor_index)
    if source_type == "window":
        return WindowSource(window_title)
    raise ValueError(f"Неизвестный тип источника видео: {source_type}")
