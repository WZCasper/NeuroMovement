"""Принудительная блокировка исходящих сетевых соединений на уровне Python.

Зачем это нужно
---------------
Библиотеки ultralytics и huggingface_hub умеют обращаться к своим серверам
(проверка обновлений, телеметрия, автоскачивание файлов). Эти обращения
выполняются через библиотеку requests -> urllib3 -> стандартный модуль
socket. Функция enable_offline_guard() подменяет класс socket.socket на
версию, которая поднимает исключение при любой попытке .connect(), — это
перекрывает ВСЕ Python-библиотеки, использующие стандартный сетевой стек
(requests, urllib, http.client, huggingface_hub и т.д.), независимо от
порядка импортов, так как socket.create_connection() всегда обращается к
классу socket.socket через глобальное пространство имён модуля socket
в момент вызова, а не через сохранённую заранее ссылку.

Почему это НЕ ломает RTSP-камеры
---------------------------------
Захват RTSP-потока в OpenCV (cv2.VideoCapture) выполняется через нативную
библиотеку FFmpeg, написанную на C. FFmpeg держит собственный сетевой стек
и открывает соединения напрямую через системные вызовы ОС, минуя модуль
socket интерпретатора Python. Поэтому подключение к камерам в локальной
сети продолжает работать даже при включённой блокировке — блокируется
только сетевой трафик, инициированный из кода на Python.
"""
from __future__ import annotations

import socket

_ORIGINAL_SOCKET_CLASS = socket.socket
_enabled = False


class OfflineGuardError(RuntimeError):
    """Возникает при попытке установить исходящее сетевое соединение
    из Python-кода, пока включён офлайн-режим."""


class _BlockedSocket(_ORIGINAL_SOCKET_CLASS):
    def connect(self, *args: object, **kwargs: object) -> None:
        raise OfflineGuardError(
            "NeuroMovement работает в строго оффлайн-режиме: "
            "попытка сетевого подключения из Python-кода заблокирована."
        )

    def connect_ex(self, *args: object, **kwargs: object) -> int:
        raise OfflineGuardError(
            "NeuroMovement работает в строго оффлайн-режиме: "
            "попытка сетевого подключения из Python-кода заблокирована."
        )


def enable_offline_guard() -> None:
    """Включает блокировку. Должна вызываться один раз при старте приложения,
    как можно раньше (до первого использования detector/ultralytics)."""
    global _enabled
    socket.socket = _BlockedSocket  # type: ignore[misc]
    _enabled = True


def disable_offline_guard() -> None:
    """Отключает блокировку. Используется только в автотестах."""
    global _enabled
    socket.socket = _ORIGINAL_SOCKET_CLASS  # type: ignore[misc]
    _enabled = False


def is_enabled() -> bool:
    return _enabled
