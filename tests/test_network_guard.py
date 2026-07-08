"""Тесты для network_guard: проверяем, что блокировка реально перехватывает
исходящие соединения (в т.ч. через urllib, эмулирующий поведение requests/
huggingface_hub/ultralytics), и что она снимается корректно."""
from __future__ import annotations

import socket
import urllib.request

import pytest

from neuromovement.network_guard import (
    OfflineGuardError,
    disable_offline_guard,
    enable_offline_guard,
    is_enabled,
)


@pytest.fixture(autouse=True)
def _reset_guard():
    disable_offline_guard()
    yield
    disable_offline_guard()


@pytest.fixture
def local_listener():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    yield port
    srv.close()


def test_disabled_by_default():
    assert is_enabled() is False


def test_connection_allowed_when_disabled(local_listener):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", local_listener))
    client.close()


def test_urllib_request_blocked_when_enabled():
    enable_offline_guard()
    with pytest.raises(OfflineGuardError):
        urllib.request.urlopen("https://pypi.org", timeout=5)


def test_local_socket_blocked_when_enabled(local_listener):
    enable_offline_guard()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OfflineGuardError):
        client.connect(("127.0.0.1", local_listener))


def test_guard_can_be_disabled_again(local_listener):
    enable_offline_guard()
    disable_offline_guard()
    assert is_enabled() is False
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", local_listener))
    client.close()
