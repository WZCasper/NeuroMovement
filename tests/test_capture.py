from __future__ import annotations

import platform
import time

import cv2
import numpy as np
import pytest

from neuromovement.capture import RTSPSource, WindowSource, create_source, list_window_titles


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        x = 20 + i * 5
        cv2.rectangle(frame, (x, 100), (x + 40, 160), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    del writer  # явно освобождаем файловый хендл (важно на Windows)
    time.sleep(0.2)  # даём ОС время снять блокировку файла перед чтением
    return str(path)


def test_rtsp_source_reads_all_frames(synthetic_video):
    # max_consecutive_failures специально завышен: этот тест проверяет
    # чтение реальных кадров конечного файла, а не логику переподключения
    # (переподключение для живого RTSP-потока — отдельная задача, ниже).
    source = RTSPSource(synthetic_video, reconnect_delay=0.05, max_consecutive_failures=10_000)
    frames_read = 0
    for _ in range(45):
        ok, frame = source.read()
        if ok:
            frames_read += 1
            assert frame.shape == (240, 320, 3)
    source.release()
    # Точное число кадров зависит от кодека/платформы (некоторые сборки
    # FFmpeg на Windows могут на 1-2 кадра отличаться от заданных 30)
    # — поэтому проверяем диапазон, а не точное равенство.
    assert 27 <= frames_read <= 30


def test_rtsp_source_reconnects_after_repeated_failures(synthetic_video):
    # А здесь наоборот: специально проверяем, что после исчерпания файла
    # (много неудачных чтений подряд) источник пытается переоткрыться —
    # именно так ведёт себя переподключение к реальной RTSP-камере после
    # обрыва связи.
    source = RTSPSource(synthetic_video, reconnect_delay=0.01, max_consecutive_failures=5)
    for _ in range(30):
        source.read()  # вычитываем все настоящие кадры файла
    # теперь несколько чтений подряд должны провалиться (конец файла)
    # и в итоге вызвать переоткрытие источника
    reopened = False
    for _ in range(20):
        ok, _frame = source.read()
        time.sleep(0.02)
        if ok:
            reopened = True
            break
    source.release()
    assert reopened, "источник должен переоткрыться и снова начать отдавать кадры"


def test_rtsp_source_invalid_url_reports_unavailable():
    source = RTSPSource("/nonexistent/path/to/video.mp4", reconnect_delay=0.05)
    ok, frame = source.read()
    assert ok is False
    assert frame is None
    source.release()


def test_window_source_unsupported_on_non_windows():
    if platform.system() == "Windows":
        pytest.skip("тест применим только к не-Windows платформам")
    with pytest.raises(RuntimeError):
        WindowSource("anything")


def test_list_window_titles_empty_on_non_windows():
    if platform.system() == "Windows":
        pytest.skip("тест применим только к не-Windows платформам")
    assert list_window_titles() == []


def test_create_source_rejects_unknown_type():
    with pytest.raises(ValueError):
        create_source("not-a-real-source")


def test_create_source_rtsp(synthetic_video):
    source = create_source("rtsp", rtsp_url=synthetic_video)
    ok, frame = source.read()
    assert ok is True
    source.release()
