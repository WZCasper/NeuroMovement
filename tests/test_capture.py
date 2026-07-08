from __future__ import annotations

import platform

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
    return str(path)


def test_rtsp_source_reads_all_frames(synthetic_video):
    source = RTSPSource(synthetic_video, reconnect_delay=0.05)
    frames_read = 0
    for _ in range(60):
        ok, frame = source.read()
        if ok:
            frames_read += 1
            assert frame.shape == (240, 320, 3)
    source.release()
    assert frames_read == 30


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
