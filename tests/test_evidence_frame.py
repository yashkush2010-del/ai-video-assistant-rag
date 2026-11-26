"""
The picture shown under a Q&A answer comes from the exact moment of the
evidence (src/processing/evidence_frame.py), not from the nearest of the
keyframes taken every 30 seconds.
"""

import os
import sys

import cv2
import numpy as np
import pytest

from src.processing.evidence_frame import extract_frame_at, frame_time


FPS = 10


@pytest.fixture
def video(tmp_path):
    """A 6-second video: second n is a flat grey of brightness 40 * n."""
    path = str(tmp_path / "steps.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))

    for frame in range(6 * FPS):
        writer.write(np.full((48, 64, 3), 40 * (frame // FPS), np.uint8))

    writer.release()
    return path


def brightness(path):
    return float(cv2.imread(path).mean())


@pytest.mark.parametrize("seconds, expected", [(0.5, 0), (2.5, 80), (4.5, 160)])
def test_the_frame_comes_from_the_asked_moment(video, tmp_path, seconds, expected):
    path = extract_frame_at(video, seconds, str(tmp_path / "frames"))

    assert path and os.path.exists(path)
    assert abs(brightness(path) - expected) < 12


def test_the_same_moment_is_read_only_once(video, tmp_path, monkeypatch):
    out = str(tmp_path / "frames")
    first = extract_frame_at(video, 2.5, out)

    def no_video(*args, **kwargs):
        raise AssertionError("the video was opened again")

    monkeypatch.setattr(cv2, "VideoCapture", no_video)

    assert extract_frame_at(video, 2.5, out) == first


def test_a_missing_video_gives_none_not_an_error(tmp_path):
    assert extract_frame_at(str(tmp_path / "nope.mp4"), 1.0, str(tmp_path / "f")) is None


def test_a_time_after_the_end_gives_none(video, tmp_path):
    assert extract_frame_at(video, 99.0, str(tmp_path / "f")) is None


@pytest.mark.parametrize("start, end, expected", [
    (17.6, 23.6, 18.6),     # 1 s after the start
    (10.0, 11.0, 10.5),     # short evidence: never past the middle
])
def test_frame_time(start, end, expected):
    assert frame_time(start, end) == pytest.approx(expected)


# ============================================================
# The whole app
# ============================================================

def test_the_app_shows_the_frame_of_the_evidence_moment(env, monkeypatch):
    from test_qa_ollama_unavailable import ask

    env.set_upload("lecture.mp4")
    at = env.start()

    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: {"message": {"content": "Machine learning learns from data."}},
    )

    at = ask(at, "What is machine learning?")

    assert not at.exception, [e.value for e in at.exception]

    evidence = at.session_state["last_evidence_segments"]
    picture = at.session_state["last_related_keyframe"]

    assert evidence and picture
    assert picture["exact"] is True
    assert os.path.exists(picture["image_path"])
    assert picture["timestamp"] == pytest.approx(
        frame_time(evidence[0]["start"], evidence[0]["end"])
    )

    texts = [str(m.value) for m in at.markdown]
    assert "##### 🖼️ At this moment in the video" in texts
    assert "##### 📝 Related Transcript" not in texts


from test_youtube_ingestion_app import env  # noqa: E402,F401
