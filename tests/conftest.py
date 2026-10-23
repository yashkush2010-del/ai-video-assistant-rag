"""Shared fixtures for the YouTube tests."""

import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    """
    A real 4-second H.264/AAC mp4 (640x360, tone + test picture).
    It lets the tests run the real ffmpeg / OpenCV parts of the pipeline.
    """
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    path = tmp_path_factory.mktemp("media") / "tiny.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )

    return path


@pytest.fixture(scope="session")
def vp9_video(tmp_path_factory):
    """A real 720p VP9/Opus webm: what yt-dlp can deliver instead of H.264."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    path = tmp_path_factory.mktemp("media") / "vp9.webm"

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=1280x720:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
            "-c:a", "libopus", "-shortest", str(path),
        ],
        capture_output=True,
    )

    if result.returncode != 0:
        pytest.skip("this ffmpeg cannot encode VP9")

    return path


@pytest.fixture(scope="session")
def other_video(tmp_path_factory):
    """A second real mp4 with different content (so its hash differs)."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    path = tmp_path_factory.mktemp("media") / "other.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=duration=3:size=480x270:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
    )

    return path