"""
The whole app, with a real 4-second video, real ffmpeg audio extraction,
real OpenCV keyframes and the real FAISS index. Only these are stand-ins:

    Whisper / MiniLM / Ollama    no model is loaded
    yt-dlp                       "downloads" by copying the test video

These tests show that

  - a local upload still goes through the pipeline as before, and
  - a YouTube video is put on the disk by the new ingestion layer and then
    handed to the SAME pipeline calls (extract_audio -> Whisper ->
    keyframes -> embeddings -> index -> summary).
"""

import os
import re
import sys
import types
from pathlib import Path

import pytest

import youtube_fakes as fakes

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"

VIDEO_ID = "aqz-KE-bpKQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"

SEGMENTS = [
    {"start": 0.0, "end": 2.0,
     "text": " Machine learning lets computers learn patterns from data."},
    {"start": 2.0, "end": 4.0,
     "text": " Supervised learning uses labelled examples to train a model."},
]


class FakeUpload:
    """What st.file_uploader returns: name, size and getvalue()."""

    def __init__(self, name, data):
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self):
        return self._data


class Env:
    """Everything a test needs: the app, the stand-ins and what they saw."""

    def __init__(self, tmp_path, tiny_video, other_video, monkeypatch):
        self.tmp = tmp_path
        self.video_bytes = Path(tiny_video).read_bytes()
        self.tiny_video = Path(tiny_video)
        self.other_video = Path(other_video)
        self.monkeypatch = monkeypatch

        self.upload = {"file": None}

        self.calls = {
            "extract_audio": [],
            "transcribe": [],
            "extract_keyframes": [],
        }

        self.ydl_log = None

    # ---- the app --------------------------------------------------
    def new_app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(APP), default_timeout=90)

    def start(self):
        at = self.new_app()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    def process_youtube(self, at, url=URL):
        at.text_input(key="youtube_url").set_value(url)
        at.button(key="process_youtube").click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    # ---- yt-dlp ---------------------------------------------------
    def fake_youtube(self, different_video=False, **kwargs):
        """different_video=True: the "download" has other content than the upload."""
        source = self.other_video if different_video else self.tiny_video
        factory, self.ydl_log = fakes.make_fake_ydl(source, **kwargs)
        self.monkeypatch.setitem(
            sys.modules, "yt_dlp", fakes.fake_yt_dlp_module(factory)
        )
        return self.ydl_log

    # ---- the upload -----------------------------------------------
    def set_upload(self, name="lecture.mp4"):
        self.upload["file"] = FakeUpload(name, self.video_bytes)

    # ---- paths ----------------------------------------------------
    def path(self, *parts):
        return self.tmp.joinpath(*parts)

    def files(self, *parts):
        return fakes.files_in(str(self.path(*parts)))


@pytest.fixture
def env(tmp_path, monkeypatch, tiny_video, other_video):
    pytest.importorskip("streamlit.testing.v1")

    import streamlit as st

    environment = Env(tmp_path, tiny_video, other_video, monkeypatch)
    calls = environment.calls

    # ---- no model is loaded -----------------------------------------
    class Whisper:
        def transcribe(self, audio_path, *args, **kwargs):
            calls["transcribe"].append(audio_path)
            # the audio must really exist when Whisper is asked to read it
            assert os.path.getsize(audio_path) > 1000
            return {"segments": [dict(s) for s in SEGMENTS]}

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda *a, **k: Whisper()

    class SentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, normalize_embeddings=False, **kwargs):
            import hashlib

            import numpy as np

            single = isinstance(texts, str)
            items = [texts] if single else list(texts)
            vectors = np.zeros((len(items), 384), dtype="float32")

            for row, text in enumerate(items):
                for word in re.findall(r"[a-z']+", text.lower()):
                    column = int(hashlib.md5(word.encode()).hexdigest(), 16) % 384
                    vectors[row, column] += 1.0

            if normalize_embeddings:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms

            return vectors[0] if single else vectors

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = SentenceTransformer

    ollama = types.ModuleType("ollama")
    ollama.chat = lambda *a, **k: {"message": {"content": ""}}

    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "ollama", ollama)
    monkeypatch.delitem(sys.modules, "src.rag.answer_generator", raising=False)

    # ---- spies on the existing pipeline (they call the real code) ----
    from src.ingestion import audio_extractor
    from src.processing import keyframe_extractor

    real_audio = audio_extractor.extract_audio
    real_frames = keyframe_extractor.extract_keyframes

    def spy_audio(*args, **kwargs):
        calls["extract_audio"].append(args[0] if args else kwargs["video_path"])
        return real_audio(*args, **kwargs)

    def spy_frames(*args, **kwargs):
        calls["extract_keyframes"].append(args[0] if args else kwargs["video_path"])
        return real_frames(*args, **kwargs)

    monkeypatch.setattr(audio_extractor, "extract_audio", spy_audio)
    monkeypatch.setattr(keyframe_extractor, "extract_keyframes", spy_frames)

    # ---- the uploader returns whatever the test puts in env.upload ----
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: environment.upload["file"])

    # ---- run in an empty folder: data/... is created there -------------
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT))

    st.cache_resource.clear()

    yield environment

    st.cache_resource.clear()


def page_text(at):
    parts = []

    for group in (at.markdown, at.error, at.success, at.info, at.warning, at.caption, at.text):
        parts += [str(element.value) for element in group]

    return "\n".join(parts)


def pipeline_ran(at):
    return "Video processing completed!" in page_text(at)


# ============================================================
# a local upload still works
# ============================================================

def test_a_local_upload_still_goes_through_the_whole_pipeline(env):
    env.set_upload("lecture.mp4")

    at = env.start()

    saved = os.path.join("data", "videos", "lecture.mp4")

    assert pipeline_ran(at)
    assert "Video uploaded successfully!" in page_text(at)

    # saved under data/videos, exactly as before
    assert at.session_state["video_path"] == saved
    assert at.session_state["video_name"] == "lecture"
    assert env.files("data", "videos") == ["lecture.mp4"]
    assert env.path(saved).read_bytes() == env.video_bytes

    # and every existing stage ran on that file
    assert env.calls["extract_audio"] == [saved]
    assert env.calls["extract_keyframes"] == [saved]
    assert len(env.calls["transcribe"]) == 1

    assert at.session_state["index"] is not None
    assert len(at.session_state["chunks"]) >= 1

    # nothing of YouTube is involved
    assert not env.path("data", "youtube").exists()
    assert at.session_state["youtube_source"] is None
    assert at.session_state["active_source"] == "upload"


def test_an_uploaded_video_is_not_processed_again_on_the_next_interaction(env):
    env.set_upload()

    at = env.start()
    at.run()

    assert len(env.calls["transcribe"]) == 1


# ============================================================
# the YouTube video enters the same pipeline
# ============================================================

def test_the_downloaded_file_is_passed_into_the_same_pipeline(env):
    log = env.fake_youtube()
    at = env.start()

    assert not pipeline_ran(at)

    env.process_youtube(at)

    cached = os.path.join("data", "youtube", f"{VIDEO_ID}.mp4")

    assert pipeline_ran(at)
    assert "YouTube video downloaded successfully!" in page_text(at)

    # yt-dlp was asked for this one video, once
    assert log["download"] == [[URL]]

    # the downloaded file IS the pipeline's video: no copy anywhere else
    assert at.session_state["video_path"] == cached
    assert at.session_state["video_name"] == f"youtube_{VIDEO_ID}"
    assert env.files("data", "youtube") == [f"{VIDEO_ID}.json", f"{VIDEO_ID}.mp4"]
    assert env.files("data", "videos") == []

    # the existing stages ran on that exact file
    assert env.calls["extract_audio"] == [cached]
    assert env.calls["extract_keyframes"] == [cached]
    assert len(env.calls["transcribe"]) == 1

    # ... and produced the same state an upload produces
    assert at.session_state["video_data"] == env.path(cached).read_bytes()
    assert at.session_state["index"] is not None
    assert len(at.session_state["chunks"]) >= 1
    assert at.session_state["multimodal_records"] is not None

    # the video summary is built from it too
    assert any("Video Summary" in str(e.label) for e in at.expander)

    assert at.session_state["active_source"] == "youtube"


def test_the_pipeline_receives_the_same_kind_of_input_for_upload_and_youtube(env):
    """Same calls, same arguments' shape: a file path in data/..., then the audio."""
    env.set_upload("lecture.mp4")
    env.start()
    upload_calls = {k: list(v) for k, v in env.calls.items()}

    for values in env.calls.values():
        values.clear()

    env.upload["file"] = None
    env.fake_youtube()

    at = env.start()
    env.process_youtube(at)

    for stage in ("extract_audio", "extract_keyframes", "transcribe"):
        assert len(env.calls[stage]) == len(upload_calls[stage]) == 1

    assert env.calls["extract_audio"][0].endswith(".mp4")
    assert upload_calls["extract_audio"][0].endswith(".mp4")


# ============================================================
# cache
# ============================================================

def test_processing_the_same_video_again_in_the_session_does_nothing_new(env):
    log = env.fake_youtube()
    at = env.start()

    env.process_youtube(at)
    env.process_youtube(at)

    assert len(log["download"]) == 1
    assert len(env.calls["transcribe"]) == 1


def test_a_new_session_reuses_the_cached_file_without_downloading(env):
    log = env.fake_youtube()

    first = env.start()
    env.process_youtube(first)

    second = env.start()
    env.process_youtube(second)

    # no second download, and the message says why
    assert len(log["download"]) == 1
    assert "found in the cache" in page_text(second)
    assert pipeline_ran(second)

    assert second.session_state["video_path"] == os.path.join(
        "data", "youtube", f"{VIDEO_ID}.mp4"
    )


def test_a_different_link_to_a_cached_video_does_not_download(env):
    log = env.fake_youtube()
    at = env.start()

    env.process_youtube(at)

    other_session = env.start()
    env.process_youtube(other_session, f"https://youtu.be/{VIDEO_ID}?t=30")

    assert len(log["download"]) == 1


# ============================================================
# errors never break the app
# ============================================================

def test_an_invalid_url_shows_a_message_and_nothing_runs(env):
    log = env.fake_youtube()
    at = env.start()

    env.process_youtube(at, "https://vimeo.com/12345")

    assert any("YouTube video link" in str(e.value) for e in at.error)
    assert not pipeline_ran(at)
    assert log["options"] == []
    assert not env.path("data", "youtube").exists()
    assert env.calls["transcribe"] == []
    assert at.session_state["video_hash"] is None


def test_an_empty_box_asks_for_a_link(env):
    env.fake_youtube()
    at = env.start()

    env.process_youtube(at, "")

    assert any("Paste a YouTube link" in str(e.value) for e in at.error)
    assert not pipeline_ran(at)


def test_a_failed_download_shows_a_readable_error_and_nothing_runs(env):
    env.fake_youtube(
        download_error=fakes.DownloadError(
            "ERROR: unable to download video data: HTTP Error 403: Forbidden"
        ),
        leave_partial_file=True,
    )
    at = env.start()

    env.process_youtube(at)

    text = page_text(at)

    assert any("refused the download" in str(e.value) for e in at.error)
    assert "HTTP Error 403" not in text
    assert "ERROR:" not in text
    assert not pipeline_ran(at)
    assert env.calls["transcribe"] == []
    assert env.files("data", "youtube") == []


@pytest.mark.parametrize("message, expected", [
    ("ERROR: [youtube] aqz-KE-bpKQ: Private video. Sign in if you've been granted access to this video", "private"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Video unavailable", "unavailable"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Sign in to confirm your age. This video may be inappropriate for some users.", "age-restricted"),
])
def test_private_or_unavailable_videos_show_a_plain_message(env, message, expected):
    env.fake_youtube(extract_error=fakes.DownloadError(message))
    at = env.start()

    env.process_youtube(at)

    text = page_text(at)

    assert any(expected in str(e.value).lower() for e in at.error)

    # never the raw yt-dlp text
    assert "[youtube]" not in text
    assert "ERROR:" not in text
    assert not pipeline_ran(at)


def test_a_failed_download_does_not_disturb_the_video_already_loaded(env):
    env.set_upload("lecture.mp4")
    env.fake_youtube(extract_error=fakes.DownloadError("ERROR: Video unavailable"))

    at = env.start()

    loaded = at.session_state["video_hash"]
    transcribed = len(env.calls["transcribe"])

    env.process_youtube(at)

    assert any("unavailable" in str(e.value).lower() for e in at.error)

    # the uploaded video is still the loaded one, nothing was reprocessed
    assert at.session_state["video_hash"] == loaded
    assert at.session_state["video_path"] == os.path.join("data", "videos", "lecture.mp4")
    assert len(env.calls["transcribe"]) == transcribed


def test_a_live_stream_is_refused_before_anything_is_downloaded(env):
    log = env.fake_youtube(info={"is_live": True})
    at = env.start()

    env.process_youtube(at)

    assert any("Live streams" in str(e.value) for e in at.error)
    assert log["download"] == []
    assert not pipeline_ran(at)


def test_a_too_long_video_is_refused_before_anything_is_downloaded(env):
    log = env.fake_youtube(info={"duration": 3 * 3600})
    at = env.start()

    env.process_youtube(at)

    assert any("180 minutes" in str(e.value) for e in at.error)
    assert log["download"] == []


def test_without_yt_dlp_installed_the_app_explains_how_to_get_it(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    at = env.start()

    env.process_youtube(at)

    assert any("pip install yt-dlp" in str(e.value) for e in at.error)
    assert not pipeline_ran(at)


def test_a_good_link_after_a_bad_one_clears_the_error(env):
    env.fake_youtube()
    at = env.start()

    env.process_youtube(at, "https://vimeo.com/12345")
    assert at.error

    env.process_youtube(at, URL)

    assert not at.error
    assert pipeline_ran(at)


# ============================================================
# which video is in use: the one the user chose last
# ============================================================

def test_uploading_a_file_after_a_youtube_video_switches_back_to_the_upload(env):
    env.fake_youtube(different_video=True)
    at = env.start()
    env.process_youtube(at)

    youtube_hash = at.session_state["video_hash"]

    env.set_upload("lecture.mp4")
    at.run()

    assert at.session_state["active_source"] == "upload"
    assert at.session_state["youtube_source"] is None
    assert at.session_state["video_path"] == os.path.join("data", "videos", "lecture.mp4")
    assert at.session_state["video_hash"] != youtube_hash
    assert len(env.calls["transcribe"]) == 2


def test_a_youtube_link_after_an_upload_switches_to_the_youtube_video(env):
    env.set_upload("lecture.mp4")
    env.fake_youtube(different_video=True)

    at = env.start()
    upload_hash = at.session_state["video_hash"]

    env.process_youtube(at)

    assert at.session_state["active_source"] == "youtube"
    assert at.session_state["video_path"] == os.path.join("data", "youtube", f"{VIDEO_ID}.mp4")
    assert at.session_state["video_hash"] != upload_hash

    # the upload box still holds the file, but it is not re-processed
    at.run()

    assert len(env.calls["transcribe"]) == 2


def test_the_two_folders_never_mix(env):
    env.set_upload("lecture.mp4")
    env.fake_youtube(different_video=True)

    at = env.start()
    env.process_youtube(at)

    assert env.files("data", "videos") == ["lecture.mp4"]
    assert env.files("data", "youtube") == [f"{VIDEO_ID}.json", f"{VIDEO_ID}.mp4"]


def test_the_same_video_from_both_sources_is_recognised_and_not_processed_twice(env):
    """Identity is the content hash, as it always was: same bytes, same video."""
    env.set_upload("lecture.mp4")
    env.fake_youtube()                      # downloads the very same bytes

    at = env.start()
    env.process_youtube(at)

    assert len(env.calls["transcribe"]) == 1
    assert at.session_state["active_source"] == "youtube"