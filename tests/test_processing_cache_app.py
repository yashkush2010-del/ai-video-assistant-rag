"""
The processing cache inside the whole app.

Uses the same environment as the YouTube app tests (real 4-second
video, real ffmpeg / OpenCV / FAISS; Whisper, MiniLM and Ollama are
stand-ins). A "new session" is a new AppTest: the same as refreshing
the page or restarting Streamlit.
"""

import json
import os

import pytest

from test_youtube_ingestion_app import (  # noqa: F401
    env,
    page_text,
    pipeline_ran,
)


CACHE_DIR = os.path.join("data", "cache")


def transcript_files():
    if not os.path.isdir(CACHE_DIR):
        return []

    return sorted(name for name in os.listdir(CACHE_DIR) if name.endswith(".json"))


def open_upload(env):
    env.set_upload("lecture.mp4")
    at = env.start()
    assert pipeline_ran(at)
    return at


def reprocess(at):
    at.button(key="reprocess_video").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture
def ollama_summary(monkeypatch):
    """
    A summary "written by Ollama" (used_llm True), counting its calls.
    The real stand-in Ollama returns nothing, which gives the fallback
    summary, and that one is deliberately never cached.
    """
    from src.rag import video_summary

    calls = []

    def fake_summary(chunks, embedding_model, *args, **kwargs):
        calls.append(len(chunks))
        chunk = chunks[0]

        return {
            "main_topic": "Machine Learning",
            "points": [{
                "title": "Basics",
                "summary": "Machine learning lets computers learn patterns from data.",
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "chunk_index": 0,
                "segment_range": [0, 0],
                "description_source": "llm",
            }],
            "used_llm": True,
        }

    monkeypatch.setattr(video_summary, "generate_video_summary", fake_summary)

    return calls


# ============================================================
# the first time: processed as before, and saved
# ============================================================

def test_the_first_time_a_video_is_processed_as_before(env):
    at = open_upload(env)

    assert len(env.calls["extract_audio"]) == 1
    assert len(env.calls["transcribe"]) == 1
    assert "Transcription completed!" in page_text(at)
    assert "loaded from cache" not in page_text(at)

    assert transcript_files() == [f"{at.session_state['video_hash']}_transcript.json"]


# ============================================================
# the next time: loaded from the cache
# ============================================================

def test_a_new_session_reuses_the_transcript(env):
    first = open_upload(env)

    second = open_upload(env)

    # audio extraction and Whisper did not run again
    assert len(env.calls["extract_audio"]) == 1
    assert len(env.calls["transcribe"]) == 1
    assert "Transcript loaded from cache" in page_text(second)

    # and the result is exactly the same
    for key in ("segments", "chunks", "multimodal_records"):
        assert second.session_state[key] == first.session_state[key]

    assert second.session_state["index"].ntotal == first.session_state["index"].ntotal


def test_everything_still_works_on_a_cached_video(env):
    at = open_upload(env)
    at = open_upload(env)

    # keyframes, index and summary are all there
    assert len(env.calls["extract_keyframes"]) == 2
    assert at.session_state["index"] is not None
    assert at.session_state["video_summary"] is not None
    assert any("Ask Questions" in str(m.value) for m in at.markdown)


def test_another_video_is_not_taken_from_the_cache(env):
    open_upload(env)

    env.fake_youtube(different_video=True)
    env.upload["file"] = None

    at = env.start()
    env.process_youtube(at)

    assert len(env.calls["transcribe"]) == 2
    assert len(transcript_files()) == 2


def test_a_broken_cache_file_means_normal_processing(env):
    first = open_upload(env)
    path = os.path.join(CACHE_DIR, f"{first.session_state['video_hash']}_transcript.json")

    with open(path, "w", encoding="utf-8") as file:
        file.write("{ broken")

    second = open_upload(env)

    assert len(env.calls["transcribe"]) == 2
    assert "Transcription completed!" in page_text(second)

    # and a good file replaces the broken one
    with open(path, encoding="utf-8") as file:
        assert json.load(file)["segments"]


# ============================================================
# the summary
# ============================================================

def test_an_ollama_summary_is_reused(env, ollama_summary):
    first = open_upload(env)
    second = open_upload(env)

    assert ollama_summary == [len(first.session_state["chunks"])]
    assert (
        second.session_state["video_summary"]["summary"]
        == first.session_state["video_summary"]["summary"]
    )


def test_the_fallback_summary_is_not_kept(env):
    # The stand-in Ollama returns nothing: the fallback summary is shown
    # but not cached, so a later run with Ollama can write a real one.
    at = open_upload(env)

    assert at.session_state["video_summary"]["summary"]["used_llm"] is False
    assert transcript_files() == [f"{at.session_state['video_hash']}_transcript.json"]


# ============================================================
# Reprocess video
# ============================================================

def test_reprocess_runs_whisper_and_the_summary_again(env, ollama_summary):
    at = open_upload(env)
    at = open_upload(env)

    assert len(env.calls["transcribe"]) == 1

    at = reprocess(at)

    assert len(env.calls["extract_audio"]) == 2
    assert len(env.calls["transcribe"]) == 2
    assert len(ollama_summary) == 2
    assert "Transcription completed!" in page_text(at)
    assert pipeline_ran(at)

    # and it is saved again for next time
    third = open_upload(env)
    assert len(env.calls["transcribe"]) == 2
    assert "Transcript loaded from cache" in page_text(third)


def test_reprocess_starts_a_new_viva(env):
    at = open_upload(env)

    at.button_group(key="section").set_value("\U0001f393 Viva")
    at.run()
    at.button(key="viva_generate").click()
    at.run()
    assert at.session_state["viva"] is not None

    at = reprocess(at)

    assert at.session_state["viva"] is None
    assert at.session_state["viva_results"] == {}


def test_reprocess_is_not_offered_once_the_upload_is_removed(env):
    at = open_upload(env)

    assert at.button(key="reprocess_video") is not None

    env.upload["file"] = None
    at.run()

    assert not any(button.key == "reprocess_video" for button in at.button)
