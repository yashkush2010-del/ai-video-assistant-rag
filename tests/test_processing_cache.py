"""
The processing cache on its own (src/processing/processing_cache.py).

No app and no models: only saving, loading and the rules for when a
cached result may be reused.
"""

import json

import numpy as np
import pytest

from src.processing import processing_cache as cache
from src.processing.chunker import create_chunks


VIDEO = "a1b2c3"
WHISPER = "base"
EMBEDDER = "all-MiniLM-L6-v2"

SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 4.5, "text": " Machine learning learns from data.",
     "tokens": [50364, 2234], "temperature": 0.0, "avg_logprob": -0.21},
    {"id": 1, "start": 4.5, "end": 9.24, "text": " Supervised learning uses labels.",
     "tokens": [50589, 431], "temperature": 0.0, "avg_logprob": -0.18},
    {"id": 2, "start": 9.24, "end": 31.0, "text": " Clustering groups similar players.",
     "tokens": [51000], "temperature": 0.2, "avg_logprob": -0.33},
]


def summary_for(chunks, used_llm=True):
    return {
        "main_topic": "Machine Learning",
        "points": [
            {
                "title": "Basics",
                "summary": "Machine learning learns from data.",
                "start": chunks[0]["start"],
                "end": chunks[0]["end"],
                "text": chunks[0]["text"],
                "chunk_index": 0,
                "segment_range": [0, 1],
                "description_source": "llm",
            }
        ],
        "used_llm": used_llm,
    }


@pytest.fixture
def folder(tmp_path):
    return str(tmp_path / "cache")


# ============================================================
# transcript
# ============================================================

def test_nothing_is_cached_at_first(folder):
    assert cache.load_transcript(VIDEO, WHISPER, folder) is None


def test_a_saved_transcript_comes_back_exactly(folder):
    assert cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    assert cache.load_transcript(VIDEO, WHISPER, folder) == SEGMENTS


def test_chunks_from_the_cached_transcript_are_identical(folder):
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    loaded = cache.load_transcript(VIDEO, WHISPER, folder)

    assert create_chunks(loaded) == create_chunks(SEGMENTS)


def test_another_video_is_not_mixed_up(folder):
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    assert cache.load_transcript("ffffff", WHISPER, folder) is None


def test_another_whisper_model_is_not_reused(folder):
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    assert cache.load_transcript(VIDEO, "small", folder) is None


def test_a_new_pipeline_version_ignores_old_results(folder, monkeypatch):
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    monkeypatch.setattr(cache, "PIPELINE_VERSION", cache.PIPELINE_VERSION + 1)

    assert cache.load_transcript(VIDEO, WHISPER, folder) is None


def test_numpy_numbers_from_whisper_are_saved_as_plain_numbers(folder):
    segments = [
        {"start": np.float64(0.0), "end": np.float32(2.5), "text": " Hello.",
         "tokens": np.array([1, 2, 3])},
    ]

    assert cache.save_transcript(VIDEO, WHISPER, segments, folder)

    loaded = cache.load_transcript(VIDEO, WHISPER, folder)

    assert loaded == [{"start": 0.0, "end": 2.5, "text": " Hello.", "tokens": [1, 2, 3]}]


@pytest.mark.parametrize("content", [
    "{ this is not json",
    "",
    "[1, 2, 3]",
    json.dumps({"settings": {}, "segments": []}),
])
def test_a_broken_cache_file_is_ignored(folder, content):
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    with open(cache._path(folder, VIDEO, "transcript"), "w", encoding="utf-8") as file:
        file.write(content)

    assert cache.load_transcript(VIDEO, WHISPER, folder) is None


def test_invalid_segments_in_the_file_are_ignored(folder):
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)
    path = cache._path(folder, VIDEO, "transcript")

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    data["segments"][1]["start"] = "four"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    assert cache.load_transcript(VIDEO, WHISPER, folder) is None


def test_an_empty_transcript_is_not_saved(folder):
    assert not cache.save_transcript(VIDEO, WHISPER, [], folder)
    assert cache.load_transcript(VIDEO, WHISPER, folder) is None


def test_a_failed_save_does_not_raise(tmp_path):
    # The cache "folder" is a file, so nothing can be written there.
    blocked = tmp_path / "not_a_folder"
    blocked.write_text("x")

    assert not cache.save_transcript(VIDEO, WHISPER, SEGMENTS, str(blocked))


def test_no_temporary_file_is_left_behind(folder):
    import os

    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)

    assert sorted(os.listdir(folder)) == [f"{VIDEO}_transcript.json"]


# ============================================================
# summary
# ============================================================

def test_a_saved_summary_comes_back_for_the_same_chunks(folder):
    chunks = create_chunks(SEGMENTS)
    summary = summary_for(chunks)

    assert cache.save_summary(VIDEO, EMBEDDER, chunks, summary, folder)

    assert cache.load_summary(VIDEO, EMBEDDER, chunks, folder) == summary


def test_a_summary_is_not_reused_for_other_chunks(folder):
    chunks = create_chunks(SEGMENTS)
    cache.save_summary(VIDEO, EMBEDDER, chunks, summary_for(chunks), folder)

    other = create_chunks(SEGMENTS, max_duration=5.0)

    assert cache.load_summary(VIDEO, EMBEDDER, other, folder) is None


def test_a_summary_is_not_reused_with_another_embedding_model(folder):
    chunks = create_chunks(SEGMENTS)
    cache.save_summary(VIDEO, EMBEDDER, chunks, summary_for(chunks), folder)

    assert cache.load_summary(VIDEO, "other-model", chunks, folder) is None


def test_the_fallback_summary_without_ollama_is_not_saved(folder):
    chunks = create_chunks(SEGMENTS)

    assert not cache.save_summary(
        VIDEO, EMBEDDER, chunks, summary_for(chunks, used_llm=False), folder
    )
    assert cache.load_summary(VIDEO, EMBEDDER, chunks, folder) is None


@pytest.mark.parametrize("summary", [
    None,
    {"main_topic": "", "points": [], "used_llm": False, "error": True},
    {"main_topic": "X", "points": [], "used_llm": True},
])
def test_failed_or_empty_summaries_are_not_saved(folder, summary):
    assert not cache.worth_caching(summary)


def test_the_fingerprint_changes_when_a_chunk_changes():
    chunks = create_chunks(SEGMENTS)
    changed = [dict(chunk) for chunk in chunks]
    changed[0]["text"] += " extra"

    assert cache.chunks_fingerprint(chunks) == cache.chunks_fingerprint(
        create_chunks(SEGMENTS)
    )
    assert cache.chunks_fingerprint(chunks) != cache.chunks_fingerprint(changed)


# ============================================================
# clearing (the "Reprocess video" button)
# ============================================================

def test_clearing_a_video_removes_its_transcript_and_summary(folder):
    chunks = create_chunks(SEGMENTS)
    cache.save_transcript(VIDEO, WHISPER, SEGMENTS, folder)
    cache.save_summary(VIDEO, EMBEDDER, chunks, summary_for(chunks), folder)
    cache.save_transcript("other", WHISPER, SEGMENTS, folder)

    cache.clear_video(VIDEO, folder)

    assert cache.load_transcript(VIDEO, WHISPER, folder) is None
    assert cache.load_summary(VIDEO, EMBEDDER, chunks, folder) is None

    # other videos are kept
    assert cache.load_transcript("other", WHISPER, folder) == SEGMENTS


def test_clearing_a_video_that_was_never_cached_is_fine(folder):
    cache.clear_video(VIDEO, folder)
