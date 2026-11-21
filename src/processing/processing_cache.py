"""
Processing cache for the AI Video Assistant.

The two slow steps of processing a video are:

    Whisper transcription     (about 55 % of the time)
    Video Summary with Ollama (about 30 % of the time)

Their results only depend on the video and the settings, so they are
saved here the first time and reused the next time the SAME video is
opened (also after a page refresh or a restart of Streamlit).

Nothing else is cached: chunks, embeddings and the FAISS index are
rebuilt from the cached transcript every time. They take about a
second and can therefore never be out of date.

A cached result is used only when it was made with the same settings:

    transcript : video hash + Whisper model + PIPELINE_VERSION
    summary    : video hash + embedding model + PIPELINE_VERSION
                 + a fingerprint of the exact chunks it was built from

Anything that does not match, cannot be read or looks wrong is simply
treated as "not cached" and the video is processed normally. The cache
must never be able to break the app.

Files: data/cache/<video hash>_transcript.json
       data/cache/<video hash>_summary.json
(data/ is not committed to Git.)
"""

import hashlib
import json
import logging
import numbers
import os


log = logging.getLogger(__name__)


# ============================================================
# SETTINGS
# ============================================================

CACHE_DIR = os.path.join("data", "cache")

# Increase this number whenever the processing code changes in a way
# that changes its results (for example how chunks are built or how the
# summary is written). Every cached result made with an older number is
# then ignored and the video is processed again.
PIPELINE_VERSION = 1


# ============================================================
# TRANSCRIPT
# ============================================================

def load_transcript(video_hash, whisper_model, cache_dir=CACHE_DIR):
    """
    The cached Whisper segments of this video, or None.

    None means: not cached yet, made with other settings, or the file
    could not be read. The caller then runs Whisper as usual.
    """
    data = _read(_path(cache_dir, video_hash, "transcript"))

    if data is None:
        return None

    if data.get("settings") != _transcript_settings(video_hash, whisper_model):
        return None

    segments = data.get("segments")

    if not _valid_segments(segments):
        log.warning("Cached transcript for %s is invalid - ignored", video_hash)
        return None

    return segments


def save_transcript(video_hash, whisper_model, segments, cache_dir=CACHE_DIR):
    """Saves the Whisper segments. Returns False if they could not be saved."""
    if not _valid_segments(segments):
        return False

    return _write(
        _path(cache_dir, video_hash, "transcript"),
        {
            "settings": _transcript_settings(video_hash, whisper_model),
            "segments": segments,
        },
    )


def _transcript_settings(video_hash, whisper_model):
    return {
        "video_hash": video_hash,
        "whisper_model": whisper_model,
        "pipeline_version": PIPELINE_VERSION,
    }


def _valid_segments(segments):
    # Only what the rest of the app relies on: start, end and text.
    # (numbers.Real also accepts NumPy numbers, which Whisper can return.)
    return (
        isinstance(segments, list)
        and bool(segments)
        and all(
            isinstance(segment, dict)
            and isinstance(segment.get("start"), numbers.Real)
            and isinstance(segment.get("end"), numbers.Real)
            and isinstance(segment.get("text"), str)
            for segment in segments
        )
    )


# ============================================================
# VIDEO SUMMARY
# ============================================================

def load_summary(video_hash, embedding_model, chunks, cache_dir=CACHE_DIR):
    """
    The cached video summary for exactly these chunks, or None.

    Summary points refer to chunks by their number, so a summary is only
    reused for the very same chunks it was made from.
    """
    data = _read(_path(cache_dir, video_hash, "summary"))

    if data is None:
        return None

    if data.get("settings") != _summary_settings(video_hash, embedding_model, chunks):
        return None

    summary = data.get("summary")

    if not isinstance(summary, dict) or not isinstance(summary.get("points"), list):
        log.warning("Cached summary for %s is invalid - ignored", video_hash)
        return None

    return summary


def save_summary(video_hash, embedding_model, chunks, summary, cache_dir=CACHE_DIR):
    """
    Saves the video summary. Returns False if it was not saved.

    Only a summary that Ollama really wrote is kept. When Ollama was not
    running, the app shows a simpler fallback summary; keeping that one
    would hide the better summary even after Ollama is started again.
    """
    if not worth_caching(summary):
        return False

    return _write(
        _path(cache_dir, video_hash, "summary"),
        {
            "settings": _summary_settings(video_hash, embedding_model, chunks),
            "summary": summary,
        },
    )


def worth_caching(summary):
    return (
        isinstance(summary, dict)
        and bool(summary.get("used_llm"))
        and not summary.get("error")
        and bool(summary.get("points"))
    )


def _summary_settings(video_hash, embedding_model, chunks):
    return {
        "video_hash": video_hash,
        "embedding_model": embedding_model,
        "pipeline_version": PIPELINE_VERSION,
        "chunks": chunks_fingerprint(chunks),
    }


def chunks_fingerprint(chunks):
    """A short code that changes whenever any chunk's time or text changes."""
    content = json.dumps(
        [
            [float(chunk["start"]), float(chunk["end"]), chunk["text"]]
            for chunk in chunks
        ],
        ensure_ascii=False,
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ============================================================
# REMOVING A CACHED VIDEO (the "Reprocess video" button)
# ============================================================

def clear_video(video_hash, cache_dir=CACHE_DIR):
    """Deletes everything cached for this video."""
    for kind in ("transcript", "summary"):
        path = _path(cache_dir, video_hash, kind)

        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as error:
            log.warning("Could not delete %s - %s", path, error)


# ============================================================
# FILES
# ============================================================

def _path(cache_dir, video_hash, kind):
    return os.path.join(cache_dir, f"{video_hash}_{kind}.json")


def _read(path):
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)

    except (OSError, ValueError) as error:
        log.warning("Could not read cache file %s - %s", path, error)
        return None

    return data if isinstance(data, dict) else None


def _write(path, data):
    # Written to a temporary file first and then renamed, so a crash
    # in the middle of writing can never leave half a file behind.
    temporary = path + ".tmp"

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, default=_plain)

        os.replace(temporary, path)

    except (OSError, TypeError, ValueError) as error:
        log.warning("Could not save cache file %s - %s", path, error)

        try:
            os.remove(temporary)
        except OSError:
            pass

        return False

    return True


def _plain(value):
    # Whisper and NumPy can return NumPy numbers and arrays; JSON needs
    # plain ones. tolist() turns both into plain Python values.
    if hasattr(value, "tolist"):
        return value.tolist()

    raise TypeError(f"Cannot save {type(value).__name__} in the cache")
