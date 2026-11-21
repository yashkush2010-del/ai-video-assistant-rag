"""
Processing-time measurement for the AI Video Assistant.

Runs the SAME processing steps as app.py on one video, with the
project's own functions and models (no new pipeline):

    extract_audio -> Whisper "base" -> extract_keyframes
    -> match_keyframes_with_transcript -> create_chunks
    -> MiniLM embeddings -> FAISS index -> generate_video_summary

and prints how long every step takes.

The outputs (transcript segments, chunk texts and the summary) and the
timings are saved to data/processed/<video>_<label>.json. A first run
with the default label "baseline" records how the project behaves
BEFORE an optimisation. A later run with another label can be compared
with it (--compare), to show that the output did not change and how
much time was saved.

Usage (from the project folder, with the virtual environment active):

    python scripts/measure_processing.py
    python scripts/measure_processing.py --video data/videos/other.mp4
    python scripts/measure_processing.py --no-summary   (skip Ollama)
    python scripts/measure_processing.py --label after --compare

Ollama must be running for the summary step (or use --no-summary).
"""

import argparse
import json
import os
import sys
import time

# Run from anywhere: the project folder is the parent of scripts/.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.ingestion.audio_extractor import extract_audio  # noqa: E402
from src.processing.chunker import create_chunks  # noqa: E402
from src.processing.keyframe_extractor import (  # noqa: E402
    extract_keyframes,
    match_keyframes_with_transcript,
)
from src.rag.vector_store import create_index  # noqa: E402


# ============================================================
# SETTINGS (the same values app.py uses)
# ============================================================

DEFAULT_VIDEO = os.path.join("data", "videos", "machine_learning.mp4")
DEFAULT_LABEL = "baseline"

WHISPER_MODEL = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KEYFRAME_INTERVAL = 30
KEYFRAME_TOLERANCE = 2.0


# ============================================================
# TIMING HELPER
# ============================================================

class StepTimer:
    """Collects how long each named step takes, in order."""

    def __init__(self):
        self.steps = []

    def run(self, name, function, *args, **kwargs):
        print(f"  {name}...", end="", flush=True)

        started = time.perf_counter()
        result = function(*args, **kwargs)
        seconds = time.perf_counter() - started

        self.steps.append({"step": name, "seconds": round(seconds, 2)})
        print(f" {seconds:.1f} s")

        return result

    def total(self):
        return round(sum(step["seconds"] for step in self.steps), 2)


# ============================================================
# MEASUREMENT
# ============================================================

def measure(video_path, with_summary=True):
    # Imported here so --help works without loading torch.
    import whisper
    from sentence_transformers import SentenceTransformer

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Own audio file, so the app's data/audio/sample.wav is not touched.
    audio_path = os.path.join("data", "audio", f"{video_name}.wav")
    keyframe_dir = os.path.join("data", "keyframes", video_name)

    os.makedirs(os.path.join("data", "audio"), exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    timer = StepTimer()

    print(f"\nMeasuring: {video_path}\n")

    timer.run("Extract audio (ffmpeg)", extract_audio, video_path, audio_path)

    whisper_model = timer.run(
        "Load Whisper model", whisper.load_model, WHISPER_MODEL
    )
    embedding_model = timer.run(
        "Load embedding model", SentenceTransformer, EMBEDDING_MODEL
    )

    result = timer.run("Transcribe (Whisper)", whisper_model.transcribe, audio_path)
    segments = result["segments"]

    keyframes = timer.run(
        "Extract keyframes",
        extract_keyframes,
        video_path,
        keyframe_dir,
        interval=KEYFRAME_INTERVAL,
    )

    matched = timer.run(
        "Match keyframes with transcript",
        match_keyframes_with_transcript,
        keyframes,
        segments,
        tolerance=KEYFRAME_TOLERANCE,
    )

    chunks = timer.run("Create chunks", create_chunks, segments)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = timer.run(
        "Create embeddings",
        embedding_model.encode,
        texts,
        normalize_embeddings=True,
    )

    timer.run("Create FAISS index", create_index, embeddings)

    summary = None

    if with_summary:
        from src.rag.video_summary import generate_video_summary

        summary = timer.run(
            "Generate video summary (Ollama)",
            generate_video_summary,
            chunks,
            embedding_model,
        )

    return {
        "video": video_path,
        "settings": {
            "whisper_model": WHISPER_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "keyframe_interval": KEYFRAME_INTERVAL,
        },
        "timings": timer.steps,
        "total_seconds": timer.total(),
        "counts": {
            "segments": len(segments),
            "chunks": len(chunks),
            "keyframes": len(keyframes),
            "matched_keyframes": len(matched),
        },
        "segments": [
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"].strip(),
            }
            for segment in segments
        ],
        "chunks": [
            {"start": chunk["start"], "end": chunk["end"], "text": chunk["text"]}
            for chunk in chunks
        ],
        "summary": _summary_for_saving(summary),
    }


def _summary_for_saving(summary):
    # Only what the user sees; chunk "segments" are already saved above.
    if summary is None:
        return None

    return {
        "main_topic": summary.get("main_topic", ""),
        "used_llm": summary.get("used_llm", False),
        "points": [
            {
                "title": point.get("title", ""),
                "summary": point.get("summary", ""),
                "start": point.get("start"),
                "end": point.get("end"),
            }
            for point in summary.get("points", [])
        ],
    }


# ============================================================
# REPORT AND COMPARISON
# ============================================================

def print_timings(report):
    print("\n" + "=" * 60)
    print("PROCESSING TIME")
    print("=" * 60)

    for step in report["timings"]:
        print(f"  {step['step']:<36} {step['seconds']:>8.1f} s")

    print("  " + "-" * 46)
    print(f"  {'Total':<36} {report['total_seconds']:>8.1f} s")

    counts = report["counts"]
    print(
        f"\n  {counts['segments']} transcript segments, "
        f"{counts['chunks']} chunks, "
        f"{counts['keyframes']} keyframes "
        f"({counts['matched_keyframes']} matched)"
    )


def compare(report, baseline):
    """Prints whether the outputs match the baseline, and the time saved."""

    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)

    checks = [
        ("Transcript segments", report["segments"], baseline["segments"]),
        ("Chunks", report["chunks"], baseline["chunks"]),
    ]

    for name, now, before in checks:
        print(f"  {name:<24} {'same' if now == before else 'DIFFERENT'}")

    # The summary uses Ollama, so it may legitimately differ between
    # two fresh runs; it is reported, not treated as an error.
    if report["summary"] is not None and baseline.get("summary") is not None:
        same = report["summary"] == baseline["summary"]
        print(
            f"  {'Summary':<24} "
            f"{'same' if same else 'different (Ollama output can vary)'}"
        )

    before_total = baseline["total_seconds"]
    now_total = report["total_seconds"]

    print(f"\n  Baseline total : {before_total:.1f} s")
    print(f"  This run total : {now_total:.1f} s")

    if before_total > 0:
        change = (before_total - now_total) / before_total * 100
        print(f"  Time saved     : {change:.0f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="skip the Ollama summary step",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare this run with the saved baseline",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    report = measure(args.video, with_summary=not args.no_summary)
    print_timings(report)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = os.path.join(
        "data", "processed", f"{video_name}_{args.label}.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")

    if args.compare:
        baseline_path = os.path.join(
            "data", "processed", f"{video_name}_{DEFAULT_LABEL}.json"
        )

        if not os.path.exists(baseline_path):
            print(f"\nNo baseline to compare with: {baseline_path}")
            return

        with open(baseline_path, encoding="utf-8") as file:
            compare(report, json.load(file))


if __name__ == "__main__":
    main()
