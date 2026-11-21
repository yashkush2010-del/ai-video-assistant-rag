"""
Real-video check for Q&A.

Asks a fixed list of questions about a real video with the project's
OWN Q&A code, exactly as app.py does:

    MiniLM question embedding -> retrieve_chunks (k=6, threshold 0.25)
    -> add_neighbouring_chunks -> prepare_context -> generate_answer (Ollama)
    -> find_evidence (the evidence and Jump time)

and checks every answer for the key facts it should contain. The
questions and their key facts are in scripts/qa_questions/*.txt (the
"answer key"; see the format at the top of those files).

The questions are only a TEST. They are not part of the app. A change
that improves answers should improve them on every video, so the
check is run on two videos about different topics.

The transcript comes from the app's processing cache (data/cache), so
a video that was opened in the app before is not transcribed again.

The report is printed and also saved to
data/processed/<video>_qa_report_<label>.txt and .json. Run it once
with --label before a change and once with --label after, then
compare the two with --compare before.

Usage (from the project folder, with the virtual environment active
and Ollama running):

    python scripts/check_qa_real_video.py --questions scripts/qa_questions/machine_learning.txt --label before
    python scripts/check_qa_real_video.py --questions scripts/qa_questions/internet.txt --label before
    python scripts/check_qa_real_video.py --questions scripts/qa_questions/internet.txt --label after --compare before
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

# Run from anywhere: the project folder is the parent of scripts/.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.processing import processing_cache  # noqa: E402
from src.processing.chunker import create_chunks  # noqa: E402
from src.rag.answer_generator import (  # noqa: E402
    generate_answer,
    prepare_context,
)
from src.rag.vector_store import (  # noqa: E402
    add_neighbouring_chunks,
    create_index,
    find_evidence,
    retrieve_chunks,
)
from src.rag.video_summary import format_time_range  # noqa: E402


# ============================================================
# SETTINGS (the same values app.py uses)
# ============================================================

WHISPER_MODEL = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RETRIEVE_K = 6
RETRIEVE_THRESHOLD = 0.25

NOT_IN_VIDEO = "NOT_IN_VIDEO"

# The sentence the app looks for to know that nothing was found.
NO_INFORMATION = "don't have enough information"


# ============================================================
# QUESTION FILE
# ============================================================

def read_questions(path):
    """
    Returns (video_path, questions). Every question is
    {"question": str, "facts": [[alternative, ...], ...], "answerable": bool}.
    """
    video_path = None
    questions = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line.startswith("#"):
                found = re.match(r"#\s*video:\s*(.+)$", line)
                if found:
                    video_path = found.group(1).strip()
                continue

            if not line:
                continue

            if "|" not in line:
                sys.exit(f"No '|' in question line: {line}")

            parts = [part.strip() for part in line.split("|")]
            question, facts = parts[0], parts[1]
            windows = parse_windows(parts[2]) if len(parts) > 2 else []

            if facts == NOT_IN_VIDEO:
                questions.append({
                    "question": question, "facts": [], "answerable": False,
                    "windows": [],
                })
                continue

            questions.append({
                "question": question,
                "facts": [
                    [word.strip().lower() for word in fact.split("/") if word.strip()]
                    for fact in facts.split(";")
                    if fact.strip()
                ],
                "answerable": True,
                "windows": windows,
            })

    return video_path, questions


def _seconds(clock):
    seconds = 0
    for part in clock.strip().split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def parse_windows(text):
    """'0:06-0:17, 2:29-2:41' -> [(6, 17), (149, 161)]"""
    windows = []

    for window in text.split(","):
        if "-" in window:
            start, end = window.split("-", 1)
            windows.append((_seconds(start), _seconds(end)))

    return windows


def evidence_correct(start, windows):
    """None when there is no window to check against."""
    if not windows:
        return None

    return any(first <= int(start) < last for first, last in windows)


# ============================================================
# CHECKING AN ANSWER
# ============================================================

def _normalise(text):
    return " ".join(
        str(text).lower().replace("’", "'").replace("‘", "'").split()
    )


def fact_found(answer, alternatives):
    """
    True if the answer contains any alternative, as a word or the
    START of a word ("label" matches "labeled", not "unlabeled").
    """
    answer = _normalise(answer)

    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(alternative), answer)
        for alternative in alternatives
    )


def grade(question, answer):
    """
    GOOD     : every key fact is in the answer
    PARTIAL  : some key facts are in the answer
    BAD      : none of them, or it wrongly said it had no information
    REFUSED  : (not-in-video question) correctly said it had no information
    INVENTED : (not-in-video question) answered anyway
    """
    said_no_information = NO_INFORMATION in _normalise(answer)

    if not question["answerable"]:
        return ("REFUSED" if said_no_information else "INVENTED"), [], []

    found = [fact for fact in question["facts"] if fact_found(answer, fact)]
    missing = [fact for fact in question["facts"] if fact not in found]

    if said_no_information or not found:
        return "BAD", found, missing

    return ("GOOD" if not missing else "PARTIAL"), found, missing


def points(verdict):
    return {"GOOD": 1.0, "REFUSED": 1.0, "PARTIAL": 0.5}.get(verdict, 0.0)


# ============================================================
# PIPELINE
# ============================================================

def load_segments(video_path):
    """The video's Whisper segments: from the app's cache, or Whisper now."""
    with open(video_path, "rb") as file:
        video_hash = hashlib.md5(file.read()).hexdigest()

    segments = processing_cache.load_transcript(video_hash, WHISPER_MODEL)

    if segments is not None:
        print("Transcript loaded from the app's cache.")
        return segments

    import whisper
    from src.ingestion.audio_extractor import extract_audio

    name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join("data", "audio", f"{name}.wav")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    print("Not in the cache - transcribing with Whisper (a few minutes)...")
    extract_audio(video_path, audio_path)
    segments = whisper.load_model(WHISPER_MODEL).transcribe(audio_path)["segments"]

    processing_cache.save_transcript(video_hash, WHISPER_MODEL, segments)

    return segments


def ask_all(video_path, questions):
    from sentence_transformers import SentenceTransformer

    segments = load_segments(video_path)
    chunks = create_chunks(segments)

    print("Loading the embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
    )
    index = create_index(embeddings)

    results = []

    for number, item in enumerate(questions, start=1):
        print(f"  Question {number}/{len(questions)}: {item['question']}")

        query_embedding = embedding_model.encode(
            item["question"],
            normalize_embeddings=True,
        )

        retrieved = retrieve_chunks(
            index,
            chunks,
            query_embedding,
            k=RETRIEVE_K,
            threshold=RETRIEVE_THRESHOLD,
        )

        # The same step as app.py, so this script always checks
        # what the app really does.
        retrieved = add_neighbouring_chunks(retrieved, chunks)

        started = time.perf_counter()
        answer = generate_answer(prepare_context(retrieved), item["question"])
        seconds = time.perf_counter() - started

        evidence = []
        if NO_INFORMATION not in _normalise(answer):
            evidence = find_evidence(
                retrieved, item["question"], answer, embedding_model
            )

        verdict, found, missing = grade(item, answer)

        results.append({
            "question": item["question"],
            "answerable": item["answerable"],
            "evidence_correct": (
                evidence_correct(evidence[0]["start"], item["windows"])
                if evidence else None
            ),
            "answer": answer.strip(),
            "verdict": verdict,
            "found": [fact[0] for fact in found],
            "missing": [" / ".join(fact) for fact in missing],
            "evidence": [
                {"time": format_time_range(e["start"], e["end"]), "text": e["text"]}
                for e in evidence
            ],
            "retrieved": [
                {"time": format_time_range(r["start"], r["end"]),
                 "score": round(r["score"], 3)}
                for r in retrieved
            ],
            "seconds": round(seconds, 1),
        })

    return results


# ============================================================
# REPORT
# ============================================================

def report_text(video_path, label, results):
    lines = [
        "=" * 70,
        f"Q&A CHECK - {video_path} - {label}",
        "=" * 70,
    ]

    for number, result in enumerate(results, start=1):
        lines += [
            "",
            f"Q{number}. {result['question']}",
            f"  Answer   : {result['answer']}",
            f"  Verdict  : {result['verdict']}",
        ]

        if result["missing"]:
            lines.append(f"  Missing  : {'; '.join(result['missing'])}")

        for evidence in result["evidence"]:
            mark = {True: "  (right place)", False: "  (WRONG place)"}.get(
                result.get("evidence_correct"), ""
            )
            lines.append(
                f"  Evidence : [{evidence['time']}] {evidence['text']}{mark}"
            )

        lines.append(
            "  Searched : "
            + ", ".join(f"{r['time']} ({r['score']})" for r in result["retrieved"])
        )

    total = sum(points(result["verdict"]) for result in results)
    counts = {}
    for result in results:
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1

    checked = [r["evidence_correct"] for r in results if r.get("evidence_correct") is not None]
    evidence_line = (
        f"Evidence at the right place: {sum(checked)} / {len(checked)}"
        if checked else "Evidence: no windows to check"
    )

    lines += [
        "",
        "=" * 70,
        evidence_line,
        f"SCORE: {total:g} / {len(results)}   "
        + "   ".join(f"{verdict} {count}" for verdict, count in sorted(counts.items())),
        f"Average answer time: "
        f"{sum(r['seconds'] for r in results) / max(len(results), 1):.1f} s",
        "=" * 70,
    ]

    return "\n".join(lines)


def comparison_text(results, earlier, earlier_label, label):
    lines = ["", f"COMPARISON: {earlier_label} -> {label}", "-" * 70]

    before = {result["question"]: result for result in earlier}

    for result in results:
        old = before.get(result["question"])
        old_verdict = old["verdict"] if old else "-"
        mark = ""

        if old:
            change = points(result["verdict"]) - points(old_verdict)
            mark = "  better" if change > 0 else "  WORSE" if change < 0 else ""

        lines.append(
            f"  {old_verdict:<9} -> {result['verdict']:<9}{mark}   {result['question']}"
        )

    old_total = sum(points(result["verdict"]) for result in earlier)
    new_total = sum(points(result["verdict"]) for result in results)
    lines.append(f"\n  Score: {old_total:g} -> {new_total:g} (of {len(results)})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--questions", required=True)
    parser.add_argument("--video", help="overrides the 'video:' line of the file")
    parser.add_argument("--label", default="run")
    parser.add_argument("--compare", metavar="LABEL",
                        help="compare with an earlier run of this label")
    args = parser.parse_args()

    video_path, questions = read_questions(args.questions)
    video_path = args.video or video_path

    if not video_path or not os.path.exists(video_path):
        sys.exit(f"Video not found: {video_path}")

    if not questions:
        sys.exit("No questions in the file.")

    print(f"\n{len(questions)} questions about {video_path}\n")

    results = ask_all(video_path, questions)
    text = report_text(video_path, args.label, results)

    name = os.path.splitext(os.path.basename(video_path))[0]
    base = os.path.join("data", "processed", f"{name}_qa_report")

    if args.compare:
        earlier_path = f"{base}_{args.compare}.json"

        if os.path.exists(earlier_path):
            with open(earlier_path, encoding="utf-8") as file:
                earlier = json.load(file)
            text += "\n" + comparison_text(results, earlier, args.compare, args.label)
        else:
            text += f"\n\nNothing to compare with: {earlier_path}"

    print("\n" + text)

    os.makedirs(os.path.dirname(base), exist_ok=True)

    with open(f"{base}_{args.label}.txt", "w", encoding="utf-8") as file:
        file.write(text + "\n")

    with open(f"{base}_{args.label}.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved: {base}_{args.label}.txt")


if __name__ == "__main__":
    main()
