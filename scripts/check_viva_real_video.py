"""
Real-video check for Viva Mode.

Runs Viva Mode on a real video with the project's OWN pipeline and
models (no new pipeline):

    extract_audio -> Whisper "base" -> create_chunks
    -> MiniLM embeddings -> generate_video_summary
    -> generate_viva_questions -> check_answer

and prints, for every question:
    - the question, its kind and where it came from
    - the expected answer
    - the timestamp and the transcript evidence
    - the verdict for a VALID alternative answer, a PARTIAL answer
      and an UNRELATED answer
followed by a short table of checks.

The Whisper transcript is saved the first time
(data/processed/<video>_segments.json), so later runs skip Whisper.
The whole report is also written to
data/processed/<video>_viva_report.txt.

Usage (from the project folder, with the virtual environment active):

    python scripts/check_viva_real_video.py
    python scripts/check_viva_real_video.py --no-llm     (fallback, no Ollama)
    python scripts/check_viva_real_video.py --fresh      (run Whisper again)
    python scripts/check_viva_real_video.py --video data/videos/other.mp4

Ollama must be running for the normal (LLM) run.
"""

import argparse
import json
import logging
import os
import sys

# Run from anywhere: the project folder is the parent of scripts/.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.ingestion.audio_extractor import extract_audio  # noqa: E402
from src.processing.chunker import create_chunks  # noqa: E402
from src.rag.video_summary import (  # noqa: E402
    format_time_range,
    generate_video_summary,
)
from src.rag import viva  # noqa: E402


# ============================================================
# TEST ANSWERS FOR THE MACHINE LEARNING VIDEO
# ============================================================
#
# A question is matched to a case by topic words in the question and
# its section. "valid" is a correct answer in DIFFERENT words from the
# video; "partial" is on topic but incomplete; "vague" (optional) is too
# vague to count and must be rejected.

CASES = [
    {
        "name": "supervised learning (coins)",
        "topic": ["supervised", "coin", "currenc", "weight"],
        "valid": (
            "Supervised learning trains a model using labelled data, "
            "where each input comes with the correct output."
        ),
        "partial": "The model learns from data.",
    },
    {
        "name": "unsupervised learning (cricket)",
        "topic": ["unsupervised", "cluster", "cricket", "wicket", "unlabeled"],
        "valid": (
            "The machine groups the players into two clusters without any "
            "labels: players with many runs and few wickets, and players "
            "with few runs but many wickets."
        ),
        "partial": "The machine finds two clusters of players.",
        # must stay rejected (too vague: one topic word)
        "vague": "It puts the players into groups.",
    },
    {
        "name": "reinforcement learning",
        "topic": ["reinforcement", "reward", "penalt", "feedback"],
        "valid": (
            "Reinforcement learning learns by trial and error, getting a "
            "reward for a good action and a penalty for a bad one."
        ),
        # (not "It learns from feedback": that fully answers the real
        # question "What principle does reinforcement learning work on?")
        "partial": "It learns from rewards.",
    },
    {
        "name": "Paul and songs",
        "topic": ["paul", "song", "tempo", "intensity"],
        "valid": (
            "Machine learning uses Paul's past choices, based on the tempo "
            "and intensity of songs, to predict whether he will like a new song."
        ),
        "partial": "It is about the songs Paul likes.",
    },
    {
        "name": "why machine learning now",
        "topic": ["humongous", "data available", "computational", "popular"],
        "valid": (
            "Machine learning has become popular because huge amounts of "
            "data and more computing power are available today."
        ),
        "partial": "Because of data.",
    },
    {
        "name": "applications",
        "topic": ["healthcare", "uber", "sentiment", "fraud", "used in"],
        "valid": (
            "Machine learning is used in many fields such as healthcare, "
            "sentiment analysis, fraud detection and apps like Uber."
        ),
        "partial": "It is used in healthcare.",
    },
]

UNRELATED_ANSWERS = [
    "I like playing cricket on weekends.",
    "The capital of France is Paris.",
]


def find_case(item):
    text = (item["question"] + " " + item.get("section_text", "")).lower()

    best = None
    best_hits = 0

    for case in CASES:
        hits = sum(1 for word in case["topic"] if word in text)

        if hits > best_hits:
            best, best_hits = case, hits

    return best


# ============================================================
# PIPELINE (the project's own functions)
# ============================================================

def load_segments(video_path, fresh):
    name = os.path.splitext(os.path.basename(video_path))[0]

    cache = os.path.join("data", "processed", f"{name}_segments.json")

    if os.path.exists(cache) and not fresh:
        print(f"Using saved transcript: {cache}")
        with open(cache, encoding="utf-8") as file:
            return json.load(file), name

    import whisper

    os.makedirs(os.path.join("data", "audio"), exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    audio_path = os.path.join("data", "audio", f"{name}.wav")

    print("Extracting audio ...")
    extract_audio(video_path, audio_path)

    print("Transcribing with Whisper (base) - this takes a minute or two ...")
    result = whisper.load_model("base").transcribe(audio_path)

    segments = [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"],
        }
        for segment in result["segments"]
    ]

    with open(cache, "w", encoding="utf-8") as file:
        json.dump(segments, file, indent=2, ensure_ascii=False)

    print(f"Transcript saved: {cache}")

    return segments, name


def no_llm(prompt):
    raise ConnectionError("--no-llm: the language model is switched off")


# ============================================================
# REPORT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--video",
        default=os.path.join("data", "videos", "machine_learning.mp4")
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    # Viva prints why an LLM question was not used; show it here too.
    class Collect(logging.Handler):
        def emit(self, record):
            out(f"  [log] {record.getMessage()}")

    logging.getLogger("src.rag.viva").addHandler(Collect())
    logging.getLogger("src.rag.viva").propagate = False

    if not os.path.exists(arguments.video):
        print(f"Video not found: {arguments.video}")
        return 1

    segments, name = load_segments(arguments.video, arguments.fresh)

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm = no_llm if arguments.no_llm else None

    chunks = create_chunks(segments)

    out("=" * 70)
    out(f"VIVA REAL-VIDEO CHECK: {arguments.video}")
    out(f"Mode: {'NO LLM (fallback)' if arguments.no_llm else 'LLM (Ollama llama3.2)'}")
    out(f"Whisper segments: {len(segments)}   chunks: {len(chunks)}")
    out("=" * 70)

    summary = generate_video_summary(chunks, embedding_model, llm=llm)

    out(f"Summary points: {len(summary['points'])}")
    for point in summary["points"]:
        out(
            f"  - {format_time_range(point['start'], point['end'])}  "
            f"{point['title']}"
        )
    out()

    questions = viva.generate_viva_questions(
        summary["points"],
        llm=llm,
        segments=segments,
        embedding_model=embedding_model
    )

    out(f"Viva questions: {len(questions)}")

    checks = []

    for number, item in enumerate(questions, start=1):

        out()
        out("-" * 70)
        out(f"Q{number} [{item['kind']}, from {item['source']}]")
        out(f"  Question : {item['question']}")
        out(f"  Expected : {item['answer']}")
        out(
            f"  Time     : {format_time_range(item['start'], item['end'])}"
            f"   (section {format_time_range(item['section_start'], item['section_end'])})"
        )
        out(f"  Evidence : {item['text']}")

        # (a fill-in-the-blank answer is a phrase: check its sentence)
        answers_question = viva._answer_problem(
            item.get("answer_sentence") or item["answer"],
            item["question"],
            item["section_text"]
        )
        inside = item["section_start"] <= item["start"] <= item["end"] <= item["section_end"]

        case = find_case(item)

        tests = []

        if item["source"] == "transcript_blank":
            # A fill-in-the-blank question: the prepared explanations do
            # not fit it; its own transcript phrase must be accepted.
            out(f"  Sentence : {item['answer_sentence']}")
            out("  Case     : (fill in the blank; using its own answer)")
            tests.append(("blank", item["answer"], "correct"))

        elif case:
            out(f"  Case     : {case['name']}")
            tests.append(("valid", case["valid"], "correct"))
            tests.append(("partial", case["partial"], "partial"))

            if case.get("vague"):
                tests.append(("vague", case["vague"], "incorrect"))
        else:
            out("  Case     : (no prepared case; using the expected answer)")
            tests.append(("expected", item["answer"], "correct"))

        for answer in UNRELATED_ANSWERS:
            tests.append(("unrelated", answer, "incorrect"))

        for label, answer, wanted in tests:
            result = viva.check_answer(
                answer, item, embedding_model, llm=llm
            )

            ok = result["verdict"] == wanted

            out(
                f"  {'OK ' if ok else 'CHECK'} {label:9} -> {result['verdict']:9} "
                f"({result['kind']}, by {result['judged_by']}, "
                f"sim {result['similarity']:.2f}, cov {result['coverage']:.2f})"
            )
            out(f"        answer  : {answer}")
            out(f"        feedback: {result['feedback']}")

            checks.append((number, label, wanted, result["verdict"], ok))

        out(
            f"  Expected answer answers the question: "
            f"{'yes' if answers_question is None else 'NO - ' + answers_question}"
        )
        out(f"  Timestamp inside its section: {'yes' if inside else 'NO'}")

    out()
    out("=" * 70)
    out("SUMMARY")
    out("=" * 70)

    passed = sum(1 for check in checks if check[4])
    out(f"Answer checks as expected: {passed} / {len(checks)}")

    for number, label, wanted, got, ok in checks:
        if not ok:
            out(f"  Q{number} {label}: expected {wanted}, got {got}")

    report = os.path.join("data", "processed", f"{name}_viva_report.txt")

    with open(report, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print()
    print(f"Report saved: {report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
