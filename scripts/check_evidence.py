"""
Compares ways of choosing the Q&A evidence (the sentence and Jump time
shown under an answer).

The evidence is one transcript sentence from the parts the answer was
made from. Today it is the sentence most similar to the ANSWER; for a
long answer that mentions several topics this often lands on the wrong
sentence (e.g. "What is machine learning?" -> 3:35 "Hence, supervised
learning uses labeled data").

This script needs NO Ollama: it takes the answers saved by
scripts/check_qa_real_video.py (data/processed/<video>_qa_report_<label>.json),
rebuilds the same search as app.py, and lets every method pick an
evidence sentence. A pick is correct when it starts inside one of the
windows in the question file (the third part of a line), or at most
EARLY_TOLERANCE seconds before one.

Methods:
  A  answer         most similar to the whole answer (the app today)
  B  question       most similar to the question
  C  both           half question, half whole answer
  D  first sentence half question, half the answer's first real sentence
  E  both, found    like C, but only sentences from the parts the search
                    really found (not the added neighbours)
  F  best sentence  half question, half the best-matching answer sentence
  G  2 pieces       like F, but the evidence is 2 consecutive transcript
                    pieces (Whisper often cuts a sentence into short pieces)
  H  3 pieces       like F, with 3 consecutive pieces
  I  3 pieces, A    like A (whole answer only), with 3 consecutive pieces

Usage (from the project folder, with the virtual environment active):

    python scripts/check_evidence.py
    python scripts/check_evidence.py --label final
    python scripts/check_evidence.py --questions scripts/qa_questions/internet.txt
"""

import argparse
import json
import os
import re
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
os.chdir(PROJECT_DIR)

import numpy as np  # noqa: E402

from check_qa_real_video import (  # noqa: E402
    EMBEDDING_MODEL,
    RETRIEVE_K,
    RETRIEVE_THRESHOLD,
    load_segments,
    read_questions,
)
from src.processing.chunker import create_chunks  # noqa: E402
from src.rag.vector_store import (  # noqa: E402
    add_neighbouring_chunks,
    create_index,
    retrieve_chunks,
)
from src.rag.video_summary import format_timestamp  # noqa: E402


QUESTION_FILES = [
    os.path.join("scripts", "qa_questions", "machine_learning.txt"),
    os.path.join("scripts", "qa_questions", "internet.txt"),
]

METHODS = {
    "A": "answer (app today)",
    "B": "question",
    "C": "question + answer",
    "D": "question + first answer sentence",
    "E": "question + answer, found parts only",
    "F": "question + best answer sentence",
    "G": "question + best answer sentence, 2 pieces",
    "H": "question + best answer sentence, 3 pieces",
    "I": "answer, 3 pieces",
}

# Pieces at most this far apart (seconds) are consecutive speech.
PIECE_GAP = 1.0

# A jump that starts up to this many seconds BEFORE the right place
# still lands on it for the viewer. Used for every method alike.
EARLY_TOLERANCE = 5


def lands_right(start, windows):
    return any(first - EARLY_TOLERANCE <= int(start) < last for first, last in windows)

GROUP_SIZES = {"G": 2, "H": 3, "I": 3}

# Openers that are not the real first sentence of an answer.
_OPENER = re.compile(r"^(here'?s the answer|according to the video)[:.,]?\s*$", re.I)


def answer_sentences(answer):
    sentences = [
        part.strip(" -*\t")
        for part in re.split(r"(?<=[.!?:])\s+|\n+", str(answer))
    ]
    return [
        sentence for sentence in sentences
        if len(sentence.split()) >= 4 and not _OPENER.match(sentence)
    ]


def group_pieces(candidates, size):
    """
    Every run of `size` consecutive pieces as one candidate:
    {"start" of the first, "end" of the last, "text" joined}.
    Falls back to single pieces when there is no such run.
    """
    pieces = []
    seen = set()

    for segment, found in sorted(candidates, key=lambda c: float(c[0]["start"])):
        key = (float(segment["start"]), segment["text"].strip())
        if key not in seen:
            seen.add(key)
            pieces.append((segment, found))

    groups = []

    for first in range(len(pieces) - size + 1):
        run = pieces[first:first + size]

        consecutive = all(
            float(later[0]["start"]) - float(earlier[0]["end"]) <= PIECE_GAP
            for earlier, later in zip(run, run[1:])
        )

        if consecutive:
            groups.append((
                {
                    "start": float(run[0][0]["start"]),
                    "end": float(run[-1][0]["end"]),
                    "text": " ".join(piece[0]["text"].strip() for piece in run),
                },
                any(piece[1] for piece in run),
            ))

    return groups or pieces


def pick(method, question, answer, results, found_indexes, model):
    candidates = []

    for result in results:
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if text:
                candidates.append(
                    (segment, result.get("chunk_index") in found_indexes)
                )

    if method == "E":
        only_found = [c for c in candidates if c[1]]
        candidates = only_found or candidates

    if method in GROUP_SIZES:
        candidates = group_pieces(candidates, GROUP_SIZES[method])

    if not candidates:
        return None

    texts = [segment["text"].strip() for segment, _ in candidates]
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True))

    def similarity(text):
        return vectors @ np.asarray(model.encode(text, normalize_embeddings=True))

    sentences = answer_sentences(answer) or [answer]

    if method in ("A", "I"):
        scores = similarity(answer)
    elif method == "B":
        scores = similarity(question)
    elif method in ("C", "E"):
        scores = 0.5 * similarity(question) + 0.5 * similarity(answer)
    elif method == "D":
        scores = 0.5 * similarity(question) + 0.5 * similarity(sentences[0])
    else:  # F, G, H
        best = np.max(
            np.vstack([similarity(sentence) for sentence in sentences]), axis=0
        )
        scores = 0.5 * similarity(question) + 0.5 * best

    return candidates[int(np.argmax(scores))][0]


def check_video(question_file, label, model):
    video_path, questions = read_questions(question_file)
    name = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join("data", "processed", f"{name}_qa_report_{label}.json")

    if not os.path.exists(report_path):
        sys.exit(
            f"No saved answers: {report_path}\n"
            f"Run scripts/check_qa_real_video.py with --label {label} first."
        )

    with open(report_path, encoding="utf-8") as file:
        answers = {r["question"]: r["answer"] for r in json.load(file)}

    chunks = create_chunks(load_segments(video_path))
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = create_index(embeddings)

    rows = []

    for item in questions:
        if not item["answerable"] or not item["windows"]:
            continue

        answer = answers.get(item["question"])
        if answer is None:
            continue

        found = retrieve_chunks(
            index, chunks,
            model.encode(item["question"], normalize_embeddings=True),
            k=RETRIEVE_K, threshold=RETRIEVE_THRESHOLD,
        )
        found_indexes = {r["chunk_index"] for r in found}
        results = add_neighbouring_chunks(found, chunks)

        picks = {}
        for method in METHODS:
            segment = pick(method, item["question"], answer, results, found_indexes, model)
            picks[method] = (
                segment,
                bool(segment) and lands_right(segment["start"], item["windows"]),
            )

        rows.append((item["question"], picks))

    return video_path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--label", default="final",
                        help="which saved answers to use (default: final)")
    parser.add_argument("--questions", nargs="+", default=QUESTION_FILES,
                        help="question files (default: both test videos)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    print("Loading the embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    totals = {method: 0 for method in METHODS}
    count = 0
    lines = []

    for question_file in args.questions:
        video_path, rows = check_video(question_file, args.label, model)

        lines += ["", "=" * 70, video_path, "=" * 70]

        for question, picks in rows:
            count += 1
            lines.append(f"\n{question}")

            for method, (segment, correct) in picks.items():
                totals[method] += int(correct)
                if segment is None:
                    lines.append(f"  {method}  -")
                    continue
                lines.append(
                    f"  {method} {'ok   ' if correct else 'WRONG'} "
                    f"[{format_timestamp(segment['start'])}] {segment['text'].strip()[:70]}"
                )

    lines += ["", "=" * 70, f"RIGHT PLACE (of {count} questions)", "=" * 70]
    for method, title in METHODS.items():
        lines.append(f"  {method}  {totals[method]:>2} / {count}   {title}")

    text = "\n".join(lines)
    print(text)

    output = os.path.join("data", "processed", f"evidence_check_{args.label}.txt")
    with open(output, "w", encoding="utf-8") as file:
        file.write(text + "\n")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
