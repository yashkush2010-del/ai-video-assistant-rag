"""
Q&A evidence: find_evidence() (src/rag/vector_store.py).

The evidence is the transcript text shown under an answer, with its
Jump time. Measured on two videos (scripts/check_evidence.py), the
right place went from 13 to 15 of 18 questions.

These tests use a simple word-overlap embedder instead of MiniLM, and
real transcript pieces from the machine learning video.
"""

import hashlib
import re
import sys

import numpy as np
import pytest

from src.rag.vector_store import (
    EVIDENCE_PIECES,
    _answer_sentences,
    find_evidence,
)


class WordEmbedder:
    """Bag of words: texts that share words are similar."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        self.calls += 1
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = np.zeros((len(items), 256), dtype="float32")

        for row, text in enumerate(items):
            for word in re.findall(r"[a-z']+", text.lower()):
                if len(word) > 3:
                    column = int(hashlib.md5(word.encode()).hexdigest(), 16) % 256
                    vectors[row, column] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        return vectors[0] if single else vectors


def piece(start, end, text):
    return {"start": start, "end": end, "text": " " + text}


def result_with(*pieces):
    return [{
        "score": 0.5,
        "chunk_index": 0,
        "start": pieces[0]["start"],
        "end": pieces[-1]["end"],
        "text": " ".join(p["text"].strip() for p in pieces),
        "segments": list(pieces),
    }]


# Real pieces, machine learning video 6:24 - 7:24
APPLICATIONS_AND_UBER = [
    piece(384.0, 387.0, "So there are a lot of applications of machine learning out there."),
    piece(387.0, 390.0, "To name a few, machine learning is used in healthcare"),
    piece(390.0, 393.0, "where diagnostics are predicted for doctors' review."),
    piece(409.0, 413.0, "While booking a gap, you must have encountered surge pricing often,"),
    piece(413.0, 416.0, "where it says the fair of your trip has been updated."),
    piece(415.0, 418.0, "Well, that's an interesting machine learning model"),
    piece(418.0, 422.0, "which is used by global taxi giant Uber and others."),
    piece(422.0, 424.0, "Where they have differential pricing in real time,"),
    piece(424.0, 427.0, "based on demand, the number of cars available,"),
    piece(434.0, 437.0, "Also, it uses predictive modeling to predict"),
    piece(437.0, 440.0, "where the demand will be high, with a goal"),
]

UBER_ANSWER = (
    "According to the video, Uber uses machine learning in two ways:\n\n"
    "1. Differential pricing in real time, based on demand, the number of "
    "cars available, bad weather, rush hour, etc.\n"
    "2. Predictive modeling to predict where the demand will be high, so that "
    "drivers can take care of the demand and surge pricing can be minimized."
)


# ============================================================
# The scoring rule: half the question, half the best answer sentence
# ============================================================

class FixedEmbedder:
    """Returns the vector the test chose for each text (others: zero)."""

    def __init__(self, vectors):
        self.vectors = {text: np.asarray(v, dtype="float32") for text, v in vectors.items()}

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        size = len(next(iter(self.vectors.values())))
        rows = []
        for text in items:
            vector = self.vectors.get(text, np.zeros(size, dtype="float32"))
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        rows = np.asarray(rows, dtype="float32")
        return rows[0] if single else rows


def test_the_question_and_the_best_answer_sentence_decide():
    # The real Uber case: the app jumped to "So there are a lot of
    # applications of machine learning out there", which is close to
    # the whole answer but not to the question.
    intro = "So there are a lot of applications of machine learning out there."
    intro_next = "To name a few, machine learning is used in healthcare"
    uber = "which is used by global taxi giant Uber and others."
    uber_next = "Where they have differential pricing in real time,"

    question = "How does Uber use machine learning?"
    sentence = "Uber uses differential pricing in real time based on demand."
    answer = "Here's the answer:\n" + sentence

    embedder = FixedEmbedder({
        f"{intro} {intro_next}": [1.0, 0.0, 0.2],
        f"{intro_next} {uber}": [0.6, 0.6, 0.0],
        f"{uber} {uber_next}": [0.3, 1.0, 1.0],
        question: [0.4, 1.0, 0.0],
        sentence: [0.2, 0.3, 1.0],
        answer: [1.0, 0.1, 0.1],
    })

    evidence = find_evidence(
        result_with(
            piece(384.0, 387.0, intro), piece(387.0, 390.0, intro_next),
            piece(390.0, 393.0, uber), piece(393.0, 396.0, uber_next),
        ),
        question, answer, embedder,
    )[0]

    assert evidence["text"] == f"{uber} {uber_next}"
    assert evidence["start"] == 390.0


def test_the_evidence_is_two_consecutive_pieces_joined():
    evidence = find_evidence(
        result_with(*APPLICATIONS_AND_UBER),
        "How does Uber use machine learning?",
        UBER_ANSWER,
        WordEmbedder(),
    )[0]

    assert EVIDENCE_PIECES == 2

    starts = [p["start"] for p in APPLICATIONS_AND_UBER]
    first = starts.index(evidence["start"])
    second = APPLICATIONS_AND_UBER[first + 1]

    assert evidence["end"] == second["end"]
    assert evidence["text"] == (
        APPLICATIONS_AND_UBER[first]["text"].strip() + " " + second["text"].strip()
    )
    assert isinstance(evidence["score"], float)


# ============================================================
# The question counts, not only the (long) answer
# ============================================================

def test_a_side_topic_in_a_long_answer_does_not_win():
    pieces = [
        piece(6.0, 11.0, "But what if humans can train the machines to learn from their past data"),
        piece(11.0, 14.0, "and do what humans can do at much faster?"),
        piece(14.0, 17.0, "Well, that's called machine learning, but it's a lot more than just learning."),
        piece(215.0, 218.0, "Hence, supervised learning uses labeled data"),
        piece(218.0, 219.0, "to train the model."),
    ]
    answer = (
        "Machine learning means humans train the machines to learn from their "
        "past data and do what humans can do much faster. There are different "
        "types, and supervised learning uses labeled data to train the model."
    )

    evidence = find_evidence(
        result_with(*pieces), "What is machine learning?", answer, WordEmbedder()
    )[0]

    assert evidence["start"] < 20.0


# ============================================================
# Grouping
# ============================================================

def test_pieces_with_a_gap_between_them_are_not_joined():
    pieces = [
        piece(0.0, 3.0, "Packets are small pieces of information."),
        piece(30.0, 33.0, "Routers direct packets to their destination."),
    ]

    evidence = find_evidence(
        result_with(*pieces), "What are packets?",
        "Packets are small pieces of information.", WordEmbedder(),
    )[0]

    # no two pieces follow each other: a single piece, as before
    assert evidence["text"] == "Packets are small pieces of information."
    assert (evidence["start"], evidence["end"]) == (0.0, 3.0)


def test_a_piece_in_two_results_is_used_once():
    shared = piece(10.0, 13.0, "Routers direct packets to their destination.")
    results = (
        result_with(piece(7.0, 10.0, "Anywhere parts of the Internet meet there is a router."), shared)
        + result_with(shared, piece(13.0, 16.0, "Every website visit uses many routers."))
    )

    evidence = find_evidence(
        results, "What does a router do?",
        "A router directs packets to their destination.", WordEmbedder(),
    )[0]

    assert evidence["text"].count("Routers direct packets") == 1


def test_pieces_are_used_in_video_order():
    pieces = [
        piece(13.0, 16.0, "to its destination."),
        piece(10.0, 13.0, "A router helps each packet get one step closer"),
    ]

    evidence = find_evidence(
        result_with(*pieces), "What does a router do?",
        "A router helps each packet get one step closer to its destination.",
        WordEmbedder(),
    )[0]

    assert evidence["text"] == (
        "A router helps each packet get one step closer to its destination."
    )
    assert evidence["start"] == 10.0


# ============================================================
# Nothing to show
# ============================================================

@pytest.mark.parametrize("results, answer", [
    ([], "An answer."),
    (result_with(piece(0.0, 2.0, "Some text.")), ""),
    ([{"score": 0.5, "start": 0.0, "end": 2.0, "text": "x", "segments": []}], "An answer."),
])
def test_no_evidence_when_there_is_nothing_to_choose(results, answer):
    assert find_evidence(results, "Question?", answer, WordEmbedder()) == []


# ============================================================
# Answer sentences
# ============================================================

def test_openers_and_tiny_fragments_are_not_answer_sentences():
    sentences = _answer_sentences(
        "Here's the answer:\n\n1. Gmail sends the email to AOL.\n"
        "2. Aunt Ruth retrieves the email the next day.\nYes."
    )

    # "1." and "2." are split off and, being too short, dropped
    assert sentences == [
        "Gmail sends the email to AOL.",
        "Aunt Ruth retrieves the email the next day.",
    ]


def test_a_one_line_answer_still_works():
    evidence = find_evidence(
        result_with(piece(0.0, 2.0, "Supervised learning uses labeled data"),
                    piece(2.0, 3.0, "to train the model.")),
        "What does supervised learning use?", "Labeled data.", WordEmbedder(),
    )

    assert evidence and evidence[0]["start"] == 0.0


# ============================================================
# The whole app
# ============================================================

def test_the_app_shows_the_joined_evidence(env, monkeypatch):
    import test_youtube_ingestion_app as app_env
    from test_qa_ollama_unavailable import ask

    monkeypatch.setattr(app_env, "SEGMENTS", [
        {"start": 0.0, "end": 3.0, "text": " Cooking pasta needs salted water."},
        {"start": 3.0, "end": 6.0, "text": " Where parts of the Internet meet, routers wait."},
        {"start": 6.0, "end": 9.0, "text": " Routers direct your packets closer to their destination."},
        {"start": 30.0, "end": 33.0, "text": " The weather is sunny on the coast."},
    ])

    env.set_upload("lecture.mp4")
    at = env.start()

    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: {"message": {"content":
            "Routers direct your packets closer to their destination."}},
    )

    at = ask(at, "What do routers do with your packets?")

    assert not at.exception, [e.value for e in at.exception]

    evidence = at.session_state["last_evidence_segments"]

    assert len(evidence) == 1
    assert evidence[0]["start"] == 3.0
    assert evidence[0]["text"] == (
        "Where parts of the Internet meet, routers wait. "
        "Routers direct your packets closer to their destination."
    )


from test_youtube_ingestion_app import env  # noqa: E402,F401
