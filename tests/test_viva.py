"""
Viva Mode (src/rag/viva.py).

No model is loaded: the LLM is a plain function that returns fixed
text, and the embedding model is a small bag-of-words stand-in, so
these tests run without Ollama, Whisper or MiniLM.
"""

import hashlib
import re

import numpy as np
import pytest

from src.rag import viva
from src.rag.viva import (
    check_answer,
    generate_viva_questions,
    viva_score,
)


# ============================================================
# STAND-INS
# ============================================================

class FakeEmbedder:
    """Same idea as the stand-in in the app tests: one column per word."""

    def encode(self, texts, normalize_embeddings=False, **kwargs):
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


def fixed_llm(reply):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return reply

    llm.calls = calls
    return llm


def broken_llm(prompt):
    raise ConnectionError("Ollama is not running")


@pytest.fixture(autouse=True)
def no_real_ollama(monkeypatch):
    """
    On a machine where Ollama runs, a test that does not pass llm=
    would otherwise ask the real model. These tests never do.
    """
    monkeypatch.setattr(viva, "_default_llm", broken_llm)
    monkeypatch.setattr(viva, "_judge_llm", broken_llm)


# Round 4: when the LLM's own question is not used, the question now
# comes from the one-sentence LLM step or is a fill-in-the-blank question
# (before: the broad "What does the video explain about ...?" form).
FALLBACK_SOURCES = ("llm_answer_sentence", "transcript_blank", "transcript")


def answer_sentence(item):
    """The transcript sentence of a question (whole sentence of a blank)."""
    return item.get("answer_sentence") or item["answer"]


def assert_llm_pair_not_used(item, rejected_text):
    """The LLM's own question/answer pair was rejected and not shown."""
    assert item["source"] != "llm"
    assert rejected_text not in item["question"]
    assert rejected_text not in item["answer"]


POINTS = [
    {
        "title": "Supervised Learning",
        "summary": "Supervised learning trains a model on labelled examples.",
        "start": 12.0,
        "end": 31.5,
        "text": (
            "Supervised learning trains a model on labelled examples. "
            "Each example has an input and the correct output."
        ),
    },
    {
        "title": "Overfitting",
        "summary": "Overfitting happens when a model memorises the training data.",
        "start": 95.0,
        "end": 118.0,
        "text": (
            "Overfitting happens when a model memorises the training data "
            "and then performs badly on new data."
        ),
    },
]


# ============================================================
# QUESTIONS
# ============================================================

def test_a_grounded_llm_question_is_used():
    llm = fixed_llm(
        "QUESTION: What does supervised learning train a model on?\n"
        "ANSWER: It trains a model on labelled examples."
    )

    questions = generate_viva_questions(POINTS[:1], llm=llm)

    assert len(questions) == 1
    assert questions[0]["question"] == (
        "What does supervised learning train a model on?"
    )
    assert questions[0]["answer"] == "It trains a model on labelled examples."
    assert questions[0]["source"] == "llm"


def test_timestamps_and_evidence_are_copied_from_the_point():
    llm = fixed_llm(
        "QUESTION: What does supervised learning train a model on?\n"
        "ANSWER: It trains a model on labelled examples."
    )

    item = generate_viva_questions(POINTS[:1], llm=llm)[0]

    assert item["start"] == POINTS[0]["start"]
    assert item["end"] == POINTS[0]["end"]
    assert item["text"] == POINTS[0]["text"]
    assert item["title"] == POINTS[0]["title"]


def test_an_answer_not_supported_by_the_transcript_is_rejected():
    llm = fixed_llm(
        "QUESTION: What does supervised learning train a model on?\n"
        "ANSWER: Reinforcement agents receive rewards from environments."
    )

    item = generate_viva_questions(POINTS[:1], llm=llm)[0]

    assert_llm_pair_not_used(item, "Reinforcement agents")


PAUL_POINT = {
    "title": "Basics Of Machine Learning",
    "summary": "He loves listening to new songs.",
    "start": 17.0,
    "end": 31.0,
    "text": (
        "It's also about understanding and reasoning. So today we will "
        "learn about the basics of machine learning. So that's Paul. He "
        "loves listening to new songs. He either likes them or dislikes them."
    ),
}


def test_the_real_meta_answer_from_llama_is_rejected():
    # Exactly what llama3.2 produced on the real Machine Learning video.
    llm = fixed_llm(
        "QUESTION: What does Paul use machine learning for?\n"
        "ANSWER: He doesn't explicitly mention what machine learning is "
        "used for, but it is stated that he either likes or dislikes "
        "new songs."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert_llm_pair_not_used(item, "explicitly")


@pytest.mark.parametrize("answer", [
    "This is not mentioned in the excerpt.",
    "The transcript says he likes songs.",
    "The video does not explain why he likes songs.",
    "It is not clearly stated, but he likes new songs.",
    "He didn’t say which songs he likes.",
])
def test_answers_that_talk_about_the_transcript_are_rejected(answer):
    llm = fixed_llm(
        "QUESTION: What does Paul love listening to?\n"
        f"ANSWER: {answer}"
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert_llm_pair_not_used(item, answer)


def test_an_answer_that_only_repeats_the_question_is_rejected():
    # Exactly what llama3.2 produced on the real Machine Learning video.
    llm = fixed_llm(
        "QUESTION: What is Paul's primary reason for liking or "
        "disliking new songs?\n"
        "ANSWER: He likes them or dislikes them."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] != "llm"
    assert item["answer"] != "He likes them or dislikes them."


def test_with_the_fallback_a_vague_answer_is_no_longer_correct():
    llm = fixed_llm(
        "QUESTION: What is Paul's primary reason for liking or "
        "disliking new songs?\n"
        "ANSWER: He likes them or dislikes them."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    result = check_answer("Paul likes songs.", item, FakeEmbedder())

    assert result["verdict"] != "correct"


def test_a_question_that_assumes_what_the_video_does_not_say_is_rejected():
    # Exactly what llama3.2 produced on the real Machine Learning video.
    llm = fixed_llm(
        "QUESTION: What is Paul primarily using his machine learning "
        "knowledge for?\n"
        "ANSWER: Paul loves listening to new songs."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES


@pytest.mark.parametrize("question", [
    "What does Paul love listening to?",
    "How does Paul react to new songs?",
    "What does Paul think of new songs?",
])
def test_a_question_with_at_most_one_new_word_is_kept(question):
    llm = fixed_llm(
        f"QUESTION: {question}\n"
        "ANSWER: He either likes them or dislikes them."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] == "llm"


def test_a_possessive_matches_the_name():
    from src.rag.viva import _unsupported_words

    assert _unsupported_words(
        "What are Paul's songs?",
        "So that's Paul. He loves new songs."
    ) == set()


@pytest.mark.parametrize("question", [
    # Exactly what llama3.2 produced on the real Machine Learning video.
    "Does Paul like listening to new songs?",
    "Is supervised learning trained on labelled examples?",
    "Can a model learn from labelled examples?",
    "Are labelled examples used in supervised learning?",
])
def test_yes_no_questions_are_rejected(question):
    llm = fixed_llm(
        f"QUESTION: {question}\n"
        "ANSWER: Yes, he loves listening to new songs."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES


def test_a_question_starting_with_a_similar_word_is_not_yes_no():
    from src.rag.viva import _is_yes_no_question

    assert not _is_yes_no_question("Dozens of what are shown?")
    assert not _is_yes_no_question("Isolation of what is explained?")
    assert not _is_yes_no_question("What does Paul love listening to?")
    assert _is_yes_no_question("does Paul like songs?")


@pytest.mark.parametrize("title, topic", [
    # Exactly the title from the real Machine Learning video.
    ("Supervised Learning Explained", "supervised learning"),
    ("Machine Learning Basics", "machine learning basics"),
    ("Introduction to Neural Networks", "neural networks"),
    ("RAG Overview", "RAG"),
    ("Explained", "this part"),
    ("", "this part"),
])
def test_the_fallback_question_reads_naturally(title, topic):
    point = dict(POINTS[0], title=title)

    assert viva._fallback_question(point) == (
        f"What does the video explain about {topic}?"
    )


def test_a_question_with_two_new_words_is_kept():
    llm = fixed_llm(
        "QUESTION: What is the main type of songs Paul loves?\n"
        "ANSWER: Paul loves listening to new songs."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] == "llm"


def test_a_question_about_the_transcript_is_rejected():
    llm = fixed_llm(
        "QUESTION: What does the excerpt say about Paul and new songs?\n"
        "ANSWER: Paul loves listening to new songs."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES


def test_a_direct_answer_about_paul_is_kept():
    llm = fixed_llm(
        "QUESTION: What does Paul love listening to?\n"
        "ANSWER: Paul loves listening to new songs."
    )

    item = generate_viva_questions([PAUL_POINT], llm=llm)[0]

    assert item["source"] == "llm"


def test_explicitly_programmed_is_real_content_not_meta():
    # The classic definition of machine learning must not be rejected.
    point = {
        "title": "Machine Learning",
        "summary": "Machines learn from data without being explicitly programmed.",
        "start": 21.0,
        "end": 40.0,
        "text": (
            "Machine learning lets machines learn from data without "
            "being explicitly programmed."
        ),
    }

    llm = fixed_llm(
        "QUESTION: How do machines learn in machine learning?\n"
        "ANSWER: They learn from data without being explicitly programmed."
    )

    item = generate_viva_questions([point], llm=llm)[0]

    assert item["source"] == "llm"


# ============================================================
# EXACT TIMESTAMP OF THE ANSWER
# ============================================================

# The real Whisper segments of the 0:17 - 0:31 section.
PAUL_SEGMENTS = [
    {"start": 17.0, "end": 20.5,
     "text": " It's also about understanding and reasoning."},
    {"start": 20.5, "end": 24.0,
     "text": " So today we will learn about the basics of machine learning."},
    {"start": 24.0, "end": 26.0,
     "text": " So that's Paul."},
    {"start": 26.0, "end": 28.5,
     "text": " He loves listening to new songs."},
    {"start": 28.5, "end": 31.0,
     "text": " He either likes them or dislikes them."},
    # outside the section: must never be used
    {"start": 31.0, "end": 35.0,
     "text": " Paul loves listening to new songs every day."},
]


def paul_question(llm_reply=""):
    return generate_viva_questions(
        [PAUL_POINT],
        llm=fixed_llm(llm_reply),
        segments=PAUL_SEGMENTS,
        embedding_model=FakeEmbedder()
    )[0]


def test_the_timestamp_points_at_the_answer_not_the_section_start():
    item = paul_question()

    # Fallback answer: "He loves listening to new songs."
    assert item["start"] == 26.0
    assert item["end"] == 28.5
    assert item["text"] == "He loves listening to new songs."


def test_the_whole_section_is_still_known():
    item = paul_question()

    assert item["section_start"] == 17.0
    assert item["section_end"] == 31.0


def test_an_llm_answer_is_located_too():
    item = paul_question(
        "QUESTION: How does Paul react to new songs?\n"
        "ANSWER: He either likes them or dislikes them."
    )

    assert item["source"] == "llm"
    assert item["start"] == 28.5
    assert item["end"] == 31.0


def test_segments_outside_the_section_are_never_used():
    item = paul_question()

    assert 17.0 <= item["start"] < item["end"] <= 31.0


def test_a_half_sentence_segment_is_joined_with_the_next():
    segments = [
        {"start": 17.0, "end": 22.0,
         "text": " It's also about understanding and reasoning."},
        {"start": 22.0, "end": 26.0,
         "text": " He loves listening to"},
        {"start": 26.0, "end": 31.0,
         "text": " new songs every single day."},
    ]

    item = generate_viva_questions(
        [PAUL_POINT],
        llm=broken_llm,
        segments=segments,
        embedding_model=FakeEmbedder()
    )[0]

    assert item["start"] == 22.0
    assert item["end"] == 31.0
    assert item["text"] == "He loves listening to new songs every single day."


def test_without_segments_the_whole_section_is_used():
    item = generate_viva_questions([PAUL_POINT], llm=broken_llm)[0]

    assert item["start"] == PAUL_POINT["start"]
    assert item["end"] == PAUL_POINT["end"]
    assert item["text"] == PAUL_POINT["text"]


def test_a_transcript_answer_is_located_without_the_embedding_model():
    class Broken:
        def encode(self, *args, **kwargs):
            raise RuntimeError("no model")

    item = generate_viva_questions(
        [PAUL_POINT],
        llm=broken_llm,
        segments=PAUL_SEGMENTS,
        embedding_model=Broken()
    )[0]

    # "He loves listening to new songs." is found word for word
    assert answer_sentence(item) == "He loves listening to new songs."
    assert item["answer"] == "new songs"
    assert item["start"] == 26.0
    assert item["end"] == 28.5


def test_a_broken_embedding_model_keeps_the_whole_section():
    class Broken:
        def encode(self, *args, **kwargs):
            raise RuntimeError("no model")

    # An LLM-written answer (not word for word in the transcript)
    item = generate_viva_questions(
        [PAUL_POINT],
        llm=fixed_llm(
            "QUESTION: What does Paul love listening to?\n"
            "ANSWER: Paul loves listening to new songs."
        ),
        segments=PAUL_SEGMENTS,
        embedding_model=Broken()
    )[0]

    assert item["source"] == "llm"
    assert item["start"] == PAUL_POINT["start"]
    assert item["text"] == PAUL_POINT["text"]


def test_a_question_without_a_question_mark_is_rejected():
    llm = fixed_llm(
        "QUESTION: Explain supervised learning\n"
        "ANSWER: It trains a model on labelled examples."
    )

    item = generate_viva_questions(POINTS[:1], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES


def test_unformatted_llm_output_falls_back_to_a_plain_question():
    llm = fixed_llm("Sure! Here is a great question for you.")

    item = generate_viva_questions(POINTS[:1], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES
    assert "great question" not in item["question"]


def test_without_ollama_every_point_still_gets_a_question():
    questions = generate_viva_questions(POINTS, llm=broken_llm)

    assert len(questions) == 2
    assert all(item["source"] in FALLBACK_SOURCES for item in questions)


def test_the_llm_is_not_retried_after_a_failure():
    calls = []

    def failing(prompt):
        calls.append(prompt)
        raise ConnectionError("down")

    generate_viva_questions(POINTS, llm=failing)

    assert len(calls) == 1


def test_only_the_first_of_several_questions_is_kept():
    llm = fixed_llm(
        "QUESTION: What does supervised learning train a model on? "
        "And why is that useful?\n"
        "ANSWER: It trains a model on labelled examples."
    )

    item = generate_viva_questions(POINTS[:1], llm=llm)[0]

    assert item["question"] == (
        "What does supervised learning train a model on?"
    )


def test_markdown_formatting_from_the_model_is_ignored():
    llm = fixed_llm(
        "**QUESTION:** What does supervised learning train a model on?\n"
        "**ANSWER:** It trains a model on labelled examples."
    )

    item = generate_viva_questions(POINTS[:1], llm=llm)[0]

    assert item["source"] == "llm"


def test_a_repeated_question_is_skipped():
    same_point = dict(POINTS[0])
    same_point["start"] = 200.0

    llm = fixed_llm(
        "QUESTION: What does supervised learning train a model on?\n"
        "ANSWER: It trains a model on labelled examples."
    )

    questions = generate_viva_questions([POINTS[0], same_point], llm=llm)

    assert len(questions) == 1


def test_the_number_of_questions_is_limited():
    topics = [
        "Regression", "Classification", "Clustering", "Overfitting",
        "Neural Networks", "Decision Trees", "Embeddings", "Evaluation",
    ]

    sentences = [
        "Regression predicts house prices from the size of each house.",
        "Classification sorts incoming emails into spam and normal mail.",
        "Clustering groups customers by their shopping habits.",
        "Overfitting memorises training noise and fails on new data.",
        "Neural networks stack many layers of simple neurons.",
        "Decision trees split the data with yes or no conditions.",
        "Embeddings turn words into vectors of real numbers.",
        "Evaluation measures accuracy on a separate test set.",
    ]

    points = [
        dict(POINTS[0], title=topic, text=sentence, summary=sentence,
             start=float(i * 30))
        for i, (topic, sentence) in enumerate(zip(topics, sentences))
    ]

    questions = generate_viva_questions(
        points,
        llm=broken_llm,
        max_questions=3
    )

    assert len(questions) == 3


def test_questions_are_in_video_order():
    questions = generate_viva_questions(
        list(reversed(POINTS)),
        llm=broken_llm
    )

    starts = [item["start"] for item in questions]

    assert starts == sorted(starts)


def test_no_points_means_no_questions():
    assert generate_viva_questions([], llm=broken_llm) == []
    assert generate_viva_questions(None, llm=broken_llm) == []


def test_a_point_without_text_is_skipped():
    empty = dict(POINTS[0], text="", summary="")

    assert generate_viva_questions([empty], llm=broken_llm) == []


def test_the_prompt_contains_only_the_transcript_of_that_point():
    llm = fixed_llm("")

    generate_viva_questions(POINTS[:1], llm=llm)

    assert POINTS[0]["text"] in llm.calls[0]
    assert POINTS[1]["text"] not in llm.calls[0]


# ============================================================
# CHECKING ANSWERS
# ============================================================

ITEM = {
    "question": "What does supervised learning train a model on?",
    "answer": "Supervised learning trains a model on labelled examples.",
    "start": 12.0,
    "end": 31.5,
    "text": POINTS[0]["text"],
    "title": "Supervised Learning",
    "source": "llm",
}


def test_a_correct_answer_is_marked_correct():
    result = check_answer(
        "It trains the model using labelled examples.",
        ITEM,
        FakeEmbedder()
    )

    assert result["verdict"] == "correct"
    assert result["score"] == 1.0
    assert result["missing"] == []


def test_a_partly_correct_answer_is_marked_partial():
    result = check_answer(
        "The model learns from many examples.",
        ITEM,
        FakeEmbedder()
    )

    assert result["verdict"] == "partial"
    assert result["score"] == 0.5
    # "labelled" and "labeled" are the same word
    assert "labeled" in result["missing"]
    assert "labeled" in result["feedback"]


def test_a_wrong_answer_is_marked_incorrect():
    result = check_answer(
        "Clustering groups unlabelled points by distance.",
        ITEM,
        FakeEmbedder()
    )

    assert result["verdict"] == "incorrect"
    assert result["score"] == 0.0


def test_an_empty_answer_is_incorrect():
    for answer in ("", "   ", None, "the and of"):
        result = check_answer(answer, ITEM, FakeEmbedder())

        assert result["verdict"] == "incorrect"
        assert result["feedback"] == "No answer was given."


def test_repeating_the_question_is_not_an_answer():
    result = check_answer(
        "What does supervised learning train a model on",
        ITEM,
        FakeEmbedder()
    )

    assert result["verdict"] == "incorrect"
    assert "repeats the question" in result["feedback"]


def test_words_already_in_the_question_are_not_required():
    result = check_answer(
        "The model learns from many examples.",
        ITEM,
        FakeEmbedder()
    )

    # "supervised" and "learning" are in the question itself
    assert "supervised" not in result["missing"]
    assert "learning" not in result["missing"]
    assert "labeled" in result["missing"]


def test_checking_works_without_the_llm():
    # (the fixture above makes every LLM call fail)
    result = check_answer(
        "It trains the model using labelled examples.",
        ITEM,
        FakeEmbedder()
    )

    assert result["verdict"] == "correct"


def test_checking_gives_the_same_result_every_time():
    first = check_answer("The model learns from many examples.", ITEM, FakeEmbedder())
    second = check_answer("The model learns from many examples.", ITEM, FakeEmbedder())

    assert first == second


def test_scores_are_between_zero_and_one():
    for answer in (
        "It trains the model using labelled examples.",
        "It uses examples.",
        "Clustering groups unlabelled points.",
    ):
        result = check_answer(answer, ITEM, FakeEmbedder())

        assert 0.0 <= result["similarity"] <= 1.0 + 1e-6
        assert 0.0 <= result["coverage"] <= 1.0


# ============================================================
# FINAL SCORE
# ============================================================

def test_final_score_adds_up_verdicts():
    results = {
        0: {"score": 1.0},
        1: {"score": 0.5},
        2: {"score": 0.0},
    }

    final = viva_score(results, 4)

    assert final == {"score": 1.5, "total": 4, "percent": 38}


def test_final_score_with_no_questions():
    assert viva_score({}, 0) == {"score": 0, "total": 0, "percent": 0}


@pytest.mark.parametrize("verdict, points", [
    ("correct", 1.0),
    ("partial", 0.5),
    ("incorrect", 0.0),
])
def test_verdict_points(verdict, points):
    assert viva.VERDICT_SCORES[verdict] == points


# ============================================================
# REAL-VIDEO CASES  (Machine Learning video, 7:51)
# ============================================================
#
# The sections below follow the coins (supervised learning, about
# 2:53) and cricket (unsupervised learning, about 3:50 - 4:31) parts of
# the real test video. Their wording is reconstructed from the
# transcript lines the app saved; the script
# scripts/check_viva_real_video.py checks the exact real transcript.

COINS_TEXT = (
    "Suppose your friend gives you one million coins of three different "
    "currencies, say one rupee, one euro and one dirham. Each coin has "
    "different weights. Here, the weight becomes the feature of coins, "
    "while currency becomes their label. When you feed this data to the "
    "machine learning model, it learns which feature is associated with "
    "which label. Let's give a new coin to the machine. On the basis of "
    "the weight of the new coin, your model will predict the currency. "
    "Hence, supervised learning uses labeled data to train the model."
)

CRICKET_TEXT = (
    "Suppose you have a cricket dataset of various players with their "
    "respective scores and wickets taken. When you feed this data set to "
    "the machine, the machine identifies the pattern of player "
    "performance. While looking at the data, you will see that there are "
    "two clusters. One cluster is of the players who scored higher runs "
    "and took less wickets, while the other cluster is of the players who "
    "scored less runs but took many wickets. Hence, the learning with "
    "unlabeled data is unsupervised learning."
)

COINS_QUESTION = {
    # Exactly the question and expected answer seen on the real video.
    "question": "What does the video explain about supervised learning?",
    "answer": (
        "Suppose your friend gives you one million coins of three "
        "different currencies."
    ),
    "kind": "concept",
    "start": 173.0,
    "end": 178.0,
    "text": "Suppose your friend gives you one million coins of three different currencies.",
    "section_start": 173.0,
    "section_end": 232.0,
    "section_text": COINS_TEXT,
    "title": "Supervised Learning",
    "source": "transcript",
}

COINS_EXAMPLE_QUESTION = dict(
    COINS_QUESTION,
    question="What becomes the feature and the label of the coins in the video?",
    answer="The weight becomes the feature of coins, while currency becomes their label.",
    kind="example",
)

CRICKET_QUESTION = {
    "question": "How does the machine group the cricket players?",
    "answer": (
        "One cluster is of the players who scored higher runs and took "
        "less wickets, while the other cluster is of the players who "
        "scored less runs but took many wickets."
    ),
    "kind": "example",
    "start": 240.0,
    "end": 258.0,
    "text": CRICKET_TEXT,
    "section_start": 233.0,
    "section_end": 271.0,
    "section_text": CRICKET_TEXT,
    "title": "Types Of Players",
    "source": "llm",
}

DEFINITION = (
    "Supervised learning trains a model using labelled data, where each "
    "input comes with the correct output."
)


def examiner(verdict, answer_type, missing="NONE"):
    """A stand-in LLM examiner that records what it was asked."""
    return fixed_llm(
        f"VERDICT: {verdict}\nTYPE: {answer_type}\nMISSING: {missing}"
    )


# ---- valid alternative explanations ---------------------------------

def test_the_real_q2_definition_is_accepted_as_a_correct_concept():
    # The case that was marked incorrect on the real video.
    result = check_answer(
        DEFINITION,
        COINS_QUESTION,
        FakeEmbedder(),
        llm=examiner("CORRECT", "CONCEPT")
    )

    assert result["verdict"] == "correct"
    assert result["kind"] == "concept"
    assert result["judged_by"] == "llm"
    assert "correct explanation of the idea" in result["feedback"]


def test_a_correct_description_of_the_video_example_is_accepted():
    result = check_answer(
        "The weight of a coin is the feature and its currency is the label.",
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=examiner("CORRECT", "EXAMPLE")
    )

    assert result["verdict"] == "correct"
    assert result["kind"] in ("example", "video")


def test_an_answer_worded_like_another_sentence_of_the_section_counts():
    # Not the expected answer, but what the same section says.
    result = check_answer(
        "Supervised learning uses labeled data to train the model.",
        COINS_QUESTION,
        FakeEmbedder(),
        llm=broken_llm
    )

    assert result["verdict"] == "correct"
    assert result["kind"] == "video"


# ---- video-specific questions need the video, not outside knowledge ---

def test_outside_knowledge_alone_is_partial_for_an_example_question():
    result = check_answer(
        DEFINITION,
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=examiner("CORRECT", "CONCEPT")
    )

    assert result["verdict"] == "partial"
    assert result["kind"] == "general_only"
    assert "what the video itself shows" in result["feedback"]


def test_the_examiner_is_told_which_kind_of_question_it_is():
    concept_llm = examiner("CORRECT", "CONCEPT")
    example_llm = examiner("CORRECT", "EXAMPLE")

    check_answer(DEFINITION, COINS_QUESTION, FakeEmbedder(), llm=concept_llm)
    check_answer(
        "The weight is the feature and the currency is the label.",
        COINS_EXAMPLE_QUESTION, FakeEmbedder(), llm=example_llm
    )

    assert "understanding of an IDEA" in concept_llm.calls[0]
    assert "what THIS VIDEO says" in example_llm.calls[0]


# ---- unrelated answers are never accepted ----------------------------

@pytest.mark.parametrize("answer", [
    "I like playing cricket on weekends.",
    "The capital of France is Paris.",
    "donot know",
])
def test_unrelated_answers_are_rejected_even_if_the_llm_says_correct(answer):
    llm = examiner("CORRECT", "CONCEPT")

    result = check_answer(answer, COINS_QUESTION, FakeEmbedder(), llm=llm)

    assert result["verdict"] == "incorrect"

    # the LLM was not even asked
    assert llm.calls == []


def test_only_the_question_own_section_is_evidence():
    # A true sentence of the video - but of the CRICKET part.
    cricket_answer = (
        "One cluster is of the players who scored higher runs and took "
        "less wickets."
    )

    for_coins = check_answer(
        cricket_answer, COINS_QUESTION, FakeEmbedder(), llm=broken_llm
    )
    for_cricket = check_answer(
        cricket_answer, CRICKET_QUESTION, FakeEmbedder(), llm=broken_llm
    )

    assert for_coins["verdict"] == "incorrect"
    assert for_cricket["verdict"] in ("correct", "partial")


def test_the_examiner_only_sees_the_question_own_section():
    llm = examiner("PARTIAL", "INCOMPLETE")

    check_answer(DEFINITION, COINS_QUESTION, FakeEmbedder(), llm=llm)

    assert "one million coins" in llm.calls[0]
    assert "cricket" not in llm.calls[0]


# ---- meaning, not shared words ----------------------------------------

def test_an_answer_that_says_the_opposite_is_incorrect():
    # Shares almost every word with the video, but says "unlabeled".
    result = check_answer(
        "Supervised learning uses unlabeled data to train the model.",
        dict(COINS_QUESTION, answer="Hence, supervised learning uses labeled data to train the model."),
        FakeEmbedder(),
        llm=examiner("INCORRECT", "WRONG")
    )

    assert result["verdict"] == "incorrect"


def test_the_video_own_answer_is_never_failed_by_the_examiner():
    result = check_answer(
        COINS_EXAMPLE_QUESTION["answer"],
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=examiner("INCORRECT", "WRONG")
    )

    assert result["verdict"] == "correct"


def test_partially_correct_is_not_read_as_correct():
    result = check_answer(
        "The model learns which feature goes with which label.",
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=fixed_llm("VERDICT: PARTIALLY CORRECT\nTYPE: INCOMPLETE\nMISSING: NONE")
    )

    assert result["verdict"] == "partial"


# ---- partial answers: missing CONCEPTS, not missing words -------------

def test_a_partial_answer_names_the_missing_concept():
    result = check_answer(
        "The model learns which feature goes with which label.",
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=examiner("PARTIAL", "INCOMPLETE", "the weight is the feature and currency is the label")
    )

    assert result["verdict"] == "partial"
    assert result["missing_concept"] == (
        "The weight is the feature and currency is the label"
    )
    assert "Missing: The weight is the feature" in result["feedback"]
    assert "Key points to mention" not in result["feedback"]


def test_a_missing_concept_the_video_does_not_say_is_not_shown():
    result = check_answer(
        "The model learns which feature goes with which label.",
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=examiner("PARTIAL", "INCOMPLETE", "gradient descent and backpropagation")
    )

    assert result["verdict"] == "partial"
    assert result["missing_concept"] is None
    assert "gradient" not in result["feedback"]


def test_filler_words_are_never_key_points():
    item = dict(
        CRICKET_QUESTION,
        question="What did the video say about the types of learning?",
        answer=(
            "So we saw supervised learning where the data was labeled "
            "and the unsupervised learning where the data was unlabeled."
        ),
        kind="concept",
    )

    result = check_answer(
        "There are three types: supervised, unsupervised and "
        "reinforcement learning.",
        item,
        FakeEmbedder(),
        llm=broken_llm
    )

    assert "saw" not in result["missing"]


def test_british_spelling_matches_the_video():
    result = check_answer(
        "It uses labelled data to train the model.",
        dict(COINS_QUESTION, answer="Hence, supervised learning uses labeled data to train the model."),
        FakeEmbedder(),
        llm=broken_llm
    )

    assert "labeled" not in result["missing"]
    assert result["verdict"] == "correct"


# ---- without Ollama ---------------------------------------------------

def test_without_ollama_similarity_decides_and_says_so():
    result = check_answer(
        COINS_EXAMPLE_QUESTION["answer"],
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=broken_llm
    )

    assert result["verdict"] == "correct"
    assert result["judged_by"] == "similarity"
    assert "language model is not available" in result["feedback"]


def test_an_unreadable_examiner_reply_falls_back_to_similarity():
    result = check_answer(
        COINS_EXAMPLE_QUESTION["answer"],
        COINS_EXAMPLE_QUESTION,
        FakeEmbedder(),
        llm=fixed_llm("I think this is a great answer!")
    )

    assert result["verdict"] == "correct"
    assert result["judged_by"] == "similarity"


def test_an_old_question_without_section_text_still_works():
    old = {
        key: value
        for key, value in COINS_EXAMPLE_QUESTION.items()
        if key not in ("section_text", "kind")
    }

    result = check_answer(old["answer"], old, FakeEmbedder(), llm=broken_llm)

    assert result["verdict"] == "correct"


# ============================================================
# REAL-VIDEO CASES: QUESTIONS
# ============================================================

CRICKET_POINT = {
    # The summary's sentence for this part was a fragment on the real video.
    "title": "Types Of Players",
    "summary": (
        "While the other cluster is of the players who scored less runs "
        "but took many wickets."
    ),
    "start": 233.0,
    "end": 271.0,
    "text": CRICKET_TEXT,
}

COINS_POINT = {
    "title": "Supervised Learning Explained",
    "summary": (
        "Suppose your friend gives you one million coins of three "
        "different currencies, say one rupee, one euro and one dirham."
    ),
    "start": 173.0,
    "end": 232.0,
    "text": COINS_TEXT,
}


def test_a_fragment_is_never_an_expected_answer():
    item = generate_viva_questions([CRICKET_POINT], llm=broken_llm)[0]

    assert not answer_sentence(item).startswith("While")
    assert viva._comparable(answer_sentence(item)) in viva._comparable(CRICKET_TEXT)
    assert item["source"] in FALLBACK_SOURCES


def test_without_ollama_q2_expects_the_section_sentence_about_its_topic():
    # The real Q2: not the coins example's first sentence any more.
    item = generate_viva_questions([COINS_POINT], llm=broken_llm)[0]

    # (round 4: asked as fill-in-the-blank on that sentence)
    assert answer_sentence(item) == (
        "Hence, supervised learning uses labeled data to train the model."
    )
    assert viva.BLANK in item["question"]
    assert item["answer"] in answer_sentence(item)


def test_without_ollama_a_definition_now_matches_q2():
    item = generate_viva_questions([COINS_POINT], llm=broken_llm)[0]

    result = check_answer(DEFINITION, item, FakeEmbedder(), llm=broken_llm)

    assert result["verdict"] in ("correct", "partial")


def test_a_topic_sentence_must_say_something_new():
    # "So today we will learn about the basics of machine learning."
    # adds only "today" to the question: the summary sentence is used.
    item = generate_viva_questions([PAUL_POINT], llm=broken_llm)[0]

    assert answer_sentence(item) == "He loves listening to new songs."


def test_a_fallback_answer_points_at_the_segment_that_says_it():
    segments = [
        {"start": 233.0, "end": 250.0, "text": " Suppose you have a cricket dataset of various players with their respective scores and wickets taken."},
        {"start": 250.0, "end": 271.0, "text": " Hence, the learning with unlabeled data is unsupervised learning."},
    ]

    item = generate_viva_questions(
        [dict(CRICKET_POINT, title="Introduction")],
        llm=broken_llm,
        segments=segments,
        embedding_model=FakeEmbedder()
    )[0]

    # the answer is a real sentence, and Jump goes to where it is said
    said_at = [s for s in segments if item["answer"] in s["text"]]

    if said_at:
        assert item["start"] == said_at[0]["start"]

    assert 233.0 <= item["start"] < item["end"] <= 271.0


def answer_first_llm(question_reply, answer_first_reply):
    """Step 1 gets one reply, the answer-first prompt another."""
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if "Passage:" in prompt:
            return answer_first_reply
        return question_reply

    llm.calls = calls
    return llm


def test_a_vague_question_is_replaced_by_an_answer_first_question():
    # (Changed expectation, round 3: the answer-first passage is now the
    # focused 1-2 sentences about the clusters, not the first ~320
    # characters of the section. Uses the REAL Whisper transcript, whose
    # cluster sentence is clearly the most informative one.)
    llm = answer_first_llm(
        # step 1: a yes/no question - rejected
        "QUESTION: Does the machine find clusters of players?\n"
        "ANSWER: Yes, it finds two clusters.",
        # step 2: a specific question for the focused passage
        "QUESTION: How are the players divided into the two clusters by "
        "their runs and wickets?"
    )

    item = generate_viva_questions([REAL_CRICKET_POINT], llm=llm)[0]

    assert item["source"] == "llm_answer_first"
    assert item["question"] == (
        "How are the players divided into the two clusters by their runs "
        "and wickets?"
    )
    assert "types of players" not in item["question"].lower()

    # the answer is real transcript: the cluster sentences
    assert item["answer"].startswith("The one cluster are the players")
    assert viva._comparable(item["answer"]) in viva._comparable(
        REAL_CRICKET_POINT["text"]
    )


def test_an_answer_first_question_must_fit_the_focused_passage():
    # Written for the OLD wide passage ("Suppose you have a cricket
    # dataset ..."): the focused cluster passage does not answer it.
    llm = answer_first_llm(
        "QUESTION: Does the machine find clusters of players?\n"
        "ANSWER: Yes, it finds two clusters.",
        "QUESTION: How does the machine group the cricket players by "
        "their runs and wickets?"
    )

    item = generate_viva_questions([REAL_CRICKET_POINT], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES


def test_an_answer_first_question_is_checked_too():
    llm = answer_first_llm(
        "QUESTION: Does the machine find clusters?\nANSWER: Yes.",
        # invents facts: neural networks, accuracy, training
        "QUESTION: Which neural network gives the best accuracy when "
        "training on cricket?"
    )

    item = generate_viva_questions([CRICKET_POINT], llm=llm)[0]

    assert item["source"] in FALLBACK_SOURCES
    assert "neural" not in item["question"].lower()


def test_question_form_words_are_not_invented_facts():
    llm = fixed_llm(
        "QUESTION: How does the video distinguish between the two "
        "clusters of cricket players?\n"
        "ANSWER: One cluster scored higher runs and took less wickets, "
        "while the other scored less runs but took many wickets."
    )

    item = generate_viva_questions([CRICKET_POINT], llm=llm)[0]

    assert item["source"] == "llm"


def test_questions_alternate_between_concept_and_example():
    llm = fixed_llm("")

    generate_viva_questions([COINS_POINT, CRICKET_POINT], llm=llm)

    step_one = [call for call in llm.calls if "Transcript excerpt:" in call]

    assert "Ask about the IDEA" in step_one[0]
    assert "Ask about the EXAMPLE" in step_one[1]


def test_every_question_records_its_kind_and_section():
    questions = generate_viva_questions(
        [COINS_POINT, CRICKET_POINT],
        llm=broken_llm
    )

    for item in questions:
        assert item["kind"] in ("concept", "example")
        assert item["section_text"] in (COINS_TEXT, CRICKET_TEXT)
        assert item["section_start"] <= item["start"]
        assert item["end"] <= item["section_end"]


def test_every_expected_answer_answers_its_question():
    llm = answer_first_llm(
        "QUESTION: What is Paul's primary reason for liking or disliking "
        "new songs?\nANSWER: He likes them or dislikes them.",
        ""
    )

    for point in (COINS_POINT, CRICKET_POINT, PAUL_POINT):
        item = generate_viva_questions([point], llm=llm)[0]

        # (a blank's expected answer is a phrase of its checked sentence)
        assert viva._answer_problem(
            answer_sentence(item), item["question"], point["text"]
        ) is None

        assert item["answer"] in answer_sentence(item)


def test_rejections_are_logged_with_their_reason(caplog):
    llm = fixed_llm(
        "QUESTION: What is Paul primarily using his machine learning "
        "knowledge for?\nANSWER: Paul loves listening to new songs."
    )

    with caplog.at_level("WARNING", logger="src.rag.viva"):
        generate_viva_questions([PAUL_POINT], llm=llm)

    assert "LLM question not used" in caplog.text
    assert "assumes words the clip does not contain" in caplog.text
    assert "knowledge" in caplog.text


def test_without_ollama_the_reason_is_logged(caplog):
    with caplog.at_level("WARNING", logger="src.rag.viva"):
        generate_viva_questions([COINS_POINT], llm=broken_llm)

    assert "language model not available" in caplog.text


# ============================================================
# ROUND 2: EXACT WHISPER TRANSCRIPT OF THE TEST VIDEO
# ============================================================
#
# Short excerpts of the real Whisper output for machine_learning.mp4
# (data/processed/machine_learning_segments.json), with their real
# timestamps, and the questions/answers the real-video check produced.

def _segments(rows):
    return [{"start": start, "end": end, "text": " " + text} for start, end, text in rows]


def _point(title, rows, summary="(LLM-written summary description)"):
    segments = _segments(rows)
    return {
        "title": title,
        "summary": summary,
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(s["text"].strip() for s in segments),
    }, segments


REAL_PAUL_ROWS = [
    (17.6, 20.2, "It's also about understanding and reasoning."),
    (20.2, 23.6, "So today we will learn about the basics of machine learning."),
    (23.6, 27.6, "So that's Paul. He loves listening to new songs."),
    (28.6, 31.0, "He either likes them or dislikes them."),
]

REAL_TYPES_ROWS = [
    (159.2, 160.2, "More the data,"),
    (160.2, 161.4, "better the model,"),
    (161.4, 163.2, "higher will be the accuracy."),
    (163.2, 166.2, "There are many ways in which the machine learns."),
    (166.2, 168.2, "It could be either supervised learning"),
    (168.2, 170.0, "or reinforcement learning."),
    (170.0, 173.0, "Let's first quickly understand supervised learning."),
    (173.0, 176.0, "Suppose your friend gives you one million coins"),
    (176.0, 178.0, "of three different currencies."),
    (178.0, 180.0, "Say one rupee, one euro,"),
    (180.0, 181.0, "and one dirham."),
]

REAL_CRICKET_ROWS = [
    (239.6, 243.0, "So it plots this data with the respective wickets and the x-axis,"),
    (243.0, 244.6, "while runs on the y-axis."),
    (244.6, 245.6, "While looking at the data,"),
    (245.6, 248.4, "you will clearly see that there are two clusters."),
    (248.4, 251.6, "The one cluster are the players who scored high runs"),
    (251.6, 253.0, "and took less wickets."),
    (253.0, 256.8, "While the other cluster is of the players who scored less runs"),
    (256.8, 258.0, "but took many wickets."),
    (258.0, 262.2, "So here we interpret these two clusters as batsmen and bowlers."),
    (262.2, 264.2, "The important point to note here is that"),
    (264.2, 267.0, "there were no labels of batsmen and bowlers."),
    (267.0, 271.0, "Hence, the learning with unlabeled data is unsupervised learning."),
]

REAL_RL_ROWS = [
    (271.0, 273.8, "So we saw supervised learning where the data was labeled"),
    (273.8, 277.0, "and the unsupervised learning where the data was unlabeled."),
    (277.0, 279.0, "And then there is reinforcement learning,"),
    (279.0, 280.8, "which is a reward-based learning."),
    (280.8, 283.2, "Or we can say that it works on the principle of feedback."),
    (283.2, 286.4, "Here, let's say you provide the system with an image of a dog"),
    (286.4, 288.2, "and ask it to identify it."),
]

REAL_PAUL_POINT, REAL_PAUL_SEGMENTS = _point(
    "Basics of Machine Learning", REAL_PAUL_ROWS,
    # what the summary's LLM really wrote for this part: NOT true
    "Paul explains the importance of understanding and reasoning in "
    "machine learning, introducing the basics of the subject."
)
REAL_TYPES_POINT, REAL_TYPES_SEGMENTS = _point("Types of Machine Learning", REAL_TYPES_ROWS)
REAL_CRICKET_POINT, REAL_CRICKET_SEGMENTS = _point("Types of Players", REAL_CRICKET_ROWS)
REAL_RL_POINT, REAL_RL_SEGMENTS = _point("Principle System", REAL_RL_ROWS)

# Q4 of the real-video check: a real llama3.2 question.
REAL_RL_QUESTION = {
    "question": "What principle does reinforcement learning work on?",
    "answer": "It works on the principle of feedback.",
    "kind": "example",
    "start": 280.8,
    "end": 283.2,
    "text": "Or we can say that it works on the principle of feedback.",
    "section_start": 271.0,
    "section_end": 288.2,
    "section_text": REAL_RL_POINT["text"],
    "title": "Principle System",
    "source": "llm",
}

# Q3 of the real-video check (simple question form).
REAL_CRICKET_QUESTION = {
    "question": "What does the video explain about types of players?",
    "answer": (
        "So here we interpret these two clusters as batsmen and bowlers. "
        "The important point to note here is that there were no labels of "
        "batsmen and bowlers. Hence, the learning with unlabeled data is "
        "unsupervised learning."
    ),
    "kind": "concept",
    "start": 253.0,
    "end": 271.0,
    "text": REAL_CRICKET_POINT["text"],
    "section_start": 239.6,
    "section_end": 271.0,
    "section_text": REAL_CRICKET_POINT["text"],
    "title": "Types of Players",
    "source": "transcript",
}

REWARDS_AND_PENALTIES = (
    "Reinforcement learning learns by trial and error, getting a reward "
    "for a good action and a penalty for a bad one."
)

CRICKET_CLUSTERS = (
    "The machine groups the players into two clusters without any labels: "
    "players with many runs and few wickets, and players with few runs but "
    "many wickets."
)


def with_similarity(monkeypatch, value):
    """Make MiniLM's (stand-in) similarity a fixed value for one test."""
    monkeypatch.setattr(
        viva,
        "_similarities",
        lambda answer, references, model: [value] * len(references)
    )


# ---- valid alternative explanations ------------------------------

def test_real_rewards_and_penalties_explain_feedback():
    # On the real video llama3.2 failed this answer; with the clearer
    # prompt a correct verdict is accepted as correct.
    result = check_answer(
        REWARDS_AND_PENALTIES,
        REAL_RL_QUESTION,
        FakeEmbedder(),
        llm=examiner("CORRECT", "EXAMPLE")
    )

    assert result["verdict"] == "correct"


def test_real_rewards_and_penalties_survive_a_too_strict_examiner(monkeypatch):
    # The examiner says INCORRECT, but the meaning is close to the video
    # and nothing contradicts it: partly correct, not zero.
    with_similarity(monkeypatch, 0.70)

    result = check_answer(
        REWARDS_AND_PENALTIES,
        REAL_RL_QUESTION,
        FakeEmbedder(),
        llm=fixed_llm("VERDICT: INCORRECT\nTYPE: WRONG\nCONTRADICTS: NO\nMISSING: principle of feedback")
    )

    assert result["verdict"] == "partial"


def test_real_cricket_clustering_answer_is_correct():
    result = check_answer(
        CRICKET_CLUSTERS,
        REAL_CRICKET_QUESTION,
        FakeEmbedder(),
        llm=examiner("CORRECT", "EXAMPLE")
    )

    assert result["verdict"] == "correct"


def test_real_cricket_answer_survives_a_too_strict_examiner(monkeypatch):
    # The real-video check: similarity 0.66, examiner said INCORRECT.
    with_similarity(monkeypatch, 0.66)

    result = check_answer(
        CRICKET_CLUSTERS,
        REAL_CRICKET_QUESTION,
        FakeEmbedder(),
        llm=fixed_llm("VERDICT: INCORRECT\nTYPE: WRONG\nCONTRADICTS: NO\nMISSING: NONE")
    )

    assert result["verdict"] == "partial"


def test_naming_both_sides_is_not_a_contradiction():
    # "many runs and few wickets" uses both opposite words, correctly.
    assert viva._contradiction(CRICKET_CLUSTERS, REAL_CRICKET_QUESTION) is None
    assert viva._contradiction(REWARDS_AND_PENALTIES, REAL_RL_QUESTION) is None


def test_an_incomplete_reading_by_the_examiner_is_partial():
    result = check_answer(
        "It learns from rewards.",
        REAL_RL_QUESTION,
        FakeEmbedder(),
        llm=fixed_llm("VERDICT: INCORRECT\nTYPE: INCOMPLETE\nCONTRADICTS: NO\nMISSING: NONE")
    )

    assert result["verdict"] == "partial"


# ---- opposite or contradictory answers ------------------------------

OPPOSITE_CASES = [
    # (answer, question) - similar words, opposite meaning
    ("Supervised learning uses unlabeled data to train the model.",
     dict(COINS_QUESTION, answer="Hence, supervised learning uses labeled data to train the model.")),
    ("Reinforcement learning does not work on the principle of feedback.",
     REAL_RL_QUESTION),
    ("The clusters had labels of batsmen and bowlers, so it is supervised learning.",
     REAL_CRICKET_QUESTION),
]


@pytest.mark.parametrize("answer, item", OPPOSITE_CASES)
def test_an_opposite_answer_is_not_rescued_by_similarity(monkeypatch, answer, item):
    with_similarity(monkeypatch, 0.80)

    result = check_answer(
        answer,
        item,
        FakeEmbedder(),
        llm=fixed_llm("VERDICT: INCORRECT\nTYPE: WRONG\nCONTRADICTS: NO\nMISSING: NONE")
    )

    assert result["verdict"] == "incorrect"


@pytest.mark.parametrize("answer, item", OPPOSITE_CASES)
def test_the_plain_rules_find_the_opposite_meaning(answer, item):
    assert viva._contradiction(answer, item) is not None


def test_the_examiner_saying_contradicts_blocks_the_safety_net(monkeypatch):
    with_similarity(monkeypatch, 0.80)

    result = check_answer(
        REWARDS_AND_PENALTIES,
        REAL_RL_QUESTION,
        FakeEmbedder(),
        llm=fixed_llm("VERDICT: INCORRECT\nTYPE: WRONG\nCONTRADICTS: YES\nMISSING: NONE")
    )

    assert result["verdict"] == "incorrect"


def test_without_ollama_an_opposite_answer_is_not_correct(monkeypatch):
    with_similarity(monkeypatch, 0.80)

    result = check_answer(
        "Supervised learning uses unlabeled data to train the model.",
        dict(COINS_QUESTION, answer="Hence, supervised learning uses labeled data to train the model."),
        FakeEmbedder(),
        llm=broken_llm
    )

    assert result["verdict"] != "correct"


def test_the_examiner_prompt_defines_the_three_verdicts():
    llm = examiner("CORRECT", "CONCEPT")

    check_answer(REWARDS_AND_PENALTIES, REAL_RL_QUESTION, FakeEmbedder(), llm=llm)

    prompt = llm.calls[0]

    assert "CORRECT - the answer has the right meaning, even in completely different words" in prompt
    assert "PARTIAL - the answer is relevant and factually correct, but incomplete" in prompt
    assert "INCORRECT - the answer is wrong, contradicts the video" in prompt
    assert "CONTRADICTS:" in prompt


# ---- fallback answers: real, informative transcript sentences ---------

def test_real_transition_sentence_is_never_a_fallback_answer():
    item = generate_viva_questions([REAL_TYPES_POINT], llm=broken_llm)[0]

    assert answer_sentence(item) != "Let's first quickly understand supervised learning."
    assert answer_sentence(item) == (
        "It could be either supervised learning or reinforcement learning."
    )
    assert item["answer"] == "reinforcement learning"


@pytest.mark.parametrize("sentence", [
    "Let's first quickly understand supervised learning.",
    "So today we will learn about the basics of machine learning.",
    "Now let's look at reinforcement learning.",
    "Comment below some interesting everyday examples around you, "
    "where machines are learning and doing amazing jobs.",
])
def test_transition_sentences_are_recognised(sentence):
    assert viva._is_transition(sentence)
    assert not viva._is_informative(sentence)


@pytest.mark.parametrize("sentence", [
    "So that's Paul.",
    "Simple, right?",
    "It's also about understanding and reasoning.",
])
def test_low_information_sentences_are_not_answers(sentence):
    assert not viva._is_informative(sentence)


def test_the_llm_summary_description_is_never_the_expected_answer():
    item = generate_viva_questions([REAL_PAUL_POINT], llm=broken_llm)[0]

    assert "Paul explains" not in answer_sentence(item)
    assert "Paul explains" not in item["question"]
    assert answer_sentence(item) == "He loves listening to new songs."


def test_every_fallback_answer_is_real_transcript():
    for point in (REAL_PAUL_POINT, REAL_TYPES_POINT, REAL_CRICKET_POINT, REAL_RL_POINT):
        item = generate_viva_questions([point], llm=broken_llm)[0]

        # word for word in the Whisper text of that point
        assert viva._comparable(item["answer"]) in viva._comparable(point["text"])


def test_an_idea_beats_an_example_for_the_plain_question():
    item = generate_viva_questions([REAL_RL_POINT], llm=broken_llm)[0]

    assert answer_sentence(item) == (
        "Or we can say that it works on the principle of feedback."
    )
    assert item["answer"] == "feedback"


def test_a_while_continuation_is_joined_but_a_while_sentence_is_not():
    sentences = viva._split_sentences(REAL_CRICKET_POINT["text"])

    assert (
        "The one cluster are the players who scored high runs and took less "
        "wickets. While the other cluster is of the players who scored less "
        "runs but took many wickets."
    ) in sentences
    assert (
        "While looking at the data, you will clearly see that there are two "
        "clusters."
    ) in sentences


def test_a_spoken_conjunction_does_not_make_a_fragment():
    assert not viva._is_fragment("Or we can say that it works on the principle of feedback.")
    assert viva._is_fragment("Or reinforcement learning.")
    assert viva._is_fragment("While the other cluster is of the players who scored less runs.")
    assert not viva._is_fragment("While looking at the data, you will clearly see two clusters.")


# ---- questions with normal question vocabulary ------------------------

@pytest.mark.parametrize("point, reply", [
    (REAL_RL_POINT,
     "QUESTION: How does the video illustrate the concept of reinforcement learning?\n"
     "ANSWER: Reinforcement learning is a reward-based learning that works on the principle of feedback."),
    (REAL_CRICKET_POINT,
     "QUESTION: What is the interpretation of the two clusters in the scenario?\n"
     "ANSWER: The two clusters are interpreted as batsmen and bowlers."),
    (REAL_CRICKET_POINT,
     "QUESTION: How does the process of clustering relate to the players?\n"
     "ANSWER: The machine finds two clusters of players: high runs and less wickets, or less runs and many wickets."),
    (REAL_TYPES_POINT,
     "QUESTION: What does the video reveal about the ways in which the machine learns?\n"
     "ANSWER: It could be either supervised learning or reinforcement learning."),
])
def test_questions_with_normal_question_vocabulary_are_kept(point, reply):
    item = generate_viva_questions([point], llm=fixed_llm(reply))[0]

    assert item["source"] == "llm"


def test_question_vocabulary_is_matched_by_word_start():
    for word in ("illustrate", "illustrates", "illustration", "interpretation",
                 "interpret", "concept", "concepts", "scenario", "process",
                 "processes", "relate", "relationship", "reveal", "reveals"):
        assert viva._is_question_vocabulary(viva._stem(word)), word


def test_invented_claims_are_still_rejected_with_question_vocabulary():
    # the question words are fine; "neural", "accuracy", "gradient" are not
    llm = fixed_llm(
        "QUESTION: How does the video illustrate the concept of neural "
        "network accuracy and gradient descent?\n"
        "ANSWER: Reinforcement learning is a reward-based learning."
    )

    item = generate_viva_questions([REAL_RL_POINT], llm=llm)[0]

    assert item["source"] != "llm"


# ---- unrelated and vague answers are still rejected -----------------

@pytest.mark.parametrize("item", [REAL_RL_QUESTION, REAL_CRICKET_QUESTION])
@pytest.mark.parametrize("answer", [
    "I like playing cricket on weekends.",
    "The capital of France is Paris.",
])
def test_real_unrelated_answers_are_rejected(item, answer):
    llm = examiner("CORRECT", "CONCEPT")

    result = check_answer(answer, item, FakeEmbedder(), llm=llm)

    assert result["verdict"] == "incorrect"
    assert llm.calls == []


def test_the_vague_cricket_answer_stays_rejected(monkeypatch):
    # Real-video check: "It puts the players into groups." similarity 0.39
    with_similarity(monkeypatch, 0.39)
    llm = examiner("CORRECT", "CONCEPT")

    result = check_answer(
        "It puts the players into groups.",
        REAL_CRICKET_QUESTION,
        FakeEmbedder(),
        llm=llm
    )

    assert result["verdict"] == "incorrect"
    assert result["kind"] == "off_topic"
    assert llm.calls == []


# ---- timestamps --------------------------------------------------------

def test_real_cricket_answer_starts_where_it_is_said():
    item = generate_viva_questions(
        [REAL_CRICKET_POINT],
        llm=broken_llm,
        segments=REAL_CRICKET_SEGMENTS,
        embedding_model=FakeEmbedder()
    )[0]

    assert answer_sentence(item).startswith("The one cluster are the players")
    assert item["start"] == 248.4      # 4:08, not 4:13
    assert item["end"] == 258.0


def test_real_feedback_answer_points_at_4_40():
    item = generate_viva_questions(
        [REAL_RL_POINT],
        llm=broken_llm,
        segments=REAL_RL_SEGMENTS,
        embedding_model=FakeEmbedder()
    )[0]

    assert item["start"] == 280.8
    assert item["end"] == 283.2


def test_real_paul_answer_points_at_its_segment():
    item = generate_viva_questions(
        [REAL_PAUL_POINT],
        llm=broken_llm,
        segments=REAL_PAUL_SEGMENTS,
        embedding_model=FakeEmbedder()
    )[0]

    assert item["start"] == 23.6
    assert item["end"] == 27.6


def test_every_real_timestamp_is_a_whisper_segment_inside_its_section():
    for point, segments in (
        (REAL_PAUL_POINT, REAL_PAUL_SEGMENTS),
        (REAL_TYPES_POINT, REAL_TYPES_SEGMENTS),
        (REAL_CRICKET_POINT, REAL_CRICKET_SEGMENTS),
        (REAL_RL_POINT, REAL_RL_SEGMENTS),
    ):
        item = generate_viva_questions(
            [point], llm=broken_llm,
            segments=segments, embedding_model=FakeEmbedder()
        )[0]

        assert any(s["start"] == item["start"] for s in segments)
        assert any(s["end"] == item["end"] for s in segments)
        assert item["section_start"] <= item["start"] < item["end"] <= item["section_end"]


# ============================================================
# ROUND 3: SHORT ANSWERS AND FOCUSED ANSWER-FIRST PASSAGES
# ============================================================

REAL_PAUL_QUESTION = {
    "question": "What does Paul love listening to?",
    "answer": "He loves listening to new songs.",
    "kind": "concept",
    "start": 23.6,
    "end": 27.6,
    "text": "So that's Paul. He loves listening to new songs.",
    "section_start": 17.6,
    "section_end": 31.0,
    "section_text": REAL_PAUL_POINT["text"],
    "title": "Basics of Machine Learning",
    "source": "llm_answer_first",
}

REAL_TYPES_QUESTION = {
    "question": "What happens to the model when there is more data?",
    "answer": "More the data, better the model, higher will be the accuracy.",
    "kind": "concept",
    "start": 159.2,
    "end": 163.2,
    "text": "More the data, better the model, higher will be the accuracy.",
    "section_start": 159.2,
    "section_end": 181.0,
    "section_text": REAL_TYPES_POINT["text"],
    "title": "Types of Machine Learning",
    "source": "llm_answer_first",
}

# The three short answers llama3.2 marked fully correct on the real run.
REAL_VAGUE_ANSWERS = [
    ("It is about the songs Paul likes.", REAL_PAUL_QUESTION, 0.35),
    ("The model learns from data.", REAL_TYPES_QUESTION, 0.75),
    ("It learns from rewards.", REAL_RL_QUESTION, 0.63),
]


# ---- the cap only turns CORRECT into PARTIAL ---------------------------

@pytest.mark.parametrize("answer, item, similarity", REAL_VAGUE_ANSWERS)
def test_real_vague_answers_are_capped_at_partial(monkeypatch, answer, item, similarity):
    with_similarity(monkeypatch, similarity)

    # (EXAMPLE, so no other rule lowers the verdict first: the cap is
    # what is tested here)
    result = check_answer(
        answer, item, FakeEmbedder(),
        llm=examiner("CORRECT", "EXAMPLE")
    )

    assert result["verdict"] == "partial"
    assert result["kind"] == "too_brief"
    assert "too brief" in result["feedback"]


def test_the_cap_never_raises_an_incorrect_verdict(monkeypatch):
    with_similarity(monkeypatch, 0.50)

    result = check_answer(
        "It learns from rewards.", REAL_RL_QUESTION, FakeEmbedder(),
        llm=fixed_llm("VERDICT: INCORRECT\nTYPE: WRONG\nCONTRADICTS: NO\nMISSING: NONE")
    )

    assert result["verdict"] == "incorrect"


def test_the_cap_leaves_a_partial_verdict_alone(monkeypatch):
    with_similarity(monkeypatch, 0.50)

    result = check_answer(
        "It learns from rewards.", REAL_RL_QUESTION, FakeEmbedder(),
        llm=examiner("PARTIAL", "INCOMPLETE")
    )

    assert result["verdict"] == "partial"
    assert result["kind"] == "partial"


@pytest.mark.parametrize("answer", ["Paris.", "I like cricket.", "Cats."])
def test_short_unrelated_answers_stay_incorrect(answer):
    llm = examiner("CORRECT", "CONCEPT")

    result = check_answer(answer, REAL_RL_QUESTION, FakeEmbedder(), llm=llm)

    assert result["verdict"] == "incorrect"
    assert result["kind"] == "off_topic"
    assert llm.calls == []


def test_a_short_contradicting_answer_is_never_correct(monkeypatch):
    # Even if the LLM wrongly says CORRECT.
    with_similarity(monkeypatch, 0.60)

    result = check_answer(
        "Not feedback, never.", REAL_RL_QUESTION, FakeEmbedder(),
        llm=examiner("CORRECT", "CONCEPT")
    )

    assert result["verdict"] != "correct"


# ---- complete answers are not affected ---------------------------------

@pytest.mark.parametrize("answer, item, answer_type", [
    (REWARDS_AND_PENALTIES, REAL_RL_QUESTION, "EXAMPLE"),
    (CRICKET_CLUSTERS, REAL_CRICKET_QUESTION, "EXAMPLE"),
    (DEFINITION, COINS_QUESTION, "CONCEPT"),
    ("Paul loves listening to new songs and either likes or dislikes them.",
     REAL_PAUL_QUESTION, "EXAMPLE"),
])
def test_valid_complete_answers_stay_correct(answer, item, answer_type):
    result = check_answer(
        answer, item, FakeEmbedder(),
        llm=examiner("CORRECT", answer_type)
    )

    assert result["verdict"] == "correct"


# ---- the exception: the video's own short answer -----------------------

def test_the_videos_own_short_answer_stays_correct(monkeypatch):
    # "Feedback." names the expected answer's only key word and is
    # similar in meaning to "It works on the principle of feedback."
    with_similarity(monkeypatch, 0.55)

    result = check_answer(
        "Feedback.", REAL_RL_QUESTION, FakeEmbedder(),
        llm=examiner("CORRECT", "CONCEPT")
    )

    assert result["verdict"] == "correct"


@pytest.mark.parametrize("similarity, coverage, contradiction, own", [
    (0.95, 0.0, None, True),         # meaning practically identical
    (0.55, 1.0, None, True),         # all key words AND similar meaning
    (0.30, 1.0, None, False),        # key words but not the same meaning
    (0.60, 0.5, None, False),        # similar but incomplete
    (0.95, 1.0, "negation", False),  # a contradiction always fails
])
def test_the_exception_needs_meaning_and_no_contradiction(similarity, coverage, contradiction, own):
    assert viva._is_videos_own_answer(similarity, coverage, contradiction) is own


# ---- counting meaningful words -------------------------------------------

@pytest.mark.parametrize("answer, too_brief", [
    ("data data data data model model", True),           # 2 different words
    ("So basically it is the data that we saw.", True),  # filler and stop words
    ("It learns from rewards.", True),
    ("The model learns from data.", True),
    ("Supervised learning trains a model on labeled data.", False),
    (REWARDS_AND_PENALTIES, False),
])
def test_only_different_meaningful_words_are_counted(answer, too_brief):
    assert viva._is_too_brief(answer) is too_brief


# ---- without Ollama: unchanged -------------------------------------------

def test_without_ollama_grading_is_unchanged(monkeypatch):
    # Safeguard: the no-Ollama path keeps its existing behaviour; the
    # brevity cap only applies to the LLM examiner's verdicts.
    with_similarity(monkeypatch, 0.75)

    result = check_answer(
        "The model learns from data.", REAL_TYPES_QUESTION, FakeEmbedder(),
        llm=broken_llm
    )

    assert result["verdict"] == "correct"
    assert result["judged_by"] == "similarity"


# ---- focused answer-first passages ---------------------------------------

def test_real_paul_passage_is_its_informative_sentences():
    # (the filler before it - "It's also about understanding and
    # reasoning.", "So today we will learn ...", "So that's Paul." - is
    # left out)
    assert viva._focused_passage(REAL_PAUL_POINT["text"]) == (
        "He loves listening to new songs. "
        "He either likes them or dislikes them."
    )


def test_real_cricket_passage_is_about_the_clusters():
    passage = viva._focused_passage(REAL_CRICKET_POINT["text"])

    assert passage.startswith("The one cluster are the players")
    assert len(viva._split_sentences(passage)) <= viva.MAX_FOCUSED_SENTENCES


def test_real_types_passage_does_not_run_into_the_coins_example():
    passage = viva._focused_passage(REAL_TYPES_POINT["text"])

    assert "Suppose" not in passage
    assert "coins" not in passage
    assert "Let's first quickly" not in passage


def test_focused_passages_are_consecutive_real_transcript():
    for point in (REAL_PAUL_POINT, REAL_TYPES_POINT, REAL_CRICKET_POINT, REAL_RL_POINT):
        passage = viva._focused_passage(point["text"])

        assert passage
        assert viva._comparable(passage) in viva._comparable(point["text"])
        assert len(viva._split_sentences(passage)) <= viva.MAX_FOCUSED_SENTENCES

        for sentence in viva._split_sentences(passage):
            assert viva._is_informative(sentence)


def test_the_answer_first_prompt_gets_only_the_focused_passage():
    llm = answer_first_llm(
        "QUESTION: Does the machine find clusters?\nANSWER: Yes.",
        ""
    )

    generate_viva_questions([REAL_TYPES_POINT], llm=llm)

    prompt = next(call for call in llm.calls if "Passage:" in call)

    assert "coins" not in prompt
    assert "Let's first quickly" not in prompt


# ---- exact timestamps, or the whole section --------------------------------

def cricket_answer_first_llm():
    return answer_first_llm(
        "QUESTION: Does the machine find clusters of players?\n"
        "ANSWER: Yes, it finds two clusters.",
        "QUESTION: How are the players divided into the two clusters by "
        "their runs and wickets?"
    )


def test_an_answer_first_passage_gets_its_exact_timestamp():
    item = generate_viva_questions(
        [REAL_CRICKET_POINT],
        llm=cricket_answer_first_llm(),
        segments=REAL_CRICKET_SEGMENTS,
        embedding_model=FakeEmbedder()
    )[0]

    assert item["source"] == "llm_answer_first"
    assert item["start"] == 248.4      # 4:08, "The one cluster ..."
    assert item["end"] == 262.2        # end of "... batsmen and bowlers."


def test_an_unmatched_passage_keeps_the_whole_section():
    # Whisper's segments say something slightly different from the point
    # text: the passage cannot be found word for word, so no position is
    # guessed - the question keeps its whole section.
    different = [
        dict(segment, text=segment["text"].replace("cluster", "group"))
        for segment in REAL_CRICKET_SEGMENTS
    ]

    item = generate_viva_questions(
        [REAL_CRICKET_POINT],
        llm=cricket_answer_first_llm(),
        segments=different,
        embedding_model=FakeEmbedder()
    )[0]

    assert item["source"] == "llm_answer_first"
    assert item["start"] == REAL_CRICKET_POINT["start"]
    assert item["end"] == REAL_CRICKET_POINT["end"]


def test_without_ollama_questions_use_the_same_sentences():
    # (Round 4: without Ollama the question is fill-in-the-blank, on the
    # SAME transcript sentences as before.)
    expected = {
        "Basics of Machine Learning": "He loves listening to new songs.",
        "Types of Machine Learning": "It could be either supervised learning or reinforcement learning.",
        "Principle System": "Or we can say that it works on the principle of feedback.",
    }

    for point in (REAL_PAUL_POINT, REAL_TYPES_POINT, REAL_RL_POINT):
        item = generate_viva_questions([point], llm=broken_llm)[0]

        assert item["source"] == "transcript_blank"
        assert answer_sentence(item) == expected[point["title"]]
        assert item["answer"] in expected[point["title"]]


# ============================================================
# ROUND 4: QUESTION/ANSWER ALIGNMENT AND NATURAL FEEDBACK
# ============================================================

def routed_llm(step_one="", answer_first="", one_sentence="", judge=""):
    """A stand-in LLM that answers each prompt type differently."""
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if "Answer sentence from the video:" in prompt:
            return one_sentence
        if "Passage:" in prompt:
            return answer_first
        if "Student's answer:" in prompt:
            return judge
        return step_one

    llm.calls = calls
    return llm


REJECTED_STEP_ONE = (
    # a yes/no question: always rejected
    "QUESTION: Does Paul like listening to new songs?\n"
    "ANSWER: Yes, he loves listening to new songs."
)


# ---- 1. specific questions for specific answers ------------------------

def test_the_one_sentence_step_asks_what_paul_loves():
    # The case from the real video: a broad question for "He loves
    # listening to new songs." is replaced by a specific one.
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        answer_first="QUESTION: Does Paul like songs?",
        one_sentence="QUESTION: What does Paul love listening to?",
    )

    item = generate_viva_questions(
        [REAL_PAUL_POINT], llm=llm,
        segments=REAL_PAUL_SEGMENTS, embedding_model=FakeEmbedder()
    )[0]

    assert item["source"] == "llm_answer_sentence"
    assert item["question"] == "What does Paul love listening to?"
    assert item["answer"] == "He loves listening to new songs."

    # exact timestamp of that sentence (0:23)
    assert item["start"] == 23.6
    assert item["end"] == 27.6


def test_the_one_sentence_step_gets_the_sentence_before_as_context():
    llm = routed_llm(step_one=REJECTED_STEP_ONE, answer_first="")

    generate_viva_questions([REAL_PAUL_POINT], llm=llm)

    prompt = next(c for c in llm.calls if "Answer sentence from the video:" in c)

    assert "So that's Paul." in prompt                     # context
    assert "He loves listening to new songs." in prompt    # the answer
    assert "today we will learn" not in prompt             # nothing more


def test_the_one_sentence_question_is_validated_too():
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        # invents a claim: Paul does not use machine learning in the video
        one_sentence="QUESTION: What is Paul primarily using his machine "
                     "learning knowledge for?",
    )

    item = generate_viva_questions([REAL_PAUL_POINT], llm=llm)[0]

    assert item["source"] == "transcript_blank"
    assert "primarily" not in item["question"]


def test_no_broad_question_is_paired_with_a_narrow_answer():
    # Without Ollama, none of the real sections gets the broad
    # "What does the video explain about ...?" question any more.
    for point in (REAL_PAUL_POINT, REAL_TYPES_POINT, REAL_CRICKET_POINT, REAL_RL_POINT):
        item = generate_viva_questions([point], llm=broken_llm)[0]

        assert not item["question"].startswith("What does the video explain about")


# ---- fill in the blank ------------------------------------------------

@pytest.mark.parametrize("point, question, answer", [
    (REAL_PAUL_POINT,
     'Complete what the video says: "He loves listening to ______."',
     "new songs"),
    (REAL_TYPES_POINT,
     'Complete what the video says: "It could be either supervised learning or ______."',
     "reinforcement learning"),
    (REAL_RL_POINT,
     'Complete what the video says: "Or we can say that it works on the principle of ______."',
     "feedback"),
])
def test_real_fill_in_the_blank_questions(point, question, answer):
    item = generate_viva_questions([point], llm=broken_llm)[0]

    assert item["source"] == "transcript_blank"
    assert item["question"] == question
    assert item["answer"] == answer


def test_a_blank_is_aligned_with_its_answer_by_construction():
    for point in (REAL_PAUL_POINT, REAL_TYPES_POINT, REAL_CRICKET_POINT, REAL_RL_POINT):
        item = generate_viva_questions([point], llm=broken_llm)[0]

        filled = item["question"].replace(viva.BLANK, item["answer"])

        # filling the blank gives back the real transcript sentence
        assert filled == f'Complete what the video says: "{item["answer_sentence"]}"'
        assert viva._comparable(item["answer_sentence"]) in viva._comparable(point["text"])


def test_a_blank_question_keeps_its_exact_timestamp():
    item = generate_viva_questions(
        [REAL_PAUL_POINT], llm=broken_llm,
        segments=REAL_PAUL_SEGMENTS, embedding_model=FakeEmbedder()
    )[0]

    assert item["answer"] == "new songs"
    assert item["start"] == 23.6     # the sentence's segment, 0:23
    assert item["end"] == 27.6


def test_a_blank_keeps_enough_of_the_sentence():
    # Nothing would be left around the blank: no blank question.
    assert viva._blank_question("Reinforcement learning matters.") is None


def test_a_blank_matches_british_spelling():
    question, answer = viva._blank_question(
        "Supervised learning trains a model on labelled examples."
    )

    assert answer == "labelled examples"
    assert question.endswith('on ______."')


@pytest.mark.parametrize("answer, verdict", [
    ("new songs", "correct"),
    ("He loves listening to new songs.", "correct"),
    ("I like playing cricket.", "incorrect"),
    ("The capital of France is Paris.", "incorrect"),
])
def test_grading_a_blank_answer(answer, verdict):
    item = generate_viva_questions([REAL_PAUL_POINT], llm=broken_llm)[0]

    result = check_answer(
        answer, item, FakeEmbedder(),
        llm=examiner("CORRECT", "EXAMPLE") if verdict == "correct" else broken_llm
    )

    assert result["verdict"] == verdict


def test_a_one_word_blank_answer_is_not_rejected_as_unrelated():
    item = generate_viva_questions([REAL_RL_POINT], llm=broken_llm)[0]

    assert item["answer"] == "feedback"

    result = check_answer(
        "feedback", item, FakeEmbedder(),
        llm=examiner("CORRECT", "EXAMPLE")
    )

    assert result["verdict"] == "correct"


def test_a_wrong_one_word_blank_answer_is_still_rejected():
    item = generate_viva_questions([REAL_RL_POINT], llm=broken_llm)[0]

    llm = examiner("CORRECT", "EXAMPLE")

    result = check_answer("cricket", item, FakeEmbedder(), llm=llm)

    assert result["verdict"] == "incorrect"
    assert llm.calls == []


# ---- 2. missing PHRASES, not words ---------------------------------------

REAL_TYPES_BLANKLESS = {
    # the real question/answer seen in the app
    "question": "What does the video explain about types of machine learning?",
    "answer": "It could be either supervised learning or reinforcement learning.",
    "kind": "concept",
    "start": 166.2,
    "end": 170.0,
    "text": "It could be either supervised learning or reinforcement learning.",
    "section_start": 159.2,
    "section_end": 181.0,
    "section_text": REAL_TYPES_POINT["text"],
    "title": "Types of Machine Learning",
    "source": "transcript",
}


def test_the_real_types_feedback_is_natural(monkeypatch):
    with_similarity(monkeypatch, 0.55)

    result = check_answer(
        "Supervised learning.", REAL_TYPES_BLANKLESS, FakeEmbedder(),
        llm=examiner("PARTIAL", "INCOMPLETE")
    )

    assert result["verdict"] == "partial"
    assert (
        "You mentioned supervised learning. "
        "Also mention reinforcement learning."
    ) in result["feedback"]
    assert "either" not in result["feedback"]
    assert "Key points to mention" not in result["feedback"]


def test_the_feedback_is_the_same_without_ollama(monkeypatch):
    with_similarity(monkeypatch, 0.55)

    result = check_answer(
        "Supervised learning.", REAL_TYPES_BLANKLESS, FakeEmbedder(),
        llm=broken_llm
    )

    assert result["verdict"] == "partial"
    assert "Also mention reinforcement learning." in result["feedback"]
    assert "either" not in result["feedback"]


@pytest.mark.parametrize("text, phrases", [
    ("It could be either supervised learning or reinforcement learning.",
     ["supervised learning", "reinforcement learning"]),
    ("He loves listening to new songs.", ["loves listening", "new songs"]),
    ("And then there is reinforcement learning, which is a reward-based learning.",
     ["reinforcement learning", "reward-based learning"]),
    ("Neither supervised nor unsupervised learning is used here.",
     ["supervised", "unsupervised learning", "used"]),
])
def test_key_phrases_skip_grammatical_words(text, phrases):
    assert viva._key_phrases(text) == phrases

    for phrase in viva._key_phrases(text):
        for word in ("either", "neither", "could", "be", "or", "nor"):
            assert word not in phrase.split()


def test_a_shared_word_does_not_count_as_mentioning_a_phrase():
    # "learning" is in both phrases: "Reinforcement learning." mentions
    # only reinforcement learning, not supervised learning.
    mentioned, missing = viva._phrase_feedback(
        "Reinforcement learning.",
        "It could be either supervised learning or reinforcement learning.",
        "What does the video explain about types of machine learning?"
    )

    assert mentioned == ["reinforcement learning"]
    assert missing == ["supervised learning"]


def test_phrases_the_question_already_contains_are_not_asked_for():
    mentioned, missing = viva._phrase_feedback(
        "It is reward based.",
        "Reinforcement learning works on the principle of feedback.",
        "What principle does reinforcement learning work on?"
    )

    assert "reinforcement learning" not in missing
    assert "feedback" in missing


def test_a_grounded_llm_missing_concept_is_still_preferred(monkeypatch):
    with_similarity(monkeypatch, 0.55)

    result = check_answer(
        "Supervised learning.", REAL_TYPES_BLANKLESS, FakeEmbedder(),
        llm=examiner("PARTIAL", "INCOMPLETE", "reinforcement learning")
    )

    assert "Missing: Reinforcement learning." in result["feedback"]
    assert "Also mention" not in result["feedback"]


def test_an_ungrounded_llm_missing_concept_falls_back_to_phrases(monkeypatch):
    with_similarity(monkeypatch, 0.55)

    result = check_answer(
        "Supervised learning.", REAL_TYPES_BLANKLESS, FakeEmbedder(),
        llm=examiner("PARTIAL", "INCOMPLETE", "gradient descent optimisation")
    )

    assert "gradient" not in result["feedback"]
    assert "Also mention reinforcement learning." in result["feedback"]


def test_unrelated_answers_get_no_missing_point_hints():
    result = check_answer(
        "I like playing cricket.", REAL_TYPES_BLANKLESS, FakeEmbedder(),
        llm=examiner("CORRECT", "CONCEPT")
    )

    assert result["verdict"] == "incorrect"
    assert "Also mention" not in result["feedback"]


def test_grading_levels_are_unchanged_by_the_feedback(monkeypatch):
    with_similarity(monkeypatch, 0.55)

    for verdict_text, expected in (
        ("VERDICT: CORRECT\nTYPE: CONCEPT\nCONTRADICTS: NO\nMISSING: NONE", "correct"),
        ("VERDICT: PARTIAL\nTYPE: INCOMPLETE\nCONTRADICTS: NO\nMISSING: NONE", "partial"),
        ("VERDICT: INCORRECT\nTYPE: WRONG\nCONTRADICTS: YES\nMISSING: NONE", "incorrect"),
    ):
        result = check_answer(
            "It could be supervised learning or reinforcement learning, two types.",
            REAL_TYPES_BLANKLESS, FakeEmbedder(), llm=fixed_llm(verdict_text)
        )

        assert result["verdict"] == expected


def test_the_next_sentence_is_blanked_when_the_first_cannot_be():
    # Found on the real transcript: "This was a basic machine learning
    # algorithm also." is almost one phrase, so another sentence of the
    # section is used - not the broad last-resort question.
    point = {
        "title": "Definitely Algorithms",
        "summary": "(LLM-written summary description)",
        "start": 120.0,
        "end": 140.0,
        "text": (
            "If we go for the majority votes, we can say that Paul will "
            "definitely like the song. This was a basic machine learning "
            "algorithm also. So this is just a small example in one of the "
            "many machine learning algorithms."
        ),
    }

    item = generate_viva_questions([point], llm=broken_llm)[0]

    assert item["source"] == "transcript_blank"
    assert not item["question"].startswith("What does the video explain about")
    assert viva.BLANK in item["question"]


# ============================================================
# ROUND 5: A "DIFFERENCE" OR "WHY" QUESTION NEEDS SUCH AN ANSWER
# ============================================================

REAL_DIFFERENCE_QUESTION = (
    # what llama3.2 asked on the real video for "It could be either
    # supervised learning or reinforcement learning."
    "QUESTION: What is the difference between supervised learning and "
    "reinforcement learning?"
)


def test_the_real_difference_question_is_not_used_for_a_plain_list():
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        answer_first=REAL_DIFFERENCE_QUESTION,
        one_sentence=REAL_DIFFERENCE_QUESTION,
    )

    item = generate_viva_questions([REAL_TYPES_POINT], llm=llm)[0]

    assert "difference" not in item["question"].lower()

    # the safe next step: fill in the blank on the same sentence
    assert item["source"] == "transcript_blank"
    assert item["answer_sentence"] == (
        "It could be either supervised learning or reinforcement learning."
    )
    assert item["answer"] == "reinforcement learning"


def test_the_reason_is_logged(caplog):
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        answer_first=REAL_DIFFERENCE_QUESTION,
        one_sentence=REAL_DIFFERENCE_QUESTION,
    )

    with caplog.at_level("WARNING", logger="src.rag.viva"):
        generate_viva_questions([REAL_TYPES_POINT], llm=llm)

    assert "asks for a difference the answer does not give" in caplog.text


def test_a_difference_question_with_a_real_difference_is_kept():
    llm = fixed_llm(
        "QUESTION: What is the difference between the two clusters of "
        "players?\n"
        "ANSWER: The one cluster are the players who scored high runs and "
        "took less wickets, while the other cluster scored less runs but "
        "took many wickets."
    )

    item = generate_viva_questions([REAL_CRICKET_POINT], llm=llm)[0]

    assert item["source"] == "llm"
    assert "difference" in item["question"]


@pytest.mark.parametrize("question, answer, problem", [
    # difference
    ("What is the difference between supervised and reinforcement learning?",
     "It could be either supervised learning or reinforcement learning.", True),
    ("How do the two clusters differ?",
     "One cluster scored high runs, while the other took many wickets.", False),
    ("Compare supervised and unsupervised learning in the video?",
     "Supervised learning uses labeled data, unlike unsupervised learning.", False),
    # reason
    ("Why is it called unsupervised learning?",
     "It is unsupervised learning.", True),
    ("Why is it called unsupervised learning?",
     "Because there were no labels of batsmen and bowlers.", False),
    ("What is the reason it is unsupervised learning?",
     "Hence, the learning with unlabeled data is unsupervised learning.", False),
    # ordinary questions are not affected
    ("What does Paul love listening to?",
     "He loves listening to new songs.", False),
    ("What principle does reinforcement learning work on?",
     "It works on the principle of feedback.", False),
])
def test_questions_that_ask_for_more_than_a_fact(question, answer, problem):
    found = viva._asks_more_than_the_answer_gives(question, answer)

    assert (found is not None) is problem


def test_an_ordinary_one_sentence_question_is_still_used():
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        answer_first="",
        one_sentence="QUESTION: What does Paul love listening to?",
    )

    item = generate_viva_questions([REAL_PAUL_POINT], llm=llm)[0]

    assert item["source"] == "llm_answer_sentence"



# ============================================================
# ROUND 6: A CAUSE / EFFECT QUESTION NEEDS A CAUSE IN THE ANSWER
# ============================================================

REAL_CAUSE_QUESTION = (
    # what llama3.2 asked on the real video for "So that's Paul. He
    # loves listening to new songs. He either likes them or dislikes them."
    "QUESTION: What influences his preference for new songs?"
)


def test_the_real_influence_question_is_not_used_without_a_cause():
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        answer_first=REAL_CAUSE_QUESTION,
        one_sentence=REAL_CAUSE_QUESTION,
    )

    item = generate_viva_questions([REAL_PAUL_POINT], llm=llm)[0]

    assert "influence" not in item["question"].lower()
    assert item["source"] not in ("llm_answer_first", "llm_answer_sentence")


def test_the_cause_reason_is_logged(caplog):
    llm = routed_llm(
        step_one=REJECTED_STEP_ONE,
        answer_first=REAL_CAUSE_QUESTION,
        one_sentence=REAL_CAUSE_QUESTION,
    )

    with caplog.at_level("WARNING", logger="src.rag.viva"):
        generate_viva_questions([REAL_PAUL_POINT], llm=llm)

    assert "asks for a cause the answer does not give" in caplog.text


@pytest.mark.parametrize("question, answer, problem", [
    # a cause is asked and not given
    ("What influences his preference for new songs?",
     "He loves listening to new songs. He either likes them or dislikes them.", True),
    ("What causes surge pricing?",
     "Surge pricing is used by Uber.", True),
    ("Which factors affect the price of a trip?",
     "The fare of your trip has been updated.", True),
    ("How does more data affect the model?",
     "There are many ways in which the machine learns.", True),
    ("What is the effect of the feedback on the machine?",
     "That is reinforcement learning.", True),
    # a cause is asked and given
    ("What influences Paul's choice of songs?",
     "Paul decides this on the basis of the song's tempo, genre, intensity and the gender of voice.", False),
    ("What determines the surge price?",
     "Prices change in real time based on demand and the number of cars available.", False),
    ("How does more data affect the model?",
     "More data leads to a better model and higher accuracy.", False),
    # not cause questions: unchanged
    ("What does the model determine from the weight of a coin?",
     "The currency of the coin.", False),
    ("How does Paul decide whether he likes a song?",
     "On the tempo and intensity of the song.", False),
    ("What does supervised learning use?",
     "Labeled data.", False),
])
def test_cause_questions(question, answer, problem):
    found = viva._asks_more_than_the_answer_gives(question, answer)

    assert (found is not None) is problem
