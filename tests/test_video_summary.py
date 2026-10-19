import hashlib
import random
import re

import numpy as np
import pytest

from src.processing.chunker import create_chunks
from src.rag.video_summary import (
    MAX_EXTEND_SECONDS,
    MAX_POINTS,
    _build_sentences,
    _fallback_pool,
    _pick_fallback_sentence,
    _snap_windows,
    _starts_lowercase,
    _transcript_index,
    escape_markdown,
    format_time_range,
    format_timestamp,
    generate_video_summary,
    is_complete_sentence,
    sentence_problem,
)


# ------------------------------------------------------------
# Test doubles (no Whisper / MiniLM / Ollama needed)
# ------------------------------------------------------------

class FakeEmbeddingModel:
    """Deterministic bag-of-words embeddings."""

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        vectors = np.zeros((len(texts), 384), dtype="float32")

        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z']+", text.lower()):
                column = int(
                    hashlib.md5(word.encode()).hexdigest(), 16
                ) % 384
                vectors[row, column] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        return vectors


def segment(start, end, text):
    return {"start": start, "end": end, "text": " " + text}


# The real transcript of the project's sample video
REAL_SEGMENTS = [
    segment(0.0, 3.4, "We know humans learn from their past experiences"),
    segment(28.6, 31.0, "He either likes them or dislikes them."),
    segment(58.6, 63.6, "while he dislikes the song with relaxed tempo and light intensity."),
    segment(87.6, 90.2, "Let's say Paul listens to a new song."),
    segment(119.2, 121.2, "Let's see how, in the same example,"),
    segment(149.6, 151.0, "as in the case of song B?"),
    segment(180.0, 181.0, "and one dirham."),
    segment(209.0, 210.6, "Let's give a new coin to the machine."),
    segment(239.6, 243.0, "So it plots this data with the respective wickets and the x-axis,"),
    segment(267.0, 271.0, "Hence, the learning with unlabeled data is unsupervised learning."),
    segment(299.2, 301.4, "it will be able to classify it correctly."),
    segment(329.8, 330.6, "Simple, right?"),
    segment(357.6, 360.6, "Well, that's because today we have humongous data available."),
    segment(387.6, 390.6, "To name a few, machine learning is used in healthcare"),
    segment(418.6, 422.6, "which is used by global taxi giant Uber and others."),
    segment(448.6, 450.6, "Okay, I'll remind you."),
]


def good_llm(prompt):
    """A well-behaved model that only uses the excerpt it is given."""
    if "Transcript excerpt:" in prompt:
        excerpt = prompt.split("Transcript excerpt:")[1]
        excerpt = excerpt.split("Reply in exactly")[0].strip()
        words = [w for w in re.findall(r"[A-Za-z]{6,}", excerpt)][:2]
        return (
            f"TITLE: {' '.join(words) or 'Topic'}\n"
            f"SUMMARY: {excerpt.rstrip('.')}."
        )
    return "MAIN TOPIC: Machine Learning"


def chunk_triples(chunks):
    return {(c["start"], c["end"], c["text"]) for c in chunks}


# ------------------------------------------------------------
# Timestamps and evidence are never invented
# ------------------------------------------------------------

def test_every_point_uses_a_real_chunk_timestamp_and_text():
    chunks = create_chunks(REAL_SEGMENTS)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    assert result["points"]

    real = chunk_triples(chunks)

    for point in result["points"]:
        assert (
            point["start"], point["end"], point["text"]
        ) in real

        # the chunk it claims to come from really is that chunk
        source = chunks[point["chunk_index"]]
        assert point["start"] == source["start"]
        assert point["end"] == source["end"]
        assert point["text"] == source["text"]


def test_llm_cannot_change_timestamps_even_if_it_tries():
    chunks = create_chunks(REAL_SEGMENTS)

    def lying_llm(prompt):
        return (
            "TITLE: Something\n"
            "SUMMARY: At 999 seconds the video says something.\n"
            "start: 999"
        )

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=lying_llm
    )

    real = chunk_triples(chunks)

    for point in result["points"]:
        assert (point["start"], point["end"], point["text"]) in real
        assert point["start"] != 999


def test_each_point_has_its_own_evidence_and_timestamp():
    chunks = create_chunks(REAL_SEGMENTS)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    starts = [p["start"] for p in result["points"]]
    texts = [p["text"] for p in result["points"]]
    indexes = [p["chunk_index"] for p in result["points"]]

    assert len(set(starts)) == len(starts)
    assert len(set(texts)) == len(texts)
    assert len(set(indexes)) == len(indexes)

    for point in result["points"]:
        assert point["text"].strip()
        assert point["end"] >= point["start"]


def test_points_are_in_video_order_and_capped():
    chunks = create_chunks(REAL_SEGMENTS)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    starts = [p["start"] for p in result["points"]]

    assert starts == sorted(starts)
    assert len(result["points"]) <= MAX_POINTS

    capped = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm, max_points=2
    )

    assert len(capped["points"]) <= 2


# ------------------------------------------------------------
# Only meaningful chunks become points
# ------------------------------------------------------------

def test_filler_chunks_are_not_summary_points():
    chunks = create_chunks(REAL_SEGMENTS)

    # No cap: every chunk that passes the filter may become a point,
    # so only the "meaningful chunk" filter can keep filler out.
    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm, max_points=50
    )

    shown = " ".join(p["text"] for p in result["points"])

    for filler in (
        "Simple, right?",
        "and one dirham.",
        "Okay, I'll remind you.",
        "as in the case of song B?",
        "Let's see how, in the same example,",
    ):
        assert filler not in shown

    assert result["points"]


# ------------------------------------------------------------
# Duplicate topics collapse into one point
# ------------------------------------------------------------

def test_chunks_describing_the_same_topic_give_one_point():
    segments = [
        segment(10, 14, "Machine learning is widely used in healthcare for diagnosing diseases"),
        segment(60, 64, "Machine learning is widely used in healthcare for diagnosing diseases early"),
        segment(110, 114, "Machine learning is widely used in healthcare to diagnose diseases"),
        segment(160, 164, "Sentiment analysis classifies customer reviews as positive or negative opinions"),
    ]

    chunks = create_chunks(segments)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    healthcare = [
        p for p in result["points"] if "healthcare" in p["text"].lower()
    ]

    assert len(healthcare) == 1
    assert len(result["points"]) == 2


def test_three_distinct_topics_give_three_points_and_evidence():
    segments = [
        segment(10, 14, "Machine learning is widely used in healthcare for diagnosing diseases"),
        segment(60, 64, "Sentiment analysis classifies customer reviews as positive or negative opinions"),
        segment(110, 114, "Self driving cars use cameras and sensors to detect pedestrians on roads"),
    ]

    chunks = create_chunks(segments)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    assert len(result["points"]) == 3
    assert [p["start"] for p in result["points"]] == [10, 60, 110]


# ------------------------------------------------------------
# The summary only uses transcript content
# ------------------------------------------------------------

def test_each_llm_call_only_sees_its_own_chunk():
    chunks = create_chunks(REAL_SEGMENTS)
    prompts = []

    def recording_llm(prompt):
        prompts.append(prompt)
        return good_llm(prompt)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=recording_llm
    )

    point_prompts = [p for p in prompts if "Transcript excerpt:" in p]

    assert len(point_prompts) >= len(result["points"]) > 0

    seen = set()

    for prompt in point_prompts:
        contained = [c["text"] for c in chunks if c["text"] in prompt]

        # exactly one transcript chunk per point prompt
        assert len(contained) == 1
        seen.add(contained[0])

    # every point was worded from the chunk that is its evidence
    assert {p["text"] for p in result["points"]} <= seen


def test_hallucinated_llm_wording_is_rejected_for_extractive_text():
    chunks = create_chunks(REAL_SEGMENTS)

    def hallucinating_llm(prompt):
        if "Transcript excerpt:" in prompt:
            return (
                "TITLE: Quantum Blockchain Economics\n"
                "SUMMARY: Quantum computers will replace every bank by 2030."
            )
        return "MAIN TOPIC: Cryptocurrency Markets"

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=hallucinating_llm
    )

    assert result["points"]
    assert result["used_llm"] is False

    everything = " ".join(
        [result["main_topic"]]
        + [p["title"] + " " + p["summary"] for p in result["points"]]
    ).lower()

    for invented in ("quantum", "blockchain", "cryptocurrency", "bank"):
        assert invented not in everything


# ------------------------------------------------------------
# Failure handling
# ------------------------------------------------------------

def test_llm_failure_falls_back_to_extractive_summary():
    chunks = create_chunks(REAL_SEGMENTS)
    calls = []

    def broken_llm(prompt):
        calls.append(prompt)
        raise ConnectionError("Ollama is not running")

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=broken_llm
    )

    assert result["points"]
    assert result["used_llm"] is False
    assert result["main_topic"]

    # after the first failure the LLM is not hammered again
    assert len(calls) == 1

    real = chunk_triples(chunks)

    for point in result["points"]:
        assert point["title"].strip()
        assert point["summary"].strip()
        assert (point["start"], point["end"], point["text"]) in real


def test_messy_llm_formatting_is_understood():
    chunks = create_chunks([
        segment(387.6, 390.6, "To name a few, machine learning is used in healthcare"),
    ])

    def messy_llm(prompt):
        if "Transcript excerpt:" in prompt:
            return (
                "**Title:** \"Healthcare\"\n"
                "- Summary: Machine learning is used in healthcare."
            )
        return "**Main Topic:** machine learning"

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=messy_llm
    )

    point = result["points"][0]

    assert point["title"] == "Healthcare"
    assert point["summary"] == "Machine learning is used in healthcare."
    assert result["main_topic"] == "Machine Learning"
    assert result["used_llm"] is True


def test_empty_and_malformed_input_does_not_crash():
    model = FakeEmbeddingModel()

    for chunks in (None, [], [{}], [{"text": "", "start": 0, "end": 1}],
                   [{"text": "hello there friend", "start": "x", "end": 1}]):
        result = generate_video_summary(chunks, model, llm=good_llm)

        assert result["points"] == []
        assert result["main_topic"] == ""


def test_short_video_still_gets_a_summary():
    chunks = create_chunks([
        segment(0, 3, "Neural networks learn patterns"),
    ])

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    assert len(result["points"]) == 1
    assert result["points"][0]["start"] == 0


def test_summary_is_deterministic():
    chunks = create_chunks(REAL_SEGMENTS)

    first = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )
    second = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    assert first == second


# ------------------------------------------------------------
# Display safety
# ------------------------------------------------------------

def test_escape_markdown_neutralises_markup():
    escaped = escape_markdown("costs $5 and $10 *bold* #tag [x](y)")

    assert "\\$5" in escaped
    assert "\\*bold\\*" in escaped
    assert "\\#tag" in escaped
    assert "\\[x\\]" in escaped


# ============================================================
# PHASE 1: COMPLETE-SENTENCE DESCRIPTIONS
# ============================================================

# Continuous speech (no silences). The sentence
#   "The more labelled data we train a classifier on, higher will be
#    the accuracy of its predictions on new data."
# is cut in two by the 20-second window: its first half ends chunk 0
# and its second half starts chunk 1.
CONTINUOUS_SEGMENTS = [
    segment(0.0, 6.0, "Supervised learning trains a model on labelled examples."),
    segment(6.0, 13.0, "Each example pairs an input with the correct answer."),
    segment(13.0, 19.0, "The more labelled data we train a classifier on,"),
    segment(19.0, 26.0, "higher will be the accuracy of its predictions on new data."),
    segment(26.0, 33.0, "Unsupervised learning is different because the data has no labels."),
    segment(33.0, 39.0, "The algorithm looks for hidden groups in the data by itself."),
    segment(39.0, 46.0, "Reinforcement learning rewards an agent when it makes a good decision."),
    segment(46.0, 52.0, "Over many trials the agent learns which actions earn the most reward."),
]

CROSS_BOUNDARY_SENTENCE = (
    "The more labelled data we train a classifier on, higher will be "
    "the accuracy of its predictions on new data."
)


def llm_with(summary_for):
    """
    An LLM whose SUMMARY depends on the excerpt it is given.
    summary_for(excerpt) -> the SUMMARY text, or None to echo the
    first sentence of the excerpt.
    """
    def llm(prompt):
        if "Transcript excerpt:" in prompt:
            excerpt = prompt.split("Transcript excerpt:")[1]
            excerpt = excerpt.split("Reply in exactly")[0].strip()
            summary = summary_for(excerpt)
            if summary is None:
                summary = re.split(r"(?<=[.!?])\s+", excerpt)[0]
            return f"TITLE: Learning Types\nSUMMARY: {summary}"
        return "MAIN TOPIC: Machine Learning"
    return llm


def fragment_for_accuracy_chunk(fragment):
    """Returns the fragment only for the chunk that starts mid-sentence."""
    return llm_with(
        lambda excerpt: fragment if "higher will be" in excerpt else None
    )


def point_at(result, start):
    matches = [p for p in result["points"] if p["start"] == start]
    assert len(matches) == 1, f"no single point at {start}s"
    return matches[0]


def point_for_chunk(result, chunk_index):
    """The point built from a given chunk (its window may have moved)."""
    matches = [p for p in result["points"] if p["chunk_index"] == chunk_index]
    assert len(matches) == 1, f"no single point for chunk {chunk_index}"
    return matches[0]


def flat_segments(chunks):
    """The real Whisper segments, in order, as the chunker received them."""
    return [
        segment_
        for chunk in chunks
        for segment_ in chunk["segments"]
        if segment_["text"].strip()
    ]


def assert_is_a_run_of_real_segments(point, chunks):
    """
    The guarantee: start, end and text are exactly a run of consecutive
    real Whisper segments. Nothing is invented, reworded or re-timed.
    """
    segments = flat_segments(chunks)
    first, last = point["segment_range"]
    run = segments[first:last + 1]

    assert run, "empty run"
    assert point["start"] == float(run[0]["start"])
    assert point["end"] == float(run[-1]["end"])
    assert point["text"] == " ".join(s_["text"].strip() for s_ in run)

    # and, independently of the indexes: it is a stretch of the transcript
    whole = " ".join(s_["text"].strip() for s_ in segments)
    assert point["text"] in whole
    assert point["start"] in {float(s_["start"]) for s_ in segments}
    assert point["end"] in {float(s_["end"]) for s_ in segments}


def to_seconds(label):
    seconds = 0
    for part in label.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


# ------------------------------------------------------------
# The completeness check itself
# ------------------------------------------------------------

@pytest.mark.parametrize("text, reason", [
    # complete sentences are accepted
    ("Supervised learning uses labelled data to train a model.", None),
    ("Machine learning is used in healthcare.", None),
    ("How does a machine learn from labelled data?", None),
    ("3 types of learning are covered in this video.", None),
    ('"Machine learning is useful," the narrator says.', None),
    # starting with a lowercase letter on purpose is fine
    ("iPhone sales rose sharply during the last quarter.", None),
    # the reported fragment: starts in the middle of a sentence
    ("higher will be the accuracy.", "starts_lowercase"),
    # too short to be a sentence
    ("Higher accuracy.", "too_short"),
    ("Simple, right?", "too_short"),
    # no sentence-ending punctuation
    ("Machine learning is used in healthcare", "no_end_punctuation"),
    ("Machine learning is used in healthcare,", "no_end_punctuation"),
    # cut off
    ("Machine learning is used in healthcare...", "truncated"),
    ("The model is trained on the.", "dangling_end"),
    ("The accuracy of the model will.", "dangling_end"),
    # opens a dependent clause
    ("Which is used by global taxi giant Uber and others.", "fragment_start"),
    ("And then the accuracy of the model improves.", "fragment_start"),
    # unbalanced quotes
    ('He said "machine learning is fun.', "unbalanced"),
    # nothing usable
    ("", "empty"),
    (None, "empty"),
    ("   ", "empty"),
    (", and the accuracy of the model improves.", "starts_with_punctuation"),
])
def test_sentence_completeness_rules(text, reason):
    assert sentence_problem(text) == reason
    assert is_complete_sentence(text) is (reason is None)


def test_overlong_description_is_rejected_not_truncated():
    long_sentence = "This sentence keeps going " + "and going " * 40 + "forever."

    assert sentence_problem(long_sentence) == "too_long"
    assert sentence_problem(long_sentence, max_chars=1000) is None


# ------------------------------------------------------------
# Rejected LLM descriptions fall back to a real, complete sentence
# ------------------------------------------------------------

@pytest.mark.parametrize("fragment", [
    "higher will be the accuracy.",                    # lowercase start (the reported case)
    "higher will be the accuracy",                     # lowercase, no full stop
    "Higher accuracy.",                                # very short fragment
    "Higher will be the accuracy of its predictions",  # no sentence-ending punctuation
])
def test_fragment_from_llm_is_replaced_by_complete_transcript_sentence(fragment):
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    result = generate_video_summary(
        chunks,
        FakeEmbeddingModel(),
        llm=fragment_for_accuracy_chunk(fragment)
    )

    point = point_for_chunk(result, 1)

    # the fragment is never shown, anywhere
    assert all(p["summary"] != fragment for p in result["points"])

    # it was replaced by a complete sentence taken from the transcript
    assert point["description_source"] == "transcript_sentence"
    assert sentence_problem(point["summary"], 320) is None
    assert point["summary"] == CROSS_BOUNDARY_SENTENCE
    assert point["summary"] in [s["text"] for s in _build_sentences(chunks)]


def test_valid_grounded_description_is_accepted_unchanged():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    good = "Unsupervised learning looks for hidden groups in data that has no labels."

    result = generate_video_summary(
        chunks,
        FakeEmbeddingModel(),
        llm=llm_with(
            lambda excerpt: good if "Unsupervised" in excerpt else None
        )
    )

    point = point_for_chunk(result, 1)

    assert point["summary"] == good
    assert point["description_source"] == "llm"
    assert result["used_llm"] is True


def test_complete_but_ungrounded_description_is_still_rejected():
    """Completeness is not enough: the grounding protection stays."""
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    invented = "Quantum computers will replace every bank by the year 2030."

    assert is_complete_sentence(invented)

    result = generate_video_summary(
        chunks,
        FakeEmbeddingModel(),
        llm=llm_with(lambda excerpt: invented)
    )

    assert all(p["summary"] != invented for p in result["points"])
    assert all(
        p["description_source"] == "transcript_sentence"
        for p in result["points"]
    )


def test_only_the_first_sentence_of_a_long_llm_answer_is_used():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    result = generate_video_summary(
        chunks,
        FakeEmbeddingModel(),
        llm=llm_with(
            lambda excerpt: (
                "Unsupervised learning looks for hidden groups in the data. "
                "It also does many other things."
                if "Unsupervised" in excerpt else None
            )
        )
    )

    point = point_for_chunk(result, 1)

    assert point["summary"] == (
        "Unsupervised learning looks for hidden groups in the data."
    )


def test_every_description_is_a_complete_sentence_without_the_llm():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    def down(prompt):
        raise ConnectionError("Ollama is not running")

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=down
    )

    assert result["points"]

    for point in result["points"]:
        assert point["description_source"] == "transcript_sentence"
        assert sentence_problem(point["summary"], 320) is None


def test_real_video_transcript_never_shows_a_fragment():
    chunks = create_chunks(REAL_SEGMENTS)

    for llm in (good_llm, lambda prompt: (_ for _ in ()).throw(ConnectionError())):

        result = generate_video_summary(
            chunks, FakeEmbeddingModel(), llm=llm, max_points=50
        )

        assert result["points"]

        for point in result["points"]:
            assert sentence_problem(point["summary"], 320) is None, point["summary"]


# ------------------------------------------------------------
# Timestamps stay tied to the transcript chunk
# ------------------------------------------------------------

def test_timestamps_stay_real_segment_boundaries_when_the_description_is_replaced():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    result = generate_video_summary(
        chunks,
        FakeEmbeddingModel(),
        llm=fragment_for_accuracy_chunk("higher will be the accuracy.")
    )

    point = point_for_chunk(result, 1)

    # Chunk 1 began in the middle of a sentence whose first half is the
    # last segment of chunk 0, so the evidence now starts at that segment
    # (13.0s). That is a real segment boundary, nothing is invented.
    assert point["description_source"] == "transcript_sentence"
    assert point["start"] == 13.0
    assert point["end"] == 39.0
    assert point["text"].startswith("The more labelled data we train")
    assert_is_a_run_of_real_segments(point, chunks)

    for each in result["points"]:
        assert_is_a_run_of_real_segments(each, chunks)


def test_timestamps_are_real_segment_boundaries_for_every_wording_path():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    def down(prompt):
        raise ConnectionError("down")

    for llm in (
        good_llm,
        down,
        fragment_for_accuracy_chunk("higher will be the accuracy."),
        llm_with(lambda excerpt: "At 999 seconds the video says something."),
    ):
        result = generate_video_summary(chunks, FakeEmbeddingModel(), llm=llm)

        assert result["points"]

        for point in result["points"]:
            assert_is_a_run_of_real_segments(point, chunks)
            assert point["start"] != 999


# ------------------------------------------------------------
# Rebuilding sentences from the transcript
# ------------------------------------------------------------

def test_sentence_split_across_two_chunks_is_rebuilt():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)

    # chunk 0 ends and chunk 1 begins in the middle of one sentence
    assert chunks[0]["text"].endswith("classifier on,")
    assert chunks[1]["text"].startswith("higher will be")

    sentences = _build_sentences(chunks)

    rebuilt = [s for s in sentences if "classifier" in s["text"]]

    assert len(rebuilt) == 1
    assert rebuilt[0]["text"] == CROSS_BOUNDARY_SENTENCE
    assert rebuilt[0]["chunks"] == [0, 1]
    assert rebuilt[0]["complete"] is True


def test_long_silence_ends_a_sentence_even_without_a_full_stop():
    run_on = [
        segment(0.0, 6.0, "Machine learning needs a lot of labelled data"),
        segment(6.2, 12.0, "Unsupervised learning finds hidden groups in the data."),
    ]

    silence = [
        segment(0.0, 6.0, "Machine learning needs a lot of labelled data"),
        segment(12.0, 18.0, "Unsupervised learning finds hidden groups in the data."),
    ]

    # one continuous sentence when the speaker does not pause ...
    assert len(_build_sentences(create_chunks(run_on))) == 1

    # ... two sentences when there is a long silence in between
    sentences = _build_sentences(create_chunks(silence))

    assert len(sentences) == 2
    assert sentences[0]["complete"] is False
    assert sentences[1]["complete"] is True


def test_abbreviations_do_not_split_sentences():
    chunks = create_chunks([
        segment(0.0, 8.0, "Dr. Smith explained the method to the whole class."),
    ])

    sentences = _build_sentences(chunks)

    assert [s["text"] for s in sentences] == [
        "Dr. Smith explained the method to the whole class."
    ]


# ------------------------------------------------------------
# Chunks that cannot be described honestly, and duplicates
# ------------------------------------------------------------

def test_fragment_only_chunks_are_left_out_instead_of_shown():
    # Fragments surrounded by silence, with complete sentences further away.
    segments = [
        segment(0.0, 6.0, "Supervised learning trains a model on labelled examples."),
        segment(30.0, 34.0, "The more labelled data we train a classifier on,"),
        segment(50.0, 58.0, "higher will be the accuracy of its predictions on new data"),
        segment(80.0, 88.0, "Unsupervised learning finds hidden groups in data that has no labels."),
    ]

    chunks = create_chunks(segments)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    assert [p["start"] for p in result["points"]] == [0.0, 80.0]

    for point in result["points"]:
        assert sentence_problem(point["summary"], 320) is None


def test_two_chunks_sharing_one_sentence_give_one_point():
    chunks = create_chunks([
        segment(0.0, 12.0, "Neural networks learn useful features from large collections of labelled images,"),
        segment(12.0, 24.0, "so that they can recognise objects in photographs they have never seen before."),
    ])

    assert len(chunks) == 2

    def down(prompt):
        raise ConnectionError("down")

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=down
    )

    summaries = [p["summary"] for p in result["points"]]

    assert len(summaries) == len(set(summaries)) == 1
    assert sentence_problem(summaries[0], 320) is None
    assert result["points"][0]["start"] == 0.0


def test_neighbouring_points_never_share_a_sentence():
    """
    One sentence (X) runs across the boundary of two chunks. It must end
    up in exactly one point; the next point gets the rest (Y).
    """
    chunks = create_chunks([
        segment(0.0, 12.0, "Neural networks learn useful features from large collections of labelled images,"),
        segment(12.0, 24.0, "so that they can recognise objects in photographs they have never seen before."),
        segment(24.0, 30.0, "Training such a network takes a lot of computing power and time."),
    ])

    assert len(chunks) == 2

    x = (
        "Neural networks learn useful features from large collections of "
        "labelled images, so that they can recognise objects in "
        "photographs they have never seen before."
    )
    y = "Training such a network takes a lot of computing power and time."

    def llm(prompt):
        if "Transcript excerpt:" in prompt:
            # a fragment whose words all lie inside sentence X
            return (
                "TITLE: Image Recognition\n"
                "SUMMARY: recognise objects in photographs they have never seen before."
            )
        return "MAIN TOPIC: Neural Networks"

    result = generate_video_summary(chunks, FakeEmbeddingModel(), llm=llm)

    assert [p["summary"] for p in result["points"]] == [x, y]
    assert [(p["start"], p["end"]) for p in result["points"]] == [(0.0, 24.0), (24.0, 30.0)]

    for point in result["points"]:
        assert_is_a_run_of_real_segments(point, chunks)


def test_the_fallback_picker_skips_sentences_that_are_already_used():
    sentences = [
        {"id": 0, "text": "Neural networks learn features from labelled images."},
        {"id": 1, "text": "Training such a network takes a lot of computing power."},
    ]
    pool = [(0, sentences[0]), (0, sentences[1])]
    draft = "learn features from labelled images"

    assert _pick_fallback_sentence(pool, draft, used=set())["id"] == 0
    assert _pick_fallback_sentence(pool, draft, used={0})["id"] == 1
    assert _pick_fallback_sentence(pool, draft, used={0, 1}) is None


def test_video_without_any_complete_sentence_still_returns_a_point():
    chunks = create_chunks([
        segment(0.0, 5.0, "neural networks learn patterns from data"),
    ])

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm
    )

    assert len(result["points"]) == 1
    assert result["points"][0]["description_source"] == "chunk_text"
    assert result["points"][0]["start"] == 0.0


# ------------------------------------------------------------
# Timestamp display
# ------------------------------------------------------------

@pytest.mark.parametrize("seconds, expected", [
    (0, "0:00"),
    (5, "0:05"),
    (59.9, "0:59"),
    (60, "1:00"),
    (143.0, "2:23"),
    (161.4, "2:41"),
    (387.6, "6:27"),
    (3599, "59:59"),
    (3600, "1:00:00"),
    (3725.2, "1:02:05"),
])
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_time_range():
    assert format_time_range(143.0, 161.4) == "2:23 \u2013 2:41"
    assert format_time_range(3725, 3790) == "1:02:05 \u2013 1:03:10"

    # no pointless "1:27 - 1:27"
    assert format_time_range(87.6, 87.9) == "1:27"


def test_jump_label_equals_the_second_the_player_seeks_to():
    chunks = create_chunks(REAL_SEGMENTS)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm, max_points=50
    )

    for point in result["points"]:
        # app.py seeks to int(start); the label must show that second
        assert to_seconds(format_timestamp(point["start"])) == int(point["start"])


# ============================================================
# REAL-VIDEO REGRESSION: a sentence that crosses a chunk boundary
# ============================================================
#
# Taken from a real run on a 7:51 Machine Learning video. The point was:
#
#     Title:        How Machine Learning Works
#     Description:  It could be either supervised learning or
#                   reinforcement learning.
#     Timestamp:    2:41 - 3:01
#     Evidence:     "higher will be the accuracy. There are many ways
#                    in which the machine learns..."
#
# The description was fine. The EVIDENCE (and the 2:41 timestamp) began in
# the middle of a sentence: the 20-second window cut it in two. The words
# before the cut are not known, so the first half below is representative;
# the logic does not depend on them.
#
# 5-second Whisper segments, so that the window boundary falls at 161.0s.
USER_CASE_SEGMENTS = [
    segment(141.0, 146.0, "Machine learning lets a computer improve at a task by learning from examples."),
    segment(146.0, 151.0, "Instead of following fixed rules it finds patterns in the training data."),
    segment(151.0, 156.0, "A model that sees more examples usually becomes better at spotting those patterns."),
    segment(156.0, 161.0, "The more data we give the machine while it learns, the"),
    segment(161.0, 166.0, "higher will be the accuracy."),
    segment(166.0, 171.0, "There are many ways in which the machine learns."),
    segment(171.0, 176.0, "It could be either supervised learning or reinforcement learning."),
    segment(176.0, 181.0, "In supervised learning the machine gets labelled examples to learn from."),
    segment(181.0, 186.0, "In reinforcement learning an agent receives a reward for each good decision."),
    segment(186.0, 191.0, "Over many trials the agent discovers which actions earn the largest reward."),
    segment(191.0, 196.0, "Robots and game playing programs are often trained in this way."),
    segment(196.0, 201.0, "Unsupervised learning on the other hand works with data that has no labels."),
]

REAL_DESCRIPTION = "It could be either supervised learning or reinforcement learning."

RECONSTRUCTED_SENTENCE = (
    "The more data we give the machine while it learns, the higher "
    "will be the accuracy."
)


def real_llm_for_user_case(prompt):
    """What the real llama3.2 returned for this chunk."""
    if "Transcript excerpt:" in prompt:
        if "higher will be the accuracy" in prompt:
            return (
                "TITLE: How Machine Learning Works\n"
                f"SUMMARY: {REAL_DESCRIPTION}"
            )
        excerpt = prompt.split("Transcript excerpt:")[1]
        excerpt = excerpt.split("Reply in exactly")[0].strip()
        return "TITLE: Learning Basics\nSUMMARY: " + excerpt.split(". ")[0].rstrip(".") + "."
    return "MAIN TOPIC: Machine Learning"


def test_user_case_the_raw_chunk_really_starts_with_the_fragment():
    """The situation itself: this is what the old code displayed."""
    chunks = create_chunks(USER_CASE_SEGMENTS)

    assert (chunks[1]["start"], chunks[1]["end"]) == (161.0, 181.0)
    assert format_time_range(161.0, 181.0) == "2:41 \u2013 3:01"
    assert chunks[1]["text"].startswith(
        "higher will be the accuracy. There are many ways"
    )
    assert _starts_lowercase(chunks[1]["text"])


def test_user_case_evidence_starts_with_the_complete_sentence():
    chunks = create_chunks(USER_CASE_SEGMENTS)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=real_llm_for_user_case
    )

    point = point_for_chunk(result, 1)

    # the description from the real run is unchanged: grounded and complete
    assert point["title"] == "How Machine Learning Works"
    assert point["summary"] == REAL_DESCRIPTION
    assert point["description_source"] == "llm"
    assert sentence_problem(point["summary"]) is None

    # the evidence no longer begins with the fragment ...
    assert not point["text"].startswith("higher will be the accuracy")
    assert not _starts_lowercase(point["text"])

    # ... it begins with the whole sentence, exactly as it was said
    assert point["text"].startswith(RECONSTRUCTED_SENTENCE + " There are many ways")

    # the timestamp is the real start of the segment where the sentence began
    assert (point["start"], point["end"]) == (156.0, 181.0)
    assert format_time_range(point["start"], point["end"]) == "2:36 \u2013 3:01"
    assert format_timestamp(point["start"]) == "2:36"
    assert to_seconds(format_timestamp(point["start"])) == int(point["start"])

    # nothing is invented: it is a run of real segments, word for word
    assert_is_a_run_of_real_segments(point, chunks)
    assert RECONSTRUCTED_SENTENCE in " ".join(
        s_["text"].strip() for s_ in flat_segments(chunks)
    )


def test_user_case_the_neighbouring_point_gives_up_the_sentence_and_ends_cleanly():
    chunks = create_chunks(USER_CASE_SEGMENTS)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=real_llm_for_user_case
    )

    before = point_for_chunk(result, 0)
    point = point_for_chunk(result, 1)

    # chunk 0 no longer ends on "..., the": the unfinished sentence moved
    assert before["text"].endswith("spotting those patterns.")
    assert before["end"] == 156.0

    # windows are disjoint: the shared sentence is shown exactly once
    assert before["end"] <= point["start"]
    assert before["text"].count("higher will be the accuracy") == 0

    for each in result["points"]:
        assert_is_a_run_of_real_segments(each, chunks)
        assert each["text"][-1] in ".!?"
        assert not _starts_lowercase(each["text"])


def test_user_case_snapped_windows_are_whole_sentences():
    chunks = create_chunks(USER_CASE_SEGMENTS)
    windows = _snap_windows(_transcript_index(chunks))

    # chunk 1 (161-181s) now holds segments 3..7 (156-181s)
    assert windows == {0: (0, 2), 1: (3, 7), 2: (8, 11)}


def test_user_case_if_the_llm_returns_the_fragment_it_is_repaired_not_shown():
    chunks = create_chunks(USER_CASE_SEGMENTS)

    def fragment_llm(prompt):
        if "Transcript excerpt:" in prompt and "higher will be the accuracy" in prompt:
            return "TITLE: Model Accuracy\nSUMMARY: higher will be the accuracy."
        return real_llm_for_user_case(prompt)

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=fragment_llm
    )

    point = point_for_chunk(result, 1)

    assert all(p["summary"] != "higher will be the accuracy." for p in result["points"])
    assert point["description_source"] == "transcript_sentence"
    assert point["summary"] == RECONSTRUCTED_SENTENCE
    assert point["summary"] in point["text"]


def test_user_case_works_without_the_llm_too():
    chunks = create_chunks(USER_CASE_SEGMENTS)

    def down(prompt):
        raise ConnectionError("Ollama is not running")

    result = generate_video_summary(chunks, FakeEmbeddingModel(), llm=down)

    for point in result["points"]:
        assert not _starts_lowercase(point["text"])
        assert sentence_problem(point["summary"], 320) is None
        assert_is_a_run_of_real_segments(point, chunks)


# ------------------------------------------------------------
# When the beginning of the sentence is too far away
# ------------------------------------------------------------

def test_a_fragment_whose_beginning_is_too_far_back_is_dropped_not_shown():
    """
    The sentence began more than MAX_EXTEND_SECONDS before the window
    (a 22 second segment) and ends more than MAX_EXTEND_SECONDS after the
    window boundary, so it can be given to neither window. The fragment
    must not be shown: the window starts at the next real sentence.
    """
    chunks = create_chunks([
        segment(0.0, 10.0, "This lecture covers the basic ideas behind learning from data."),
        segment(10.0, 32.0, "The more data we feed the machine while it learns over many long hours of training, the"),
        segment(32.0, 52.0, "higher will be the accuracy of its predictions on all of the new data that it has never seen before."),
        segment(52.0, 58.0, "There are many ways in which the machine can learn from examples."),
        segment(58.0, 64.0, "It could be either supervised learning or reinforcement learning."),
    ])

    assert MAX_EXTEND_SECONDS < 22

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm, max_points=50
    )

    assert result["points"]

    for point in result["points"]:
        assert not _starts_lowercase(point["text"])
        assert_is_a_run_of_real_segments(point, chunks)
        assert not point["text"].startswith("higher will be")


def test_a_sentence_too_far_back_goes_to_the_earlier_window_when_it_can():
    """The head is 22s back, but the tail ends just after the boundary."""
    chunks = create_chunks([
        segment(0.0, 10.0, "This lecture covers the basic ideas behind learning from data."),
        segment(10.0, 32.0, "The more data we feed the machine while it learns over many long hours of training, the"),
        segment(32.0, 36.0, "higher will be the accuracy."),
        segment(36.0, 46.0, "There are many ways in which the machine can learn from examples."),
    ])

    windows = _snap_windows(_transcript_index(chunks))
    segments = _transcript_index(chunks)["segments"]

    # the whole sentence (10s-36s) stays together in ONE window
    owners = [
        n for n, (first, last) in windows.items()
        if first <= 1 and last >= 2
    ]

    assert len(owners) == 1


# ------------------------------------------------------------
# Many different cuts of the same kind of transcript
# ------------------------------------------------------------

SENTENCE_POOL = [
    "Machine learning is a method where computers learn patterns from data instead of following fixed rules.",
    "In supervised learning the training data comes with labels that tell the model the correct answer.",
    "The more labelled data we train a classifier on, higher will be the accuracy of its predictions on new data.",
    "A spam filter is a familiar example because every training email is marked as spam or not spam.",
    "Unsupervised learning is different because the data has no labels at all.",
    "The algorithm looks for hidden groups in the data and organises similar items together by itself.",
    "Reinforcement learning works through rewards, where an agent receives a reward for a good decision.",
    "Over many trials the agent learns which actions lead to the highest total reward.",
    "If a model memorises the training data it performs badly on new examples, which is called overfitting.",
    "Deep learning uses neural networks with many layers to learn features directly from raw data.",
    "Banks use it to spot unusual transactions that may indicate fraud on a customer account.",
    "Self driving cars combine cameras and sensors with models that detect pedestrians and road signs.",
]


def randomly_cut_transcript(seed):
    """Continuous speech; every sentence is cut into 1-3 segments."""
    rng = random.Random(seed)
    segments, clock = [], 0.0

    for _ in range(rng.randint(14, 26)):
        words = rng.choice(SENTENCE_POOL).split()
        pieces = rng.choice([1, 1, 2, 3])
        cuts = sorted(rng.sample(range(1, len(words)), pieces - 1)) if pieces > 1 else []
        bounds = [0, *cuts, len(words)]

        for a, b in zip(bounds, bounds[1:]):
            duration = round(0.45 * (b - a) + rng.uniform(0.5, 2.0), 2)
            segments.append(
                segment(round(clock, 2), round(clock + duration, 2), " ".join(words[a:b]))
            )
            clock += duration

    return segments


@pytest.mark.parametrize("seed", range(40))
def test_no_point_ever_starts_mid_sentence_on_randomly_cut_transcripts(seed):
    chunks = create_chunks(randomly_cut_transcript(seed))

    def down(prompt):
        raise ConnectionError("down")

    for llm in (good_llm, down):

        result = generate_video_summary(
            chunks, FakeEmbeddingModel(), llm=llm, max_points=50
        )

        assert result["points"], f"seed {seed}: no points"

        previous_end = None

        for point in sorted(result["points"], key=lambda p: p["start"]):

            # never a lowercase fragment at the start of the evidence ...
            assert not _starts_lowercase(point["text"]), (seed, point["text"][:60])
            assert point["text"][0].isupper(), (seed, point["text"][:60])

            # ... nor at the end: whole sentences only
            assert point["text"][-1] in ".!?", (seed, point["text"][-40:])

            # grounded: exactly a run of real segments
            assert_is_a_run_of_real_segments(point, chunks)

            # the description is a complete sentence from the transcript
            assert sentence_problem(point["summary"], 320) is None, (seed, point["summary"])

            # no two points repeat the same stretch of the video
            if previous_end is not None:
                assert point["start"] >= previous_end, seed

            previous_end = point["end"]


# ------------------------------------------------------------
# The limits of the snapping, pinned exactly
# ------------------------------------------------------------

def test_a_sentence_too_far_from_the_boundary_is_given_to_neither_window():
    """
    The sentence begins 16s before the chunk boundary (segment 1) and ends
    16s after it (segment 3). MAX_EXTEND_SECONDS is 15, so it can be given
    to neither window: the earlier window stops before it, the later one
    starts after it, and no half of it is ever shown.
    """
    chunks = create_chunks([
        segment(0.0, 3.0, "This lecture covers the basic ideas behind learning from data."),
        segment(3.0, 19.0, "The more data we feed the machine while it learns over many long hours of training, the"),
        segment(19.0, 27.0, "higher will be the accuracy of its predictions on all of the"),
        segment(27.0, 35.0, "new data that it has never seen before."),
        segment(35.0, 39.0, "There are many ways in which the machine can learn from examples."),
        segment(39.0, 45.0, "It could be either supervised learning or reinforcement learning."),
    ])

    index = _transcript_index(chunks)

    assert 16.0 > MAX_EXTEND_SECONDS
    assert index["chunk_ranges"] == {0: (0, 1), 1: (2, 4), 2: (5, 5)}

    # earlier window: unfinished head left out; later window: leading
    # fragment dropped, so it starts at the next real sentence (segment 4)
    assert _snap_windows(index) == {0: (0, 0), 1: (4, 4), 2: (5, 5)}

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm, max_points=50
    )

    shown = " ".join(p["text"] for p in result["points"])

    assert "feed the machine" not in shown
    assert "higher will be" not in shown
    assert "never seen before" not in shown

    for point in result["points"]:
        assert not _starts_lowercase(point["text"])
        assert_is_a_run_of_real_segments(point, chunks)


def test_an_unfixable_fragment_is_not_shown_even_if_no_other_point_uses_its_sentence():
    """
    The head of the sentence is too short to be a point of its own, so
    nothing else uses the sentence. The fragment window must still not be
    shown. (If another point used the sentence, the duplicate rule would
    hide it, which would make this guard impossible to test.)
    """
    chunks = create_chunks([
        segment(0.0, 10.0, "This lecture covers the basic ideas behind learning from data."),
        segment(10.0, 32.0, "The more data we feed it, the"),
        segment(32.0, 52.0, "higher will be the accuracy of its predictions on all of the new data that it has never seen before."),
        segment(52.0, 58.0, "There are many ways in which the machine can learn from examples."),
    ])

    result = generate_video_summary(
        chunks, FakeEmbeddingModel(), llm=good_llm, max_points=50
    )

    assert result["points"]
    assert all(p["chunk_index"] != 2 for p in result["points"])

    for point in result["points"]:
        assert not _starts_lowercase(point["text"])


# ------------------------------------------------------------
# The two helpers, tested directly
# ------------------------------------------------------------

def test_the_fallback_pool_only_holds_complete_sentences_that_touch_the_window():
    chunks = create_chunks(CONTINUOUS_SEGMENTS)
    index = _transcript_index(chunks)
    first, last = _snap_windows(index)[2]

    pool = _fallback_pool(index, first, last)
    in_pool = {sentence["id"] for _, sentence in pool}

    assert pool

    for rank, sentence in pool:
        begin, end = sentence["segments"]

        assert sentence["complete"]
        assert end >= first and begin <= last
        assert rank == (0 if begin >= first and end <= last else 1)

    far_away = [
        sentence for sentence in index["sentences"]
        if sentence["complete"]
        and (sentence["segments"][1] < first or sentence["segments"][0] > last)
    ]

    assert far_away
    assert not any(sentence["id"] in in_pool for sentence in far_away)

    assert _fallback_pool(index, None, None) == []


def test_the_fallback_picker_prefers_the_sentence_the_rejected_draft_came_from():
    long_sentence = {
        "id": 0,
        "text": "Neural networks learn useful features from large collections of labelled images.",
    }
    short_sentence = {
        "id": 1,
        "text": "Training takes a lot of computing power.",
    }
    pool = [(0, long_sentence), (0, short_sentence)]

    # no draft: the most informative sentence
    assert _pick_fallback_sentence(pool, None, used=set())["id"] == 0

    # a fragment of the shorter sentence: that sentence, although the other
    # one carries more information
    assert _pick_fallback_sentence(pool, "lot of computing power.", used=set())["id"] == 1

    # a draft about something else changes nothing ...
    assert _pick_fallback_sentence(pool, "bank fraud detection systems", used=set())["id"] == 0

    # ... and so does one that merely shares a word with a sentence
    assert _pick_fallback_sentence(pool, "bank fraud detection systems power", used=set())["id"] == 0


def test_the_fallback_picker_prefers_a_sentence_inside_the_evidence_to_a_better_match_outside_it():
    inside = {"id": 0, "text": "Training takes a lot of computing power."}
    straddling = {"id": 1, "text": "Neural networks learn useful features from large collections of labelled images."}

    pool = [(0, inside), (1, straddling)]

    # the draft is a piece of the straddling sentence, but the description
    # has to be about the evidence that is shown
    assert _pick_fallback_sentence(
        pool, "features from large collections of labelled images.", used=set()
    )["id"] == 0


def test_user_case_a_fragment_of_a_shorter_sentence_is_repaired_with_that_sentence():
    chunks = create_chunks(USER_CASE_SEGMENTS)

    def llm(prompt):
        if "Transcript excerpt:" in prompt and "higher will be the accuracy" in prompt:
            return "TITLE: Ways To Learn\nSUMMARY: many ways in which the machine learns."
        return real_llm_for_user_case(prompt)

    result = generate_video_summary(chunks, FakeEmbeddingModel(), llm=llm)

    point = point_for_chunk(result, 1)

    # not the longest sentence of the window, but the one the draft came from
    assert point["description_source"] == "transcript_sentence"
    assert point["summary"] == "There are many ways in which the machine learns."
    assert point["summary"] in point["text"]