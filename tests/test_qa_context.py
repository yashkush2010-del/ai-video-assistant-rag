"""
Q&A context: neighbouring chunks, video order and the answer prompt.

Real problems this fixes (scripts/check_qa_real_video.py, before):

  - "Paul decides this on the basis of the song's tempo," ended one
    chunk and "genre, intensity and the gender of voice." began the
    next. Only the first was found, so the answer only named the tempo.
  - The parts were listed by search score, so the model did not
    connect "connected indirectly" (1:43) with "through an Internet
    service provider" (1:47) for "Why is a home computer a client?".
  - The prompt only said "use the context"; answers repeated one
    sentence, left facts out or added things the video did not say.
"""

import sys
import types

import numpy as np
import pytest

from src.processing.chunker import create_chunks
from src.rag.vector_store import (
    MAX_CONTEXT_CHUNKS,
    add_neighbouring_chunks,
    create_index,
    retrieve_chunks,
)


def make_chunks(count):
    return [
        {
            "start": number * 20.0,
            "end": number * 20.0 + 20.0,
            "text": f"Chunk number {number}.",
            "segments": [{"start": number * 20.0, "end": number * 20.0 + 20.0,
                          "text": f"Chunk number {number}."}],
        }
        for number in range(count)
    ]


def match(chunks, number, score):
    chunk = chunks[number]
    return {
        "score": score,
        "chunk_index": number,
        "start": chunk["start"],
        "end": chunk["end"],
        "text": chunk["text"],
        "segments": chunk["segments"],
    }


def positions(results):
    return [result["chunk_index"] for result in results]


# ============================================================
# retrieve_chunks says where each match is
# ============================================================

def test_every_match_knows_its_position_in_the_video():
    vectors = np.eye(4, dtype="float32")
    chunks = make_chunks(4)

    results = retrieve_chunks(create_index(vectors), chunks, vectors[2], k=2)

    assert results[0]["chunk_index"] == 2
    assert results[0]["text"] == "Chunk number 2."


# ============================================================
# add_neighbouring_chunks
# ============================================================

def test_the_chunk_before_and_after_a_match_are_added():
    chunks = make_chunks(10)

    results = add_neighbouring_chunks([match(chunks, 5, 0.7)], chunks)

    assert positions(results) == [4, 5, 6]


def test_the_result_is_in_video_order():
    chunks = make_chunks(10)

    results = add_neighbouring_chunks(
        [match(chunks, 7, 0.8), match(chunks, 2, 0.6)], chunks
    )

    assert positions(results) == [1, 2, 3, 6, 7, 8]
    assert [r["start"] for r in results] == sorted(r["start"] for r in results)


def test_a_chunk_is_never_added_twice():
    chunks = make_chunks(10)

    results = add_neighbouring_chunks(
        [match(chunks, 4, 0.8), match(chunks, 5, 0.7)], chunks
    )

    assert positions(results) == [3, 4, 5, 6]


def test_the_first_and_last_chunk_have_one_neighbour():
    chunks = make_chunks(5)

    results = add_neighbouring_chunks(
        [match(chunks, 0, 0.8), match(chunks, 4, 0.7)], chunks
    )

    assert positions(results) == [0, 1, 3, 4]


def test_the_total_is_capped_and_real_matches_come_first():
    chunks = make_chunks(40)
    matches = [match(chunks, n, 0.9 - n / 100) for n in (2, 8, 14, 20, 26, 32)]

    results = add_neighbouring_chunks(matches, chunks, max_chunks=8)

    assert len(results) == 8

    # all six matches are kept; only two neighbours fit, the best match's
    for number in (2, 8, 14, 20, 26, 32):
        assert number in positions(results)

    assert positions(results) == [1, 2, 3, 8, 14, 20, 26, 32]


def test_the_default_cap():
    chunks = make_chunks(60)
    matches = [match(chunks, n, 0.5) for n in range(0, 60, 5)]

    assert len(add_neighbouring_chunks(matches, chunks)) == MAX_CONTEXT_CHUNKS


def test_a_neighbour_carries_the_score_of_its_match():
    chunks = make_chunks(10)

    results = add_neighbouring_chunks([match(chunks, 5, 0.7)], chunks)

    assert [r["score"] for r in results] == [0.7, 0.7, 0.7]
    assert results[0]["segments"] == chunks[4]["segments"]


def test_no_matches_gives_nothing():
    assert add_neighbouring_chunks([], make_chunks(5)) == []


def test_results_without_a_position_are_left_unchanged():
    old_style = [{"score": 0.5, "start": 0.0, "end": 5.0, "text": "x", "segments": []}]

    assert add_neighbouring_chunks(old_style, make_chunks(5)) == old_style


# ============================================================
# prepare_context
# ============================================================

@pytest.fixture
def answer_generator(monkeypatch):
    """src.rag.answer_generator with a stand-in for ollama that records calls."""
    ollama = types.ModuleType("ollama")
    ollama.calls = []
    ollama.reply = "Paul uses the tempo, genre, intensity and gender of voice."

    def chat(**kwargs):
        ollama.calls.append(kwargs)
        return {"message": {"content": ollama.reply}}

    ollama.chat = chat
    ollama.ResponseError = type("ResponseError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "ollama", ollama)
    monkeypatch.setitem(sys.modules, "whisper", types.ModuleType("whisper"))

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    monkeypatch.delitem(sys.modules, "src.rag.answer_generator", raising=False)

    import src.rag.answer_generator as module

    module.fake_ollama = ollama

    yield module

    monkeypatch.delitem(sys.modules, "src.rag.answer_generator", raising=False)


def test_the_context_is_in_video_order_with_readable_times(answer_generator):
    chunks = make_chunks(10)
    context = answer_generator.prepare_context(
        [match(chunks, 7, 0.9), match(chunks, 2, 0.5)]
    )

    assert context.index("Chunk number 2.") < context.index("Chunk number 7.")
    assert "[0:40 - 1:00]" in context
    assert "[2:20 - 2:40]" in context
    assert "140.0" not in context


def test_a_sentence_cut_between_two_chunks_is_joined_again(answer_generator):
    # the real Paul example (machine learning video, 0:17 - 0:52)
    segments = [
        {"start": 17.6, "end": 34.6,
         "text": " Paul decides this on the basis of the song's tempo,"},
        {"start": 34.6, "end": 52.6,
         "text": " genre, intensity and the gender of voice."},
    ]
    chunks = create_chunks(segments)

    assert len(chunks) == 2

    results = add_neighbouring_chunks(
        [match(chunks, 0, 0.7)], chunks
    )
    context = answer_generator.prepare_context(results)

    assert "song's tempo, genre, intensity and the gender of voice." in context
    assert context.count("[") == 1
    assert context.startswith("[0:17 - 0:52]")


def test_parts_that_do_not_touch_stay_separate(answer_generator):
    chunks = make_chunks(10)
    context = answer_generator.prepare_context(
        [match(chunks, 1, 0.5), match(chunks, 5, 0.5)]
    )

    assert "[0:20 - 0:40]" in context
    assert "[1:40 - 2:00]" in context


def test_long_videos_show_hours(answer_generator):
    context = answer_generator.prepare_context(
        [{"start": 3725.2, "end": 3740.0, "text": "Late in the video."}]
    )

    assert context.startswith("[1:02:05 - 1:02:20]")


# ============================================================
# generate_answer
# ============================================================

def test_the_prompt_asks_for_a_complete_answer_from_the_video_only(answer_generator):
    answer_generator.generate_answer(
        "[0:17 - 0:52]\nPaul decides this on the basis of the song's tempo, genre.\n\n",
        "How does Paul decide?",
    )

    call = answer_generator.fake_ollama.calls[0]
    prompt = call["messages"][0]["content"]

    assert "How does Paul decide?" in prompt
    assert "on the basis of the song's tempo, genre." in prompt
    assert "Use only what the transcript parts say" in prompt
    assert "complete answer" in prompt
    assert "every reason, factor, step or example" in prompt
    assert "2 to 4 sentences" in prompt
    assert "I don't have enough information in the video." in prompt


def test_the_same_question_always_gets_the_same_answer(answer_generator):
    answer_generator.generate_answer("context", "question")

    options = answer_generator.fake_ollama.calls[0]["options"]

    assert options["temperature"] == 0
    assert options["num_ctx"] >= 4096


@pytest.mark.parametrize("context", ["", "   \n\n", None])
def test_when_nothing_was_found_the_model_is_not_asked(answer_generator, context):
    # Real case: "Which programming language does the video use?" found
    # no transcript part, and llama3.2 replied "Go ahead and ask your
    # question" instead of saying the video has no answer.
    answer = answer_generator.generate_answer(context, "Which programming language?")

    assert answer == "I don't have enough information in the video."
    assert answer_generator.fake_ollama.calls == []


def test_braces_in_the_transcript_do_not_break_the_prompt(answer_generator):
    answer_generator.generate_answer("[0:00 - 0:05]\nUse {curly} braces.\n\n", "What {is} it?")

    prompt = answer_generator.fake_ollama.calls[0]["messages"][0]["content"]

    assert "Use {curly} braces." in prompt
    assert "What {is} it?" in prompt


def test_the_no_information_sentence_is_the_one_the_app_looks_for(answer_generator):
    # app.py hides the evidence when the answer contains this text
    assert "don't have enough information" in answer_generator.NO_INFORMATION_ANSWER.lower()


# ============================================================
# The whole app: the neighbour reaches the language model
# ============================================================

def test_the_app_sends_the_neighbouring_chunk_to_the_model(env, monkeypatch):
    import test_youtube_ingestion_app as app_env
    from test_qa_ollama_unavailable import ask

    monkeypatch.setattr(app_env, "SEGMENTS", [
        {"start": 0.0, "end": 20.0,
         "text": " Today we learn about the weather on the coast."},
        {"start": 20.0, "end": 40.0,
         "text": " Paul decides whether he likes a song on the basis of the song's tempo,"},
        {"start": 40.0, "end": 60.0,
         "text": " genre, intensity and the gender of voice."},
        {"start": 60.0, "end": 80.0,
         "text": " Cooking pasta needs salted boiling water."},
    ])

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content":
                "Paul decides on the basis of the tempo, genre, intensity and voice."}}

    env.set_upload("lecture.mp4")
    at = env.start()

    monkeypatch.setattr(sys.modules["ollama"], "chat", chat)

    at = ask(at, "How does Paul decide whether he likes a song?")

    assert not at.exception, [e.value for e in at.exception]

    prompt = calls[-1]["messages"][0]["content"]

    # the neighbour with the rest of the sentence is there, joined
    assert "song's tempo, genre, intensity and the gender of voice." in prompt
    assert calls[-1]["options"]["temperature"] == 0

    # and the answer and its evidence are shown as before
    assert at.session_state["last_answer"].startswith("Paul decides")
    assert at.session_state["last_evidence_segments"]


def test_the_app_answers_no_information_when_nothing_matches(env, monkeypatch):
    from test_qa_ollama_unavailable import ask

    env.set_upload("lecture.mp4")
    at = env.start()

    # Only Q&A calls from here on (the Video Summary also uses Ollama).
    calls = []
    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: calls.append(kwargs) or {"message": {"content": "Go ahead."}},
    )

    at = ask(at, "Which programming language does the video use?")

    assert not at.exception, [e.value for e in at.exception]
    assert calls == []
    assert at.session_state["last_answer"] == "I don't have enough information in the video."
    assert at.session_state["last_evidence_segments"] == []


from test_youtube_ingestion_app import env  # noqa: E402,F401


# ============================================================
# The model for answers (Settings in the sidebar)
# ============================================================

def test_the_answer_model_can_be_chosen(answer_generator):
    answer_generator.generate_answer("[0:00 - 0:05]\nSome text.\n\n", "Q?", model="llama3.1")

    assert answer_generator.fake_ollama.calls[0]["model"] == "llama3.1"


def test_the_default_answer_model_is_llama3_2(answer_generator):
    answer_generator.generate_answer("[0:00 - 0:05]\nSome text.\n\n", "Q?")

    assert answer_generator.fake_ollama.calls[0]["model"] == "llama3.2"


def test_a_missing_larger_model_says_how_to_install_it(answer_generator):
    def missing(**kwargs):
        error = answer_generator.fake_ollama.ResponseError("model not found")
        error.status_code = 404
        raise error

    answer_generator.fake_ollama.chat = missing

    with pytest.raises(answer_generator.LLMUnavailableError) as raised:
        answer_generator.generate_answer("[0:00 - 0:05]\nText.\n\n", "Q?", model="llama3.1")

    assert 'ollama pull llama3.1' in str(raised.value)
    assert "llama3.2" not in str(raised.value)


def test_the_app_uses_the_model_chosen_in_settings(env, monkeypatch):
    from test_qa_ollama_unavailable import ask

    env.set_upload("lecture.mp4")
    at = env.start()

    models = []
    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: models.append(kwargs["model"]) or {"message": {"content": "An answer."}},
    )

    at.selectbox(key="answer_model_label").set_value("llama3.1 8B (better, slower)")
    at.run()

    at = ask(at, "What is machine learning?")

    assert not at.exception, [e.value for e in at.exception]
    assert models and models[-1] == "llama3.1"
