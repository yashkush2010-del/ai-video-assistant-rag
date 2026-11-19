"""
Q&A when Ollama is not available.

Before: generate_answer() let the Ollama client's error through, and
app.py showed a Streamlit traceback. Now:

  - Ollama not running      -> LLMUnavailableError ("not running ...")
  - llama3.2 not downloaded -> LLMUnavailableError ("ollama pull ...")
  - any OTHER error         -> raised as before (real bugs are not hidden)
  - Ollama running          -> exactly the same prompt, model and answer

and the app shows the message with st.error, shows no answer, and
does not crash.
"""

import sys
import types

import pytest

from test_youtube_ingestion_app import env, page_text  # noqa: F401


# ============================================================
# generate_answer() on its own
# ============================================================

@pytest.fixture
def answer_generator(monkeypatch):
    """src.rag.answer_generator with a controllable stand-in for ollama."""
    ollama = types.ModuleType("ollama")
    ollama.calls = []

    class ResponseError(Exception):
        def __init__(self, error, status_code=-1):
            super().__init__(error)
            self.error = error
            self.status_code = status_code

    ollama.ResponseError = ResponseError

    def chat(**kwargs):
        ollama.calls.append(kwargs)
        return {"message": {"content": "Machine learning learns from data."}}

    ollama.chat = chat

    for name, module in (
        ("ollama", ollama),
        ("whisper", types.ModuleType("whisper")),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    monkeypatch.delitem(sys.modules, "src.rag.answer_generator", raising=False)

    import src.rag.answer_generator as module

    module.fake_ollama = ollama

    yield module

    monkeypatch.delitem(sys.modules, "src.rag.answer_generator", raising=False)


def raising(error):
    def chat(**kwargs):
        raise error
    return chat


def test_a_normal_answer_is_unchanged(answer_generator):
    answer = answer_generator.generate_answer(
        "[0.0 - 5.0]\nMachine learning learns from data.\n\n",
        "What is machine learning?"
    )

    assert answer == "Machine learning learns from data."

    call = answer_generator.fake_ollama.calls[0]

    assert call["model"] == "llama3.2"
    assert call["messages"][0]["role"] == "user"
    assert "What is machine learning?" in call["messages"][0]["content"]
    assert "Machine learning learns from data." in call["messages"][0]["content"]
    assert "I don't have enough information in the video." in call["messages"][0]["content"]


@pytest.mark.parametrize("error", [
    # what the ollama client raises when Ollama is not running
    ConnectionError(
        "Failed to connect to Ollama. Please check that Ollama is "
        "downloaded, running and accessible. https://ollama.com/download"
    ),
    ConnectionRefusedError("[WinError 10061] No connection could be made"),
    TimeoutError("timed out"),
])
def test_ollama_not_running_gives_a_clear_error(answer_generator, error, monkeypatch):
    monkeypatch.setattr(answer_generator.fake_ollama, "chat", raising(error))

    with pytest.raises(answer_generator.LLMUnavailableError) as caught:
        answer_generator.generate_answer("context", "question")

    message = str(caught.value)

    assert "Ollama does not seem to be running" in message
    assert "ollama serve" in message

    # the original error is kept for debugging
    assert caught.value.__cause__ is error


def test_an_httpx_connection_error_is_handled_too(answer_generator, monkeypatch):
    # Older ollama versions let httpx's own error through.
    httpx = pytest.importorskip("httpx")

    monkeypatch.setattr(
        answer_generator.fake_ollama, "chat",
        raising(httpx.ConnectError("All connection attempts failed"))
    )

    with pytest.raises(answer_generator.LLMUnavailableError):
        answer_generator.generate_answer("context", "question")


def test_a_missing_model_tells_how_to_download_it(answer_generator, monkeypatch):
    error = answer_generator.fake_ollama.ResponseError(
        "model 'llama3.2' not found", status_code=404
    )
    monkeypatch.setattr(answer_generator.fake_ollama, "chat", raising(error))

    with pytest.raises(answer_generator.LLMUnavailableError) as caught:
        answer_generator.generate_answer("context", "question")

    assert "ollama pull llama3.2" in str(caught.value)


def test_another_ollama_response_error_is_reported(answer_generator, monkeypatch):
    error = answer_generator.fake_ollama.ResponseError("server busy", status_code=500)
    monkeypatch.setattr(answer_generator.fake_ollama, "chat", raising(error))

    with pytest.raises(answer_generator.LLMUnavailableError) as caught:
        answer_generator.generate_answer("context", "question")

    assert "server busy" in str(caught.value)


def test_a_real_bug_is_not_hidden(answer_generator, monkeypatch):
    # e.g. an unexpected reply format: not an "Ollama is down" situation
    monkeypatch.setattr(
        answer_generator.fake_ollama, "chat",
        lambda **kwargs: {"unexpected": "format"}
    )

    with pytest.raises(KeyError):
        answer_generator.generate_answer("context", "question")


def test_no_answer_is_made_up(answer_generator, monkeypatch):
    monkeypatch.setattr(
        answer_generator.fake_ollama, "chat",
        raising(ConnectionError("Failed to connect to Ollama."))
    )

    result = None

    with pytest.raises(answer_generator.LLMUnavailableError):
        result = answer_generator.generate_answer("context", "question")

    assert result is None


# ============================================================
# The app: ask a question while Ollama is down
# ============================================================

def ask(at, question):
    box = next(
        widget for widget in at.text_input
        if widget.label == "Enter your question:"
    )
    box.set_value(question)

    button = next(
        widget for widget in at.button
        if widget.label == "🔍 Ask Question"
    )
    button.click()

    at.run()

    return at


def start_with_upload(env):
    env.set_upload("lecture.mp4")

    at = env.start()

    assert "Video processing completed!" in page_text(at)

    return at


def ollama_down(monkeypatch):
    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        raising(ConnectionError("Failed to connect to Ollama."))
    )


def error_messages(at):
    return [str(element.value) for element in at.error]


def test_the_app_does_not_crash_when_ollama_is_down(env, monkeypatch):
    at = start_with_upload(env)

    ollama_down(monkeypatch)

    at = ask(at, "What is machine learning?")

    assert not at.exception, [e.value for e in at.exception]

    assert any(
        "Ollama does not seem to be running" in message
        for message in error_messages(at)
    )


def test_no_answer_is_shown_when_ollama_is_down(env, monkeypatch):
    at = start_with_upload(env)

    ollama_down(monkeypatch)

    at = ask(at, "What is machine learning?")

    assert at.session_state["last_answer"] is None
    assert at.session_state["last_evidence_segments"] == []
    assert "🧠 Answer" not in page_text(at)

    # the question stays in the box, ready to ask again
    assert at.session_state["last_question"] == "What is machine learning?"


def test_an_old_answer_is_not_left_on_screen(env, monkeypatch):
    at = start_with_upload(env)

    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: {"message": {"content": "An earlier answer."}}
    )

    at = ask(at, "First question?")

    assert at.session_state["last_answer"] == "An earlier answer."

    ollama_down(monkeypatch)

    at = ask(at, "Second question?")

    assert at.session_state["last_answer"] is None
    assert "An earlier answer." not in page_text(at)


def test_the_error_goes_away_when_ollama_is_back(env, monkeypatch):
    at = start_with_upload(env)

    ollama_down(monkeypatch)

    at = ask(at, "What is machine learning?")

    assert at.session_state["qa_error"]

    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: {"message": {"content": "Machine learning learns from data."}}
    )

    at = ask(at, "What is machine learning?")

    assert not at.exception
    assert at.session_state["qa_error"] is None
    assert at.session_state["last_answer"] == "Machine learning learns from data."
    assert error_messages(at) == []
    assert "🧠 Answer" in page_text(at)


def test_normal_qa_works_as_before(env, monkeypatch):
    at = start_with_upload(env)

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": "Machine learning learns from data."}}

    monkeypatch.setattr(sys.modules["ollama"], "chat", chat)

    at = ask(at, "What is machine learning?")

    assert not at.exception
    assert len(calls) == 1
    assert calls[0]["model"] == "llama3.2"
    assert at.session_state["last_answer"] == "Machine learning learns from data."
    assert at.session_state["qa_error"] is None
