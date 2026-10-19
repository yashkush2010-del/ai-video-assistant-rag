"""
Q&A evidence: timestamps are DISPLAYED as m:ss, seeking is unchanged.

  heading   ⏱️ 6:27 – 6:30        (was  [387.6s - 390.6s])
  button    ▶️ Jump to 6:27       (was  Jump to 387s)

The display uses the same formatter as the Video Summary
(src.rag.video_summary.format_timestamp / format_time_range).
The seek still uses whole seconds, exactly as before: int(segment["start"]).
"""

import sys
import types
from pathlib import Path

import pytest

from src.rag.video_summary import format_time_range, format_timestamp

APP = Path(__file__).resolve().parents[1] / "app.py"


# ============================================================
# The text that is shown
# ============================================================

@pytest.mark.parametrize("seconds, expected", [
    (0, "0:00"),
    (17, "0:17"),
    (17.5, "0:17"),
    (59.9, "0:59"),
    (60, "1:00"),
    (387, "6:27"),
    (387.6, "6:27"),
    (390, "6:30"),
    (390.6, "6:30"),
    (3599.9, "59:59"),
    (3600, "1:00:00"),
    (3725.2, "1:02:05"),
    (7384, "2:03:04"),
])
def test_qa_timestamp_text(seconds, expected):
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize("start, end, expected", [
    (387.6, 390.6, "6:27 \u2013 6:30"),
    (0, 17.5, "0:00 \u2013 0:17"),
    (17.5, 60, "0:17 \u2013 1:00"),
    (3725.2, 3790.9, "1:02:05 \u2013 1:03:10"),
])
def test_qa_time_range_text(start, end, expected):
    assert format_time_range(start, end) == expected


@pytest.mark.parametrize("start", [0, 17.5, 60, 387.6, 390.6, 3725.2])
def test_the_label_always_shows_the_second_the_player_seeks_to(start):
    """app.py seeks to int(start); the label must show exactly that second."""
    label = format_timestamp(start)

    seconds = 0
    for part in label.split(":"):
        seconds = seconds * 60 + int(part)

    assert seconds == int(start)


# ============================================================
# The real Q&A section of app.py, rendered (no models are loaded)
# ============================================================

@pytest.fixture
def render_qa(monkeypatch):
    """
    Returns render(segments) -> (AppTest, scripts_sent_to_the_browser).

    app.py imports Whisper, sentence-transformers and Ollama at the top.
    They are replaced by empty stand-ins here: this test only renders
    the Q&A evidence and never runs a model.
    """
    pytest.importorskip("streamlit.testing.v1")

    from streamlit.testing.v1 import AppTest

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda *args, **kwargs: object()

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = type(
        "SentenceTransformer", (), {"__init__": lambda self, *a, **k: None}
    )

    ollama = types.ModuleType("ollama")
    ollama.chat = lambda *args, **kwargs: {"message": {"content": ""}}

    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "ollama", ollama)

    # src.rag.answer_generator binds `ollama` when it is first imported;
    # make sure it is imported against the stand-in and not kept afterwards.
    for name in ("src.rag.answer_generator",):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # What the app sends to the browser to seek the player.
    import streamlit
    import streamlit.components.v1 as components

    sent = []

    if hasattr(streamlit, "iframe"):
        monkeypatch.setattr(
            streamlit, "iframe",
            lambda source, *args, **kwargs: sent.append(source)
        )

    monkeypatch.setattr(
        components, "html",
        lambda source, *args, **kwargs: sent.append(source)
    )

    def render(segments):
        at = AppTest.from_file(str(APP), default_timeout=60)

        # The Q&A section is shown once the video has been indexed and a
        # question has been answered; nothing else is needed to show it.
        at.session_state["index"] = object()
        at.session_state["last_question"] = "What is machine learning?"
        at.session_state["last_answer"] = "Machine learning lets computers learn."
        at.session_state["last_evidence_segments"] = segments

        at.run()

        assert not at.exception, [e.value for e in at.exception]

        return at, sent

    return render


def evidence(start, end, text="Machine learning is used in healthcare."):
    return {"start": start, "end": end, "text": text, "score": 0.9}


def markdown_texts(at):
    return [element.value for element in at.markdown]


def jump_buttons(at):
    return [b for b in at.button if "Jump to" in b.label]


def test_qa_evidence_shows_minutes_and_seconds(render_qa):
    at, _ = render_qa([evidence(387.6, 390.6)])

    # the evidence headings ("###"); "## Relevant Timestamps" is the section title
    headings = [t for t in markdown_texts(at) if t.startswith("### ")]

    assert headings == ["### \u23f1\ufe0f 6:27 \u2013 6:30"]

    # the old seconds format is gone from the heading
    assert "387.6" not in headings[0]
    assert "[" not in headings[0]


def test_qa_jump_button_shows_minutes_and_seconds(render_qa):
    at, _ = render_qa([evidence(387.6, 390.6)])

    buttons = jump_buttons(at)

    assert [b.label for b in buttons] == ["\u25b6\ufe0f Jump to 6:27"]

    # the widget key is built from the whole seconds, as before
    assert buttons[0].key == "jump_387"


@pytest.mark.parametrize("start, end, heading, label, key, seconds", [
    (0.0, 3.4, "0:00 \u2013 0:03", "0:00", "jump_0", 0),
    (17.5, 21.0, "0:17 \u2013 0:21", "0:17", "jump_17", 17),
    (60.0, 75.2, "1:00 \u2013 1:15", "1:00", "jump_60", 60),
    (387.6, 390.6, "6:27 \u2013 6:30", "6:27", "jump_387", 387),
    (3725.2, 3790.9, "1:02:05 \u2013 1:03:10", "1:02:05", "jump_3725", 3725),
])
def test_qa_evidence_for_each_kind_of_timestamp(
    render_qa, start, end, heading, label, key, seconds
):
    at, _ = render_qa([evidence(start, end)])

    assert f"### \u23f1\ufe0f {heading}" in markdown_texts(at)

    buttons = jump_buttons(at)

    assert [b.label for b in buttons] == [f"\u25b6\ufe0f Jump to {label}"]
    assert buttons[0].key == key


@pytest.mark.parametrize("start, seconds", [
    (0.0, 0), (17.5, 17), (60.0, 60), (387.6, 387), (3725.2, 3725),
])
def test_qa_jump_still_seeks_with_the_same_whole_seconds(render_qa, start, seconds):
    """Only the text changed: the click still seeks to int(start) seconds."""
    at, sent = render_qa([evidence(start, start + 3.0)])

    at.button[[b.key for b in at.button].index(f"jump_{seconds}")].click()
    at.run()

    assert not at.exception

    # the player is told to seek to the same value as before ...
    assert at.session_state["jump_time"] == seconds
    assert isinstance(at.session_state["jump_time"], int)
    assert at.session_state["seek_counter"] == 1

    # ... in the same way: the one script the app sends to the browser
    assert len(sent) == 1
    assert f"var TARGET = {float(seconds)};" in sent[0]


def test_qa_everything_else_is_unchanged(render_qa):
    at, _ = render_qa([evidence(387.6, 390.6, "Machine learning is used in healthcare.")])

    shown = markdown_texts(at)

    assert "## \u23f1\ufe0f Relevant Timestamps" in shown
    assert "Machine learning is used in healthcare." in shown

    # the caption under the evidence still reports whole seconds
    captions = [c.value for c in at.caption]

    assert (
        "Video evidence starts at approximately 387 seconds "
        "and ends at 390 seconds." in captions
    )


def test_qa_with_several_evidence_segments(render_qa):
    at, _ = render_qa([evidence(10.2, 14.0), evidence(387.6, 390.6)])

    assert [b.label for b in jump_buttons(at)] == [
        "\u25b6\ufe0f Jump to 0:10",
        "\u25b6\ufe0f Jump to 6:27",
    ]