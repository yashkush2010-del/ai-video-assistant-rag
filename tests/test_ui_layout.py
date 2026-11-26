"""
The page layout (UI cleanup).

  - the video is chosen in the sidebar
  - processing shows one status box instead of a stack of green boxes
  - next to the video: three sections (Ask Questions / Summary / Viva),
    one open at a time, kept open after a Jump
  - developer output is in one collapsed "Developer details" expander
  - every time on the page is shown as minutes:seconds

Same environment as the other app tests (real short video, stand-in
models).
"""

import sys

from test_youtube_ingestion_app import (  # noqa: F401
    env,
    page_text,
)


ASK = "\U0001f4ac Ask Questions"
SUMMARY = "\U0001f4cb Summary"
VIVA = "\U0001f393 Viva"


def loaded(env):
    env.set_upload("lecture.mp4")
    at = env.start()
    assert not at.exception, [e.value for e in at.exception]
    return at


def open_section(at, name):
    at.button_group(key="section").set_value(name)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def markdown(at):
    return "\n".join(str(m.value) for m in at.markdown)


# ============================================================
# before a video is chosen
# ============================================================

def test_without_a_video_the_page_says_where_to_start(env):
    at = env.start()

    assert any("sidebar" in str(i.value) for i in at.info)
    assert not any(r.key == "section" for r in at.get("button_group"))


def test_the_video_is_chosen_in_the_sidebar(env):
    at = env.start()

    sidebar_keys = [w.key for w in at.sidebar.text_input] + [
        b.key for b in at.sidebar.button
    ]

    assert "youtube_url" in sidebar_keys
    assert "process_youtube" in sidebar_keys


# ============================================================
# processing
# ============================================================

def test_processing_is_one_status_box_not_a_stack_of_green_boxes(env):
    at = loaded(env)

    text = page_text(at)

    assert "Transcription completed!" in text
    assert "Video processing completed!" in text

    # the old green success boxes are gone
    assert len(at.success) == 0


def test_reprocess_is_in_the_sidebar(env):
    at = loaded(env)

    assert "reprocess_video" in [b.key for b in at.sidebar.button]
    assert "Current video" in "\n".join(str(m.value) for m in at.sidebar.markdown)


# ============================================================
# sections
# ============================================================

def test_the_sections_and_ask_questions_first(env):
    at = loaded(env)

    section = at.button_group(key="section")

    # (the test framework reports the labels without their leading emoji)
    assert [str(o).strip() for o in section.options] == [
        "Ask Questions", "Summary", "Viva",
    ]
    assert section.value == ASK

    assert "Ask Questions About Your Video" in markdown(at)
    assert not any(b.key == "viva_generate" for b in at.button)


def test_one_section_at_a_time(env):
    at = loaded(env)

    at = open_section(at, VIVA)
    assert any(b.key == "viva_generate" for b in at.button)
    assert "Ask Questions About Your Video" not in markdown(at)

    at = open_section(at, SUMMARY)
    assert not any(b.key == "viva_generate" for b in at.button)
    assert "Ask Questions About Your Video" not in markdown(at)


def test_the_summary_exists_whichever_section_is_open(env):
    # made right after processing, so Viva can use it at once
    at = loaded(env)

    assert at.session_state["section"] == ASK
    assert at.session_state["video_summary"] is not None


def test_a_jump_keeps_the_open_section(env, monkeypatch):
    from src.rag import video_summary

    def fake_summary(chunks, embedding_model, *args, **kwargs):
        chunk = chunks[0]
        return {
            "main_topic": "Machine Learning",
            "points": [{
                "title": "Basics", "summary": "Machines learn from data.",
                "start": chunk["start"], "end": chunk["end"],
                "text": chunk["text"], "chunk_index": 0,
                "segment_range": [0, 0], "description_source": "llm",
            }],
            "used_llm": True,
        }

    monkeypatch.setattr(video_summary, "generate_video_summary", fake_summary)

    at = open_section(loaded(env), SUMMARY)

    jump = next(b for b in at.button if str(b.key).startswith("summary_jump_"))
    jump.click()
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.button_group(key="section").value == SUMMARY
    assert any(str(b.key).startswith("summary_jump_") for b in at.button)
    assert at.session_state["jump_time"] is not None


# ============================================================
# developer details and readable times
# ============================================================

def test_developer_output_is_in_one_collapsed_expander(env):
    at = loaded(env)

    developer = [e for e in at.expander if "Developer details" in str(e.label)]

    assert len(developer) == 1
    assert "Multimodal Record Preview" in markdown(at)


def test_no_time_on_the_page_is_shown_in_seconds(env, monkeypatch):
    from test_qa_ollama_unavailable import ask

    at = loaded(env)

    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: {"message": {"content": "Machine learning learns from data."}},
    )

    at = ask(at, "What is machine learning?")

    assert not at.exception, [e.value for e in at.exception]

    shown = [str(c.value) for c in at.caption] + [
        str(getattr(image, "caption", "") or "") for image in at.get("imgs")
    ]

    # there is an evidence caption to check, in the new form
    assert any(text.startswith("Video evidence: ") for text in shown), shown
    assert not any("seconds" in text for text in shown), shown


# ============================================================
# round 2: polish
# ============================================================

def llm_summary(monkeypatch, titles=("Types of Machine Learning", "AI in Healthcare")):
    from src.rag import video_summary

    def fake_summary(chunks, embedding_model, *args, **kwargs):
        chunk = chunks[0]
        return {
            "main_topic": "Machine Learning",
            "points": [{
                "title": title, "summary": "Machines learn from data.",
                "start": chunk["start"], "end": chunk["end"],
                "text": chunk["text"], "chunk_index": 0,
                "segment_range": [0, 0], "description_source": "llm",
            } for title in titles],
            "used_llm": True,
        }

    monkeypatch.setattr(video_summary, "generate_video_summary", fake_summary)


def test_clicking_the_open_section_again_keeps_it_open(env):
    at = open_section(loaded(env), VIVA)

    # the segmented control returns None when its option is clicked again
    at.button_group(key="section").set_value(None)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert any(b.key == "viva_generate" for b in at.button)


def test_example_questions_come_from_the_summary_titles(env, monkeypatch):
    llm_summary(monkeypatch)

    at = loaded(env)

    examples = [b.label for b in at.button if str(b.key).startswith("example_question_")]

    assert examples == [
        "What does the video say about types of machine learning?",
        "What does the video say about AI in healthcare?",
    ]


def test_clicking_an_example_asks_it(env, monkeypatch):
    llm_summary(monkeypatch)

    at = loaded(env)

    monkeypatch.setattr(
        sys.modules["ollama"], "chat",
        lambda **kwargs: {"message": {"content": "Machine learning learns from data."}},
    )

    at.button(key="example_question_0").click()
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["last_question"] == (
        "What does the video say about types of machine learning?"
    )
    # asked and answered (the stand-in transcript may not cover it)
    assert at.session_state["last_answer"] is not None

    # the examples make room for the answer
    assert not any(str(b.key).startswith("example_question_") for b in at.button)


def test_no_example_questions_without_a_language_model_summary(env):
    at = loaded(env)

    assert not any(str(b.key).startswith("example_question_") for b in at.button)


def test_the_question_is_in_a_form_so_enter_asks_it(env):
    at = loaded(env)

    assert any(
        "Ask Question" in str(b.label) and type(b).__name__ == "Button"
        for b in at.button
    )
    assert "FormSubmitter:ask_form" in " ".join(str(b.id) for b in at.button)


def test_viva_shows_progress_and_a_score_breakdown(env):
    from test_viva_app import answer, click, generate

    at = generate(open_section(loaded(env), VIVA))

    assert len(at.get("progress")) == 1

    total = len(at.session_state["viva"]["questions"])

    for number in range(total):
        item = at.session_state["viva"]["questions"][number]
        at = answer(at, item["answer"], number)
        at = click(at, f"viva_next_{number}")

    labels = [str(m.label) for m in at.metric]

    assert len(labels) == 3
    assert [str(m.value) for m in at.metric][0] == str(total)


def test_the_about_box_names_the_models(env):
    at = env.start()

    about = [e for e in at.sidebar.expander if "About this app" in str(e.label)]

    assert len(about) == 1
    assert "Whisper" in "\n".join(str(m.value) for m in at.sidebar.markdown)


def video_block_position(at):
    """Where (which top-level place) the video and panel are on the page."""
    for position, node in at.main.children.items():
        if any(True for _ in node.get("video")) if hasattr(node, "get") else False:
            return position
    return None


def test_the_video_stays_in_the_same_place_after_processing(env, monkeypatch):
    # The run that processes the video draws the processing box; the
    # next run does not. The video must not move up a place, or the
    # browser briefly shows the old page faded below the new one.
    llm_summary(monkeypatch)

    at = loaded(env)
    first = video_block_position(at)

    at.button(key="example_question_0").click()
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert first is not None
    assert video_block_position(at) == first
