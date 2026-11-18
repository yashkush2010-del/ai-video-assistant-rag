"""
Viva Mode inside the whole app.

Uses the same environment as the YouTube app tests (real 4-second
video, real ffmpeg / OpenCV / FAISS; Whisper, MiniLM and Ollama are
stand-ins). Ollama returns nothing here, so the questions use the
transcript fallback, exactly as they would with Ollama switched off.
"""

from test_youtube_ingestion_app import (  # noqa: F401
    FakeUpload,
    env,
    page_text,
)


def start_with_upload(env):
    env.set_upload("lecture.mp4")

    at = env.start()

    assert "Video processing completed!" in page_text(at)

    return at


def click(at, key):
    at.button(key=key).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def generate(at):
    return click(at, "viva_generate")


def answer(at, text, number=0, attempt=1):
    at.text_area(key=f"viva_answer_{attempt}_{number}").input(text)
    return click(at, f"viva_check_{number}")


# ============================================================
# the section exists and starts empty
# ============================================================

def test_viva_mode_is_offered_after_processing(env):
    at = start_with_upload(env)

    assert at.session_state["viva"] is None
    assert at.button(key="viva_generate") is not None


def test_viva_mode_does_not_change_the_rest_of_the_app(env):
    at = start_with_upload(env)

    # Summary, keyframes and Q&A are all still there.
    assert at.session_state["video_summary"] is not None
    assert at.session_state["index"] is not None
    assert any("Ask Questions" in str(m.value) for m in at.markdown)


# ============================================================
# questions
# ============================================================

def test_generating_questions_uses_the_summary_points(env):
    at = generate(start_with_upload(env))

    viva = at.session_state["viva"]
    points = at.session_state["video_summary"]["summary"]["points"]

    assert viva["questions"]
    assert len(viva["questions"]) <= len(points)

    segments = at.session_state["segments"]

    for item in viva["questions"]:

        # The section is exactly one summary point ...
        assert any(
            item["section_start"] == point["start"]
            and item["section_end"] == point["end"]
            for point in points
        )

        # ... and the timestamp is a real Whisper segment inside it.
        assert item["section_start"] <= item["start"]
        assert item["end"] <= item["section_end"]
        assert any(s["start"] == item["start"] for s in segments)
        assert any(s["end"] == item["end"] for s in segments)

    assert "Question 1 of" in page_text(at)


# ============================================================
# answering
# ============================================================

def test_an_empty_answer_asks_for_an_answer(env):
    at = generate(start_with_upload(env))

    at = click(at, "viva_check_0")

    assert "Please type an answer first." in page_text(at)
    assert at.session_state["viva_results"] == {}


def test_a_checked_answer_shows_verdict_expected_answer_and_jump(env):
    at = generate(start_with_upload(env))

    expected = at.session_state["viva"]["questions"][0]["answer"]

    at = answer(at, expected)

    result = at.session_state["viva_results"][0]

    assert result["verdict"] == "correct"
    assert "Correct." in page_text(at)
    assert "Expected answer:" in page_text(at)

    jump_buttons = [
        button for button in at.button
        if str(button.key).startswith("viva_jump_0_")
    ]

    assert len(jump_buttons) == 1


def test_the_viva_jump_uses_the_shared_seek_mechanism(env):
    at = generate(start_with_upload(env))

    item = at.session_state["viva"]["questions"][0]

    at = answer(at, item["answer"])

    counter = at.session_state["seek_counter"]

    at = click(at, f"viva_jump_0_{int(item['start'])}")

    assert at.session_state["jump_time"] == int(item["start"])
    assert at.session_state["seek_counter"] == counter + 1

    # sent once, then cleared (like every other Jump button)
    assert at.session_state["seek_request"] is None

    # the viva is still running after the jump
    assert at.session_state["viva"] is not None
    assert 0 in at.session_state["viva_results"]


# ============================================================
# final score and restart
# ============================================================

def finish(at):
    total = len(at.session_state["viva"]["questions"])

    for number in range(total):
        item = at.session_state["viva"]["questions"][number]
        at = answer(at, item["answer"], number)
        at = click(at, f"viva_next_{number}")

    return at


def test_finishing_shows_the_final_score(env):
    at = finish(generate(start_with_upload(env)))

    total = len(at.session_state["viva"]["questions"])

    assert f"Final Score: {total} / {total} (100%)" in page_text(at)


def test_restart_keeps_the_questions_and_clears_the_answers(env):
    at = finish(generate(start_with_upload(env)))

    questions = at.session_state["viva"]["questions"]

    at = click(at, "viva_restart")

    assert at.session_state["viva"]["questions"] == questions
    assert at.session_state["viva"]["attempt"] == 2
    assert at.session_state["viva_index"] == 0
    assert at.session_state["viva_results"] == {}

    # a fresh, empty answer box
    assert at.text_area(key="viva_answer_2_0").value == ""


def test_new_questions_starts_over(env):
    at = finish(generate(start_with_upload(env)))

    at = click(at, "viva_new")

    assert at.session_state["viva"] is None
    assert at.button(key="viva_generate") is not None


# ============================================================
# robustness
# ============================================================

def test_a_failure_while_generating_does_not_break_the_app(env, monkeypatch):
    import src.rag.viva as viva_module

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    at = start_with_upload(env)

    # app.py runs "from src.rag.viva import ..." on every rerun, so
    # patching the module makes the next run use the broken function.
    monkeypatch.setattr(viva_module, "generate_viva_questions", explode)

    at = generate(at)

    assert at.session_state["viva"]["error"] is True
    assert "could not be generated" in page_text(at)

    # The rest of the app is still there.
    assert any("Ask Questions" in str(m.value) for m in at.markdown)

    # And it can be tried again.
    at = click(at, "viva_retry")

    assert at.session_state["viva"] is None


def test_a_new_video_starts_a_new_viva(env):
    at = generate(start_with_upload(env))

    at = answer(at, "Some answer about labelled examples.")

    first_hash = at.session_state["video_hash"]

    assert at.session_state["viva"]["video_hash"] == first_hash
    assert at.session_state["viva_results"]

    # Upload a different video: it is processed as usual ...
    env.upload["file"] = FakeUpload(
        "other.mp4",
        env.other_video.read_bytes()
    )

    at.run()

    assert not at.exception
    assert at.session_state["video_hash"] != first_hash

    # ... and the viva of the first video is gone.
    assert at.session_state["viva"] is None
    assert at.session_state["viva_index"] == 0
    assert at.session_state["viva_results"] == {}