"""
The message under the video after a Jump.

Before: "🎯 Video opened at approximately 173 seconds."
        173 is not a duration: it is the POSITION the video jumped to
        (the Jump button said "Jump to 2:53"), printed as raw seconds.
Now:    "🎯 Video opened at 2:53."  - the same m:ss format as the Jump
        buttons, the Summary and Viva. Seeking itself is unchanged.
"""

import pytest

from test_youtube_ingestion_app import env, page_text  # noqa: F401


def start_with_upload(env):
    env.set_upload("lecture.mp4")

    at = env.start()

    assert "Video processing completed!" in page_text(at)

    return at


def jump_message(at):
    # (Streamlit shows the leading 🎯 as the message's icon, so the
    # text itself starts with "Video opened at".)
    return [
        str(element.value)
        for element in at.success
        if "Video opened at" in str(element.value)
    ]


@pytest.mark.parametrize("position, shown", [
    (173, "2:53"),        # the case seen on the real video
    (173.8, "2:53"),
    (0, "0:00"),
    (59, "0:59"),
    (387, "6:27"),
    (3725, "1:02:05"),
])
def test_the_jump_position_is_shown_as_minutes_and_seconds(env, position, shown):
    at = start_with_upload(env)

    at.session_state["jump_time"] = position
    at.run()

    assert not at.exception
    assert jump_message(at) == [f"Video opened at {shown}."]


def test_raw_seconds_are_not_shown(env):
    at = start_with_upload(env)

    at.session_state["jump_time"] = 173
    at.run()

    assert "173 seconds" not in page_text(at)


def test_the_message_matches_the_jump_button_label(env):
    from src.rag.video_summary import format_timestamp

    at = start_with_upload(env)

    # a Jump button labelled "Jump to 2:53" requests int(173.4) = 173
    at.session_state["jump_time"] = int(173.4)
    at.run()

    assert jump_message(at) == [f"Video opened at {format_timestamp(173.4)}."]


def test_no_message_without_a_jump(env):
    at = start_with_upload(env)

    assert jump_message(at) == []


def test_the_video_still_starts_at_the_jump_position(env):
    at = start_with_upload(env)

    at.session_state["jump_time"] = 173
    at.run()

    # the player itself is unchanged: started at the jump position
    assert at.session_state["jump_time"] == 173
