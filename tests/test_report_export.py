"""
Downloadable reports (src/rag/report_export.py) and their buttons.
"""

from src.rag.report_export import report_file_name, summary_report, viva_report


SUMMARY = {
    "main_topic": "Machine Learning",
    "used_llm": True,
    "points": [
        {"title": "Types of <Learning>", "summary": "Supervised & unsupervised.",
         "start": 159.2, "end": 181.0, "text": "It could be either supervised learning"},
        {"title": "Uber", "summary": "Surge pricing.",
         "start": 415.0, "end": 444.0, "text": "used by global taxi giant Uber"},
    ],
}

QUESTIONS = [
    {"question": "What does supervised learning use?", "answer": "labeled data",
     "start": 215.0, "end": 219.0, "text": "Hence, supervised learning uses labeled data"},
    {"question": "What is reinforcement learning?", "answer": "reward-based learning",
     "start": 277.0, "end": 280.0, "text": "which is a reward-based learning."},
    {"question": "Complete: ...", "answer": "x", "start": 1.0, "end": 2.0, "text": "x"},
]

RESULTS = {
    0: {"verdict": "correct", "feedback": "Your answer matches what the video says."},
    1: {"verdict": "partial", "feedback": "Also mention feedback."},
}

SCORE = {"score": 1.5, "total": 3, "percent": 50}


def test_summary_report_has_every_point_with_readable_times():
    page = summary_report("machine_learning", SUMMARY)

    assert page.startswith("<!DOCTYPE html>")
    assert "Machine Learning" in page
    assert "1. Types of &lt;Learning&gt;" in page      # escaped
    assert "Supervised &amp; unsupervised." in page
    assert "2:39 – 3:01" in page
    assert "6:55 – 7:24" in page
    assert "<script" not in page


def test_viva_report_has_score_counts_answers_and_expected_answers():
    page = viva_report("machine_learning", QUESTIONS, RESULTS, SCORE,
                       {0: "Labeled data <b>", 1: "rewards"})

    assert "Score: 1.5 / 3 (50%)" in page
    assert "Correct: 1" in page and "Partly correct: 1" in page and "Incorrect: 0" in page
    assert "Labeled data &lt;b&gt;" in page
    assert "Expected answer:</span> reward-based learning" in page
    assert "Not answered" in page                      # Q3
    assert "3:35 – 3:39" in page


def test_empty_summary_still_makes_a_page():
    assert "No summary points." in summary_report("v", {"points": []})


def test_file_names_are_safe():
    assert report_file_name("machine_learning", "summary") == "machine_learning_summary.html"
    assert report_file_name("a/b:c?", "viva_results") == "a_b_c_viva_results.html"
    assert report_file_name("", "summary") == "video_summary.html"


def test_the_app_offers_the_summary_download(env, monkeypatch):
    from test_ui_layout import SUMMARY as SUMMARY_SECTION, loaded, llm_summary, open_section

    llm_summary(monkeypatch)

    at = open_section(loaded(env), SUMMARY_SECTION)

    assert any(
        "Download summary" in str(getattr(e, "proto", "")) for e in at.get("download_button")
    )


def test_the_app_offers_the_viva_download_at_the_end(env):
    from test_ui_layout import VIVA, loaded, open_section
    from test_viva_app import answer, click, generate

    at = generate(open_section(loaded(env), VIVA))

    assert not at.get("download_button")

    total = len(at.session_state["viva"]["questions"])
    for number in range(total):
        at = answer(at, at.session_state["viva"]["questions"][number]["answer"], number)
        at = click(at, f"viva_next_{number}")

    assert any(
        "Download results" in str(getattr(e, "proto", "")) for e in at.get("download_button")
    )


from test_youtube_ingestion_app import env  # noqa: E402,F401
