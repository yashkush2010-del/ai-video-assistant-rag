"""
Downloadable reports for the Video Summary and Viva results.

Each report is ONE self-contained HTML page (no images, no external
files, no new Python package). It opens in any browser, and the
browser's Print -> "Save as PDF" turns it into a PDF with the same
layout (the page has print styles for that).

A real PDF library was not used on purpose: the built-in PDF fonts
cannot show characters the app uses ("0:17 – 0:31", emoji), so it
would need an extra package plus a font file.
"""

import html
from datetime import datetime

from src.rag.video_summary import format_time_range


# ============================================================
# PAGE
# ============================================================

STYLE = """
body { font-family: Segoe UI, Arial, sans-serif; color: #1d1d2b;
       max-width: 820px; margin: 32px auto; padding: 0 20px; line-height: 1.5; }
h1 { font-size: 26px; margin-bottom: 4px; }
.meta { color: #6b6b80; font-size: 13px; margin-bottom: 24px; }
.card { border: 1px solid #d9d6f5; border-radius: 10px; padding: 14px 18px;
        margin: 14px 0; page-break-inside: avoid; }
.card h3 { margin: 0 0 6px 0; font-size: 17px; }
.time { color: #6d4fd8; font-weight: 600; font-size: 14px; }
.quote { color: #555569; font-size: 13px; border-left: 3px solid #d9d6f5;
         padding-left: 10px; margin-top: 8px; }
.label { color: #6b6b80; font-size: 13px; }
.score { font-size: 22px; font-weight: 700; margin: 8px 0 4px 0; }
.counts span { display: inline-block; margin-right: 18px; font-size: 14px; }
.correct { color: #1a7f37; } .partial { color: #9a6700; } .incorrect { color: #c0362c; }
.footer { color: #8a8a9c; font-size: 12px; margin-top: 30px; }
@media print { body { margin: 0; } a { color: inherit; } }
"""

VERDICT_TEXT = {
    "correct": ("✅ Correct", "correct"),
    "partial": ("🟡 Partly correct", "partial"),
    "incorrect": ("❌ Incorrect", "incorrect"),
}


def _e(text):
    return html.escape(str(text if text is not None else ""))


def _page(title, video_name, body):
    created = datetime.now().strftime("%d %b %Y, %H:%M")

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{_e(title)} - {_e(video_name)}</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        f"<h1>{_e(title)}</h1>\n"
        f"<div class=\"meta\">Video: {_e(video_name)} &middot; created {_e(created)} "
        "with AI Video Assistant</div>\n"
        f"{body}\n"
        "<div class=\"footer\">Times refer to the video. Every point is taken "
        "from the video's own transcript.</div>\n"
        "</body>\n</html>\n"
    )


# ============================================================
# VIDEO SUMMARY
# ============================================================

def summary_report(video_name, summary):
    """The Video Summary as an HTML page (str)."""
    points = (summary or {}).get("points") or []

    parts = []

    main_topic = (summary or {}).get("main_topic")

    if main_topic:
        parts.append(f"<p><b>Main topic:</b> {_e(main_topic)}</p>")

    if not points:
        parts.append("<p>No summary points.</p>")

    for number, point in enumerate(points, start=1):
        parts.append(
            "<div class=\"card\">"
            f"<h3>{number}. {_e(point.get('title'))}</h3>"
            f"<div>{_e(point.get('summary'))}</div>"
            f"<div class=\"time\">⏱ {_e(format_time_range(point['start'], point['end']))}</div>"
            f"<div class=\"quote\">{_e(point.get('text'))}</div>"
            "</div>"
        )

    return _page("Video Summary", video_name, "\n".join(parts))


# ============================================================
# VIVA RESULTS
# ============================================================

def viva_report(video_name, questions, results, score, answers=None):
    """
    The Viva results as an HTML page (str).

    questions : the viva questions (question, answer, start, end, text)
    results   : {question number: check_answer() result}
    score     : viva_score() result (score, total, percent)
    answers   : {question number: what the student typed} (optional)
    """
    answers = answers or {}

    counts = {
        verdict: sum(1 for r in results.values() if r.get("verdict") == verdict)
        for verdict in VERDICT_TEXT
    }

    parts = [
        f"<div class=\"score\">Score: {score['score']:g} / {score['total']} "
        f"({score['percent']}%)</div>",
        "<div class=\"counts\">"
        + "".join(
            f"<span class=\"{css}\">{_e(label)}: {counts[verdict]}</span>"
            for verdict, (label, css) in VERDICT_TEXT.items()
        )
        + "</div>",
    ]

    for number, item in enumerate(questions):
        result = results.get(number)

        if result is None:
            verdict_html = "<span class=\"label\">➖ Not answered</span>"
            feedback = ""
        else:
            label, css = VERDICT_TEXT.get(result.get("verdict"), ("", "label"))
            verdict_html = f"<span class=\"{css}\"><b>{_e(label)}</b></span>"
            feedback = result.get("feedback", "")

        student = answers.get(number)

        parts.append(
            "<div class=\"card\">"
            f"<h3>Q{number + 1}. {_e(item.get('question'))}</h3>"
            + (
                f"<div><span class=\"label\">Your answer:</span> {_e(student)}</div>"
                if student else ""
            )
            + f"<div>{verdict_html} {_e(feedback)}</div>"
            f"<div><span class=\"label\">Expected answer:</span> {_e(item.get('answer'))}</div>"
            f"<div class=\"time\">⏱ {_e(format_time_range(item['start'], item['end']))}</div>"
            f"<div class=\"quote\">{_e(item.get('text'))}</div>"
            "</div>"
        )

    return _page("Viva Results", video_name, "\n".join(parts))


def report_file_name(video_name, kind):
    """'machine_learning' + 'summary' -> 'machine_learning_summary.html'"""
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(video_name or "video")
    ).strip("_") or "video"

    return f"{safe}_{kind}.html"

