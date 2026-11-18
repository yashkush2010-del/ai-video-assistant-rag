"""
Viva Mode for the AI Video Assistant.

Turns a processed video into a short oral-exam ("viva") practice:

    summary points  ->  one question per point  ->  user answers
                    ->  answer checked against that part of the video

Every question carries:
    question        the question shown to the user
    answer          the expected answer
    kind            concept | example
                    (an idea the video teaches, or what the video's
                    own example shows)
    start/end       timestamps of the answer inside the section
    text            transcript at start/end (the evidence shown)
    section_start   the whole summary point the question was made from
    section_end
    section_text
    title           topic of that summary point
    source          llm | llm_answer_first | transcript

IMPORTANT GUARANTEE
-------------------
Questions are built from the points of the Video Summary
(src/rag/video_summary.py). Those points are distinct topics, in video
order, with real Whisper timestamps and sentence-complete transcript
evidence. Timestamps are only ever those of real Whisper segments
inside that point, so a question can never point to a moment that
does not exist.

QUESTIONS: three steps for every summary point
----------------------------------------------
1. The LLM writes a question AND its answer. Both are checked against
   the transcript of the point (grounded, not a yes/no question, not
   about "the transcript", no invented facts, the answer really adds
   something to the question).
2. If that is rejected: "answer first". A complete passage of the
   point is taken from the real transcript, and the LLM only writes a
   specific question that this passage answers. The answer then fits
   the question by construction; the question is checked as in 1.
3. Without a usable LLM: a plain topic question, and a complete passage
   of the transcript as its expected answer (never a sentence fragment).
Every rejection is printed in the terminal with its reason.

CHECKING AN ANSWER
------------------
The evidence is ONLY the question's own part of the video: its
expected answer and the sentences of its section. Nothing from other
parts of the video is used, so an answer about another topic cannot be
accepted because it matches some other sentence.

    1. Similarity:  MiniLM similarity with the expected answer and with
                    each sentence of the section, and the share of the
                    expected answer's key words that are mentioned.
    2. Relevance:   an answer that shares too few topic words with the
                    question and its section, and is not similar to
                    them either, is marked incorrect at once. The LLM
                    is never asked about it.
    3. The LLM examiner judges the MEANING of the remaining answers:
                    a correct general explanation counts for a concept
                    question; for an example question, the answer must
                    agree with what the video says (outside knowledge
                    alone is at most partly correct).
Without Ollama, step 1 decides, exactly as before.

This module does not import Streamlit and does not touch the
FAISS index, the chunker, the question-answering code or the summary.
"""

import logging
import re

import numpy as np

# Shared helpers of the summary module. They are reused (not copied)
# so that viva questions are grounded and parsed exactly like the
# summary wording, by code that is already tested.
from src.rag.video_summary import (
    LLM_MODEL,
    _call_llm,
    _content_words,
    _default_llm,
    _extract_field,
    _is_grounded,
    _jaccard,
    _stem,
    _stem_set,
    sentence_problem,
)


log = logging.getLogger(__name__)


# ============================================================
# SETTINGS
# ============================================================

# Maximum number of viva questions. Fewer are returned when the
# summary has fewer points.
MAX_QUESTIONS = 5

# Minimum share of the words in an LLM-written expected answer
# that must appear in the transcript it was written from.
ANSWER_GROUNDING = 0.60

# A question may use at most this many key words that are not in its
# transcript excerpt. More than that means the question assumes
# something the video does not say. Found on a real video:
# "What is Paul primarily using his machine learning knowledge for?"
# (primarily, using, knowledge are not in the excerpt; in the video,
# Paul does not use machine learning at all). Words of the question
# form itself ("video", "explain", "difference" ...) do not count.
MAX_UNSUPPORTED_QUESTION_WORDS = 2

# A question must have at least this many words.
MIN_QUESTION_WORDS = 4

MAX_QUESTION_CHARS = 200
MAX_ANSWER_CHARS = 260

# An expected answer made of transcript sentences may be a little
# longer than an LLM-written one.
MAX_PASSAGE_CHARS = 320

# Two questions are the same question if they share this
# fraction of their content words.
DUPLICATE_QUESTION_OVERLAP = 0.80

# ---- answer checking -----------------------------------------
# Meaning (cosine similarity of MiniLM embeddings) ...
CORRECT_SIMILARITY = 0.65
PARTIAL_SIMILARITY = 0.40

# ... or key words (share of the expected answer's key words
# found in the user's answer).
CORRECT_COVERAGE = 0.60
PARTIAL_COVERAGE = 0.30

# Practically the expected answer itself. Kept as correct even if the
# LLM examiner disagrees, so a flaky judgement cannot fail a student
# who gave the video's own answer.
VERBATIM_SIMILARITY = 0.90
VERBATIM_COVERAGE = 0.90

# Section sentences shorter than this are too vague to compare with
# ("Simple, right?").
MIN_REFERENCE_WORDS = 6

# Relevance gate: an answer must share at least this many topic words
# with the question and its section (or be similar to them) before the
# LLM examiner is asked. So an unrelated answer ("I like cricket") can
# never be marked correct by the LLM.
MIN_TOPIC_WORDS = 2

# Points per verdict, used for the final score.
VERDICT_SCORES = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
}

# Shown in feedback: at most this many missed key words.
MAX_MISSING_WORDS = 4

# A "missing concept" written by the LLM is at most this long, and
# this share of its words must occur in the section.
MAX_MISSING_CONCEPT_WORDS = 12
MISSING_CONCEPT_GROUNDING = 0.50

# The examiner should give the same verdict for the same answer.
JUDGE_TEMPERATURE = 0


# ============================================================
# WORDS
# ============================================================

# The vocabulary of ASKING a question, as opposed to what it asks about.
# Such words are not "invented facts" ("How does the video ILLUSTRATE
# the CONCEPT ...?", "What is the INTERPRETATION ...?"). Found on the
# real video: llama3.2's questions were rejected for "concept,
# illustrate, scenario, process, interpretation, relate, reveal".
#
# Instead of listing every word form, these are word STARTS, grouped by
# what they do in a question; "illustrat" covers illustrate,
# illustrates, illustrated and illustration. A new word of the same
# kind needs no change; a family is only added for a new kind of word.
# "use" / "using" are deliberately NOT here: "What is Paul USING machine
# learning for?" is exactly the kind of invented claim to catch.
QUESTION_VOCABULARY = {
    # showing and telling
    "telling": (
        "explain", "describ", "illustrat", "demonstrat", "show",
        "reveal", "present", "highlight", "mention", "discuss", "tell",
        "suggest", "indicat", "represent", "say", "said", "speak",
        "video", "accord",
    ),
    # thinking about ideas
    "thinking": (
        "concept", "idea", "notion", "interpret", "understand", "mean",
        "defin", "signific", "importan", "relat", "connect", "compar",
        "contrast", "differ", "distinguish", "similar", "reason",
        "purpose", "role", "relevan",
    ),
    # structure of a topic
    "structure": (
        "scenario", "situation", "proces", "step", "method", "approach",
        "way", "type", "kind", "categor", "aspect", "factor", "result",
        "outcome", "effect", "example", "case", "part", "main", "key",
        "specific", "particular", "general", "overall", "bas", "work",
        "help", "two", "three",
    ),
}

_QUESTION_VOCABULARY_STARTS = tuple(
    start
    for starts in QUESTION_VOCABULARY.values()
    for start in starts
)


def _is_question_vocabulary(stem):
    return stem.startswith(_QUESTION_VOCABULARY_STARTS)


# A question must still be ABOUT the clip: at least this share of its
# remaining (topic) words must be in the clip.
QUESTION_TOPIC_GROUNDING = 0.50

# Words that are never a key point of an answer, so feedback does not
# ask for them. Found on the real video: "Key points to mention: saw"
# for "So we saw supervised learning where ...".
FILLER_WORDS = {
    "video", "explain", "explains", "explained",
    "saw", "see", "seen", "said", "says", "say", "suppose",
    "talk", "talks", "talked", "basically", "actually", "called",
    "hence", "therefore", "thus",
    # words of speaking flow, not of content
    "first", "quickly", "now", "today", "next", "then", "also",
    "look", "going", "start", "begin", "understand", "understanding",
    "great", "thanks", "simple", "okay",
}

_FILLER_STEMS = {_stem(word) for word in FILLER_WORDS}

# British / American spelling, so "labelled" and "labeled" are the same
# key word (the real video says "labeled", students often write
# "labelled").
_SPELLING = {
    "labelled": "labeled", "labelling": "labeling",
    "unlabelled": "unlabeled", "modelling": "modeling",
    "modelled": "modeled", "travelling": "traveling",
    "colour": "color", "colours": "colors", "behaviour": "behavior",
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "optimise": "optimize", "recognise": "recognize",
    "organise": "organize", "categorise": "categorize",
    "categorised": "categorized", "summarise": "summarize",
}

_SPELLING_PATTERN = re.compile(
    r"\b(" + "|".join(_SPELLING) + r")\b",
    flags=re.IGNORECASE
)

_POSSESSIVE = re.compile(r"(?<=[A-Za-z])['’]s\b")


def _normalise(text):
    """Same spelling and no possessive 's, so words can be compared."""
    text = _POSSESSIVE.sub("", str(text or ""))

    return _SPELLING_PATTERN.sub(
        lambda match: _SPELLING[match.group(0).lower()],
        text
    )


def _key_words(text):
    """Stemmed content words, without filler words."""
    return _stem_set(_normalise(text)) - _FILLER_STEMS


# Words that open a subordinate clause. A sentence starting with one is
# complete when its main clause follows after a comma ("While looking
# at the data, you will clearly see that there are two clusters.";
# "When you feed this data set to the machine, the machine identifies
# ..."), and a FRAGMENT when it has no main clause ("While the other
# cluster is of the players who scored less runs but took many
# wickets."). All three are real sentences of the test video.
_SUBORDINATE_STARTERS = {"while", "whereas", "when", "where"}

# A fragment starting like this continues the sentence before it.
# Found on the real video: Whisper wrote "The one cluster are the
# players who scored high runs and took less wickets." and "While the
# other cluster is ..." as two sentences; together they are one thought.
_CONTINUATION_STARTERS = {"while", "whereas"}


def _first_word(sentence):
    return str(sentence).strip().split(" ", 1)[0].lower().strip(",.;:")


def _is_subordinate_fragment(sentence):
    return (
        _first_word(sentence) in _SUBORDINATE_STARTERS
        and "," not in str(sentence)
    )


def _split_sentences(text):
    """
    Sentences of real transcript text. A continuation ("While the
    other ...") is joined to the sentence before it instead of being
    a sentence of its own.
    """
    sentences = []

    for piece in re.split(r"(?<=[.!?])\s+", str(text or "").strip()):
        piece = piece.strip()

        if not piece:
            continue

        if (
            sentences
            and _first_word(piece) in _CONTINUATION_STARTERS
            and _is_subordinate_fragment(piece)
        ):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)

    return sentences


# ---- transition and low-information sentences -------------------
#
# Found on the real video: "Let's first quickly understand supervised
# learning." was used as an expected answer. It only announces a topic.

_TRANSITION_START = re.compile(
    r"^(?:so |and |now |okay |ok |well |alright |right )?"
    r"(?:let's|let us|let me|we will|we'll|we are going to|"
    r"we're going to|today we|in this video|before we|next,? |"
    r"first,? |to begin|to start)",
    flags=re.IGNORECASE
)

# A sentence that is an answer must carry at least this many content
# words (after filler words): "So that's Paul." carries one.
MIN_INFORMATIVE_WORDS = 3


# Calls to action are not content either. Found on the real video:
# "Comment below some interesting everyday examples around you ..."
_CALL_TO_ACTION = re.compile(
    r"\b(comment below|in the comments|subscribe|hit the bell|"
    r"like this video|let us know|let me know|click the link|"
    r"link in the description|see you (?:in the )?next)",
    flags=re.IGNORECASE
)


def _is_transition(sentence):
    sentence = str(sentence).strip().replace("\u2019", "'")

    return bool(
        _TRANSITION_START.match(sentence)
        or _CALL_TO_ACTION.search(sentence)
    )


def _is_informative(sentence):
    """A complete, real sentence that says something on its own."""
    return (
        not _is_fragment(sentence)
        and not _is_transition(sentence)
        and not _is_meta(sentence)
        and len(_key_words(sentence)) >= MIN_INFORMATIVE_WORDS
    )


# ============================================================
# CHECKS FOR QUESTIONS AND EXPECTED ANSWERS
# ============================================================

# Wording that talks ABOUT the transcript instead of answering from
# it. Found on a real video, where llama3.2 answered:
#   "He doesn't explicitly mention what machine learning is used for,
#    but it is stated that he either likes or dislikes new songs."
# Most of those words are in the transcript, so the grounding check
# alone lets it through, and then correct user answers are marked
# against it ("Key points to mention: explicitly, mention, ...").
_META_PATTERN = re.compile(
    r"\b("
    r"transcript|excerpt|passage"
    r"|not (?:explicitly |directly |clearly )?"
    r"(?:mentioned|stated|specified|said|explained|discussed|given|provided)"
    r"|(?:doesn't|does not|didn't|did not|don't|do not)"
    r" (?:explicitly |directly |clearly )?"
    r"(?:mention|state|say|specify|explain|discuss)"
    r"|it is stated|it is mentioned|no information"
    r")\b",
    flags=re.IGNORECASE
)


def _is_meta(text):
    """True if the text talks about the transcript, not about the topic."""
    return bool(_META_PATTERN.search(str(text).replace("’", "'")))


def _answer_adds_nothing(answer, question):
    """
    True if every key word of the answer is already in the question.

    Found on a real video:
        Q: "What is Paul's primary reason for liking or disliking new songs?"
        A: "He likes them or dislikes them."
    The answer does not answer the question, and any user answer that
    says "likes" or "dislikes" would be marked correct.
    """
    return not (_key_words(answer) - _key_words(question))


def _question_topic_words(question):
    """The words of a question that say WHAT it asks about."""
    return {
        word
        for word in _key_words(question)
        if not _is_question_vocabulary(word)
    }


def _unsupported_words(question, source_text):
    """
    Topic words of the question that its transcript excerpt does not
    contain. A possessive 's is ignored, so "Paul's" matches "Paul";
    question vocabulary ("illustrate", "concept") is ignored too.
    """
    return _question_topic_words(question) - _key_words(source_text)


_SPOKEN_CONJUNCTIONS = {"and", "but", "or", "so"}


def _is_fragment(sentence):
    """True if the text is not a complete sentence on its own."""
    sentence = str(sentence or "").strip()

    if not sentence:
        return True

    # "While the other cluster is ..." (see _SUBORDINATE_STARTERS)
    if _is_subordinate_fragment(sentence):
        return True

    # The summary's own completeness rules (lowercase start, "Which ...",
    # "... of the."), without its length limit.
    problem = sentence_problem(sentence, max_chars=10_000)

    # Speech often starts a complete sentence with "and / but / or / so".
    # Found on the real video: "Or we can say that it works on the
    # principle of feedback." is the answer to the feedback question.
    if (
        problem == "fragment_start"
        and _first_word(sentence) in _SPOKEN_CONJUNCTIONS
    ):
        rest = sentence.split(" ", 1)[1] if " " in sentence else ""
        rest = rest[:1].upper() + rest[1:]

        return sentence_problem(rest, max_chars=10_000) is not None

    return problem is not None


# Yes/no questions are not viva questions, and they cannot be checked:
# "yes" and "no" are filler words, so the answer "Yes" would count as
# no answer at all. Found on a real video:
#   "Does Paul like listening to new songs?"
_YES_NO_START = re.compile(
    r"^(?:does|do|did|is|are|was|were|can|could|will|would|should"
    r"|has|have|had|am|may|might|shall|must)\b",
    flags=re.IGNORECASE
)


def _is_yes_no_question(question):
    return bool(_YES_NO_START.match(str(question).strip()))


def _clean_question(value):
    """First question of the model output, or None if it is not one."""
    if not value:
        return None

    value = " ".join(value.split())

    # Keep only the first question if the model wrote several.
    if "?" in value:
        value = value[:value.index("?") + 1]

    if not value.endswith("?"):
        return None

    if _is_yes_no_question(value):
        return None

    if len(value.split()) < MIN_QUESTION_WORDS:
        return None

    if len(value) > MAX_QUESTION_CHARS:
        return None

    return value[0].upper() + value[1:]


def _clean_answer(value):
    """At most two sentences of the model output, spacing tidied."""
    if not value:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", value.strip())

    answer = " ".join(" ".join(sentences[:2]).split())

    if not answer or len(answer) > MAX_ANSWER_CHARS:
        return None

    return answer


def _question_problem(question, text):
    """Why a question cannot be used, or None if it can."""
    if not question:
        return "no usable question (missing, yes/no or badly formatted)"

    if _is_meta(question):
        return "the question talks about the transcript"

    topic = _question_topic_words(question)

    if not topic:
        return "the question has no specific topic"

    unsupported = _unsupported_words(question, text)

    if len(unsupported) > MAX_UNSUPPORTED_QUESTION_WORDS:
        return (
            "the question assumes words the clip does not contain: "
            + ", ".join(sorted(unsupported))
        )

    supported = len(topic) - len(unsupported)

    if supported == 0 or supported / len(topic) < QUESTION_TOPIC_GROUNDING:
        return "the question is not about this clip"

    return None


def _answer_problem(answer, question, text):
    """
    Why an expected answer cannot be used, or None if it can.
    This is the check that the answer really answers its question.
    """
    if not answer:
        return "no usable answer"

    if _is_meta(answer):
        return "the answer talks about the transcript"

    if _is_fragment(answer):
        return "the answer is not a complete sentence"

    if _is_transition(answer):
        return "the answer only announces a topic"

    if _answer_adds_nothing(answer, question):
        return "the answer only repeats the question"

    if not _is_grounded(answer, text, ANSWER_GROUNDING):
        return "the answer is not supported by the clip"

    return None


# ============================================================
# PASSAGES OF THE TRANSCRIPT  (answer first, and fallback)
# ============================================================

# Words that show a passage is the video's own example.
_EXAMPLE_MARKERS = re.compile(
    r"\b(suppose|let's say|let us say|for example|for instance|"
    r"imagine|say you|example)\b",
    flags=re.IGNORECASE
)


def _passage(text):
    """
    The informative sentences of a section, in order, up to a readable
    length. Real transcript only: no fragment, no transition sentence
    ("Let's first quickly understand ..."), no filler ("So that's Paul.").
    """
    sentences = [
        sentence
        for sentence in _split_sentences(text)
        if _is_informative(sentence)
    ]

    passage = ""

    for sentence in sentences:
        longer = f"{passage} {sentence}".strip()

        if passage and len(longer) > MAX_PASSAGE_CHARS:
            break

        passage = longer

    return passage


# A topic sentence must add at least this many key words to the
# question ("So today we will learn about the basics of machine
# learning." adds only "today": not an answer).
MIN_NEW_ANSWER_WORDS = 2


def _topic_sentence(text, question):
    """
    The sentence of the section that is about the question's topic and
    says the most about it. Found on the real video: for "What does the
    video explain about supervised learning?" the section contains
    "Hence, supervised learning uses labeled data to train the model."
    - a much better expected answer than the example's first sentence.
    """
    topic = _question_topic_words(question)

    if not topic:
        return None

    best = None
    best_score = (0, 0, 0)

    for sentence in _split_sentences(text):

        if not _is_informative(sentence):
            continue

        words = _key_words(sentence)
        shared = len(words & topic)
        new = len(words - _key_words(question))

        if shared == 0 or new < MIN_NEW_ANSWER_WORDS:
            continue

        # The plain question asks what the video EXPLAINS: a sentence
        # that states the idea ("it works on the principle of feedback")
        # beats a sentence of an example ("let's say you provide the
        # system with an image of a dog"), found on the real video.
        states_idea = 0 if _EXAMPLE_MARKERS.search(sentence) else 1

        score = (shared, states_idea, new)

        if score > best_score:
            best, best_score = sentence, score

    return best


def _most_informative_sentence(text, question):
    """
    When no sentence names the question's topic: the real sentence of
    the section that carries the most new information.
    """
    best = None
    best_score = (0, MIN_NEW_ANSWER_WORDS - 1)

    for sentence in _split_sentences(text):

        if not _is_informative(sentence):
            continue

        new = len(_key_words(sentence) - _key_words(question))
        states_idea = 0 if _EXAMPLE_MARKERS.search(sentence) else 1

        if new < MIN_NEW_ANSWER_WORDS:
            continue

        score = (states_idea, new)

        if score > best_score:
            best, best_score = sentence, score

    return best


def _passage_kind(passage):
    return "example" if _EXAMPLE_MARKERS.search(passage) else "concept"


def _fallback_question(point):
    """A plain question about the topic of a point."""
    topic = _question_topic(point.get("title"))

    return f"What does the video explain about {topic}?"


# Title words that read badly inside a question. Found on a real video:
# "What does the video explain about Supervised Learning Explained?"
_TITLE_FILLER = {
    "explained", "explanation", "explaining", "overview",
    "introduction", "intro",
}


def _question_topic(title):
    """
    "Supervised Learning Explained" -> "supervised learning"
    Acronyms (RAG, ML, AI) keep their capitals.
    """
    words = [
        word
        for word in str(title or "").split()
        if word.lower().strip(".,:;") not in _TITLE_FILLER
    ]

    # Drop a dangling "to" / "of" left by "Introduction to ...".
    while words and words[0].lower() in ("to", "of"):
        words.pop(0)

    if not words:
        return "this part"

    return " ".join(
        word if word.isupper() and len(word) > 1 else word.lower()
        for word in words
    )


# ============================================================
# LLM WORDING (checked against the transcript)
# ============================================================

_KIND_INSTRUCTIONS = {
    "concept": (
        "Ask about the IDEA this part teaches: what something is, "
        "how it works or why it matters."
    ),
    "example": (
        "Ask about the EXAMPLE this part of the video uses and what it "
        "shows: what or who is in it, what happens, what it demonstrates."
    ),
}

QUESTION_PROMPT = """You are an examiner preparing viva (oral exam) questions about a video, using ONLY its transcript.
Do not use outside knowledge and do not add facts that are not in the excerpt.

Transcript excerpt:
__TEXT__

Write ONE open question (starting with What, How, Why, Which or Who; never a yes/no question) that a student should be able to answer after watching this part of the video, and its correct answer.
__KIND__
Use specific words (for example "How does the house price example show what regression predicts?", not "What does the video explain about houses?").
Ask only about something the excerpt states clearly.
Do not assume anything the excerpt does not say (for example, do not say that a person uses or does something unless the excerpt says so).
The answer must state that fact directly, using only the excerpt.
Never talk about the excerpt itself (do not write things like "it is not mentioned" or "the excerpt says").

Reply in exactly this format and nothing else:
QUESTION: <one question that ends with a question mark>
ANSWER: <one or two short sentences>
"""

# Step 2, "answer first": the answer is real transcript, and the LLM
# only writes a question that this transcript answers. A much easier
# task for a small model than writing both.
ANSWER_FIRST_PROMPT = """You are an examiner preparing viva (oral exam) questions about a video.
Below is a passage from the video's transcript. It is the correct answer.

Passage:
__PASSAGE__

Write ONE specific open question (starting with What, How, Why, Which or Who; never a yes/no question) that this passage answers completely.
The question must be answerable from the passage alone. Do not add facts that are not in the passage.
Never mention the passage or the transcript in the question.

Reply in exactly this format and nothing else:
QUESTION: <one question that ends with a question mark>
"""


# ============================================================
# EXACT TIMESTAMP OF THE ANSWER
# ============================================================
#
# A summary point is a whole section (often 15-20 seconds), and the
# part a question is about can start several seconds into it. Found
# on a real video: the section starts at 0:17 with "It's also about
# understanding and reasoning ...", but Paul and his songs come only
# a few seconds later, so "Jump to 0:17" landed before the answer.
#
# Same idea as find_answer_segments() in the Q&A: inside the section,
# the Whisper segment most similar to the expected answer is found,
# and the question points there. Start, end and text are still those
# of real Whisper segments; nothing is invented.

def _segments_in_section(segments, start, end):
    """The non-empty Whisper segments that lie inside [start, end]."""
    inside = []

    for segment in segments:
        try:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if not str(segment.get("text", "")).strip():
            continue

        if (
            segment_start >= float(start) - 0.01
            and segment_end <= float(end) + 0.01
        ):
            inside.append(segment)

    inside.sort(key=lambda segment: float(segment["start"]))

    return inside


def _ends_sentence(text):
    return str(text).strip().endswith((".", "!", "?"))


def _comparable(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _locate_text(answer, texts):
    """
    (first, last) segment numbers whose joined text contains the answer
    word for word, or None. Used when the expected answer IS transcript:
    found on the real video, the cricket answer starts at 4:08 ("The one
    cluster ...") but similarity pointed at its second half (4:13).
    """
    wanted = _comparable(answer)

    if not wanted:
        return None

    for first in range(len(texts)):

        first_text = _comparable(texts[first])
        joined = ""

        for last in range(first, len(texts)):
            joined = f"{joined} {_comparable(texts[last])}".strip()

            position = joined.find(wanted)

            if position >= 0:
                # The answer must begin inside the first segment;
                # otherwise a later first segment is the right one.
                if position < len(first_text):
                    return first, last
                break

            if len(joined) > len(wanted) + len(first_text):
                break

    return None


def _focus_on_answer(item, segments, embedding_model):
    """
    Move start, end and text of a question onto the Whisper segment(s)
    of its section that hold the expected answer. Changes nothing when
    the section has fewer than two segments or the search fails.
    """
    inside = _segments_in_section(
        segments,
        item["section_start"],
        item["section_end"]
    )

    if len(inside) < 2:
        return

    texts = [
        str(segment["text"]).strip()
        for segment in inside
    ]

    # An answer taken from the transcript: its exact place.
    located = _locate_text(item["answer"], texts)

    if located is not None:
        first, last = located

        item["start"] = float(inside[first]["start"])
        item["end"] = float(inside[last]["end"])
        item["text"] = " ".join(texts[first:last + 1])

        return

    try:
        vectors = np.asarray(
            embedding_model.encode(
                [item["answer"]] + texts,
                normalize_embeddings=True
            ),
            dtype="float32"
        )
    except Exception:
        return

    similarities = vectors[1:] @ vectors[0]

    best = int(np.argmax(similarities))

    first = best
    last = best

    # A Whisper segment can be half a sentence: include the segment
    # before (if this one starts lowercase) and after (if this one
    # does not end the sentence), so the evidence reads as a sentence.
    if first > 0 and texts[first][:1].islower():
        first -= 1

    if last + 1 < len(inside) and not _ends_sentence(texts[last]):
        last += 1

    item["start"] = float(inside[first]["start"])
    item["end"] = float(inside[last]["end"])
    item["text"] = " ".join(texts[first:last + 1])


# ============================================================
# ONE QUESTION FOR ONE SUMMARY POINT
# ============================================================

def _question_for_point(point, number, kind, llm, state):
    """
    The three steps (see the module docstring).
    Returns (question, answer, kind, source, whole_section) or None.
    """
    text = str(point.get("text", "")).strip()
    title = point.get("title", "")

    # ---- 1) the LLM writes the question and the answer -----------
    raw = _call_llm(
        llm,
        QUESTION_PROMPT
        .replace("__TEXT__", text)
        .replace("__KIND__", _KIND_INSTRUCTIONS[kind]),
        state
    )

    if raw:
        question = _clean_question(_extract_field(raw, "question"))
        answer = _clean_answer(_extract_field(raw, "answer"))

        # Both parts must be supported by the transcript, otherwise
        # neither is used: a good question with an invented answer
        # would mark correct answers as wrong.
        problem = (
            _question_problem(question, text)
            or _answer_problem(answer, question, text)
        )

        if problem is None:
            return question, answer, kind, "llm", False

        log.warning(
            "Viva question %d (%s): LLM question not used - %s",
            number, title, problem
        )

    # ---- 2) answer first: real transcript, LLM writes the question --
    passage = _passage(text)

    if not passage:
        return None

    if raw is not None:
        raw_question = _call_llm(
            llm,
            ANSWER_FIRST_PROMPT.replace("__PASSAGE__", passage),
            state
        )

        if raw_question:
            question = _clean_question(
                _extract_field(raw_question, "question")
            )

            problem = (
                _question_problem(question, passage)
                or _answer_problem(passage, question, text)
            )

            if problem is None:
                return (
                    question, passage, _passage_kind(passage),
                    "llm_answer_first", True
                )

            log.warning(
                "Viva question %d (%s): answer-first question not used - %s",
                number, title, problem
            )

    # ---- 3) the transcript itself ------------------------------------
    if not raw:
        log.warning(
            "Viva question %d (%s): language model not available or "
            "gave no reply, using the simple question form",
            number, title
        )

    question = _fallback_question(point)

    # Only real Whisper transcript, never the summary's description
    # (on the real video that was an LLM sentence that was not true:
    # "Paul explains the importance of understanding and reasoning").
    # 1. the section's own informative sentence about the topic,
    # 2. else its most informative sentence,
    # 3. else its informative passage.
    for sentence in (
        _topic_sentence(text, question),
        _most_informative_sentence(text, question),
    ):
        if sentence and _answer_problem(sentence, question, text) is None:
            return question, sentence, "concept", "transcript", False

    if _answer_problem(passage, question, text) is None:
        return question, passage, "concept", "transcript", True

    return None


# ============================================================
# PUBLIC FUNCTION: QUESTIONS
# ============================================================

def generate_viva_questions(
    points,
    llm=None,
    max_questions=MAX_QUESTIONS,
    segments=None,
    embedding_model=None
):
    """
    Build viva questions from the Video Summary points.

    Parameters
    ----------
    points : list
        The points of generate_video_summary()
        (dicts with title, summary, start, end, text).
    llm : callable, optional
        prompt -> text. Defaults to the local Ollama model.
    max_questions : int
        Upper limit for the number of questions.
    segments : list, optional
        The real Whisper segments of the video. Together with
        embedding_model, used to point each question at the exact
        part of its section that holds the expected answer.
    embedding_model : optional
        The already loaded SentenceTransformer.

    Returns
    -------
    list
        [
          {
            "question", "answer",
            "kind",                   # concept | example
            "start", "end", "text",   # the Whisper segment(s) inside
                                      # the summary point that hold
                                      # the answer (or the whole point)
            "section_start",          # the whole summary point
            "section_end",
            "section_text",
            "title",
            "source"                  # llm | llm_answer_first
                                      # | transcript
          }, ...                      # in video order
        ]
    """
    if not points:
        return []

    llm = llm or _default_llm
    state = {"llm_ok": True}

    questions = []

    for point in points:

        if len(questions) >= max_questions:
            break

        text = str(point.get("text", "")).strip()

        if not text:
            continue

        number = len(questions) + 1

        # Alternate, so the viva tests both understanding of the ideas
        # and the specific examples of the video.
        kind = "concept" if number % 2 == 1 else "example"

        made = _question_for_point(point, number, kind, llm, state)

        if made is None:
            log.warning(
                "Viva: no question for %s (no complete sentence in it)",
                point.get("title", "")
            )
            continue

        question, answer, kind, source, whole_section = made

        # Skip a question that repeats an earlier one.
        words = _stem_set(question)

        if any(
            _jaccard(words, _stem_set(kept["question"]))
            >= DUPLICATE_QUESTION_OVERLAP
            for kept in questions
        ):
            log.warning(
                "Viva question %d (%s): skipped, it repeats an earlier question",
                number, point.get("title", "")
            )
            continue

        item = {
            "question": question,
            "answer": answer,
            "kind": kind,
            "start": point["start"],
            "end": point["end"],
            "text": text,
            "title": point.get("title", ""),
            "source": source,
            "section_start": point["start"],
            "section_end": point["end"],
            "section_text": text,
        }

        # An answer made of the whole passage has no narrower place.
        if (
            segments
            and embedding_model is not None
            and not whole_section
        ):
            _focus_on_answer(item, segments, embedding_model)

        questions.append(item)

    # Present the questions in the order they appear in the video.
    questions.sort(key=lambda item: float(item["start"]))

    return questions


# ============================================================
# ANSWER CHECKING HELPERS
# ============================================================

def _section_references(item):
    """
    The evidence an answer is compared with: the expected answer and
    the sentences of the question's OWN section. Nothing else.
    """
    references = [item["answer"]]

    section = item.get("section_text") or item.get("text", "")

    for sentence in _split_sentences(section):
        if (
            len(sentence.split()) >= MIN_REFERENCE_WORDS
            and sentence not in references
        ):
            references.append(sentence)

    return references


def _similarities(user_answer, references, embedding_model):
    """Cosine similarity of the answer with every reference."""
    vectors = np.asarray(
        embedding_model.encode(
            [user_answer] + references,
            normalize_embeddings=True
        ),
        dtype="float32"
    )

    return [float(value) for value in vectors[1:] @ vectors[0]]


def _key_stems(expected_answer, question):
    """
    The key words of an expected answer: its content words that the
    question does not already contain. "What does supervised learning
    train a model on?" -> "labelled examples" is what has to be said;
    repeating "supervised learning" is not required.
    """
    expected = _key_words(expected_answer)

    key = expected - _key_words(question)

    return key or expected


def _coverage(user_answer, expected_answer, question):
    """Share of the expected answer's key words in the user's answer."""
    key = _key_stems(expected_answer, question)

    if not key:
        return 0.0

    return len(key & _key_words(user_answer)) / len(key)


def _missing_words(user_answer, expected_answer, question):
    """Key words of the expected answer that the user did not mention."""
    key = _key_stems(expected_answer, question)
    given = _key_words(user_answer)

    missing = []

    for word in _content_words(_normalise(expected_answer)):
        stem = _stem(word)

        if stem not in key or stem in given or word in missing:
            continue

        missing.append(word)

    return missing[:MAX_MISSING_WORDS]


def _only_repeats_question(user_answer, question):
    """True if every key word of the answer is already in the question."""
    words = _stem_set(_normalise(user_answer))

    return bool(words) and words <= _stem_set(_normalise(question))


def _topic_words(user_answer, item):
    """Topic words the answer shares with its question and section."""
    topic = (
        _key_words(item["question"])
        | _key_words(item["answer"])
        | _key_words(item.get("section_text") or item.get("text", ""))
    )

    return _key_words(user_answer) & topic


# ============================================================
# THE LLM EXAMINER
# ============================================================

_JUDGE_RULES = {
    "concept": (
        "This question tests understanding of an IDEA. A correct general "
        "explanation of the idea is CORRECT, even if the video explains it "
        "differently or only with an example."
    ),
    "example": (
        "This question tests what THIS VIDEO says. The answer must agree "
        "with the video's own content shown above. A general explanation "
        "from outside knowledge that does not give what the video says is "
        "at most PARTIAL."
    ),
}

JUDGE_PROMPT = """You are a fair viva (oral exam) examiner.
Decide whether the student's answer answers the question correctly. Judge the MEANING of the answer, not whether it uses the same words.

Question:
__QUESTION__

What the video says in this part:
__TEXT__

One correct answer (other wordings are just as good):
__EXPECTED__

Student's answer:
__ANSWER__

Verdicts:
CORRECT - the answer has the right meaning, even in completely different words, with other details, or with more detail than the video.
PARTIAL - the answer is relevant and factually correct, but incomplete or too vague.
INCORRECT - the answer is wrong, contradicts the video (for example says the opposite), or is unrelated to the question.

__RULE__

Reply in exactly this format and nothing else:
VERDICT: <CORRECT or PARTIAL or INCORRECT>
TYPE: <CONCEPT or EXAMPLE or INCOMPLETE or WRONG>
CONTRADICTS: <YES if the answer says something the video contradicts, otherwise NO>
MISSING: <the most important idea the answer is missing, in a few words, or NONE>
"""


def _judge_llm(prompt):
    # Imported here so this module can be imported and tested
    # without Ollama installed or running.
    import ollama

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": JUDGE_TEMPERATURE}
    )

    return response["message"]["content"]


def _ask_examiner(llm, user_answer, item):
    """
    The LLM's judgement: (verdict, type, missing) or None when the LLM
    is unavailable or its reply cannot be read.
    """
    kind = item.get("kind", "concept")

    raw = _call_llm(
        llm,
        JUDGE_PROMPT
        .replace("__QUESTION__", item["question"])
        .replace("__TEXT__", item.get("section_text") or item["text"])
        .replace("__EXPECTED__", item["answer"])
        .replace("__ANSWER__", user_answer)
        .replace("__RULE__", _JUDGE_RULES.get(kind, _JUDGE_RULES["concept"])),
        {"llm_ok": True}
    )

    if not raw:
        return None

    verdict = (_extract_field(raw, "verdict") or "").upper()
    answer_type = (_extract_field(raw, "type") or "").upper()
    contradicts = (_extract_field(raw, "contradicts") or "").upper()
    missing = _extract_field(raw, "missing")

    # "PARTIALLY CORRECT" must not be read as CORRECT.
    if "INCORRECT" in verdict:
        verdict = "incorrect"
    elif "PARTIAL" in verdict:
        verdict = "partial"
    elif "CORRECT" in verdict:
        verdict = "correct"
    else:
        return None

    return verdict, answer_type, contradicts.startswith("YES"), missing


# ---- contradictions ---------------------------------------------
#
# MiniLM similarity does not see meaning: "supervised learning uses
# UNLABELED data" is almost as similar to the video as the true
# sentence. Before similarity is allowed to soften a verdict, these
# plain rules look for an opposite meaning.

# Opposite words. Only a conflict when the answer uses one word, the
# section uses its opposite, the section never uses the answer's word
# and the answer never uses the opposite (a correct answer may name
# both sides: "many runs and few wickets").
OPPOSITES = [
    ("labeled", "unlabeled"), ("supervised", "unsupervised"),
    ("more", "less"), ("more", "fewer"), ("higher", "lower"),
    ("high", "low"), ("many", "few"), ("increase", "decrease"),
    ("increases", "decreases"), ("likes", "dislikes"),
    ("like", "dislike"), ("good", "bad"), ("correct", "incorrect"),
    ("true", "false"), ("possible", "impossible"),
    ("always", "never"), ("positive", "negative"),
    ("maximize", "minimize"), ("maximized", "minimized"),
    ("accept", "reject"), ("reward", "punishment"),
    ("with labels", "without labels"),
]

_NEGATION = re.compile(
    r"\b(not|never|no|none|nothing|neither|nor|cannot|can't|won't|"
    r"isn't|aren't|doesn't|don't|didn't|wasn't|weren't|"
    r"shouldn't|wouldn't|couldn't)\b",
    flags=re.IGNORECASE
)


def _has_phrase(text, phrase):
    return re.search(
        rf"\b{re.escape(phrase)}\b",
        _normalise(text).lower()
    ) is not None


def _contradiction(user_answer, item):
    """
    A reason why the answer may say the opposite of the video, or None.
    """
    section = (item.get("section_text") or item.get("text", "")) + " " + item["answer"]

    for first, second in OPPOSITES:
        for said, opposite in ((first, second), (second, first)):
            if (
                _has_phrase(user_answer, said)
                and _has_phrase(section, opposite)
                and not _has_phrase(section, said)
                and not _has_phrase(user_answer, opposite)
            ):
                return f'the answer says "{said}" where the video says "{opposite}"'

    # A negation the video does not have ("does NOT use labeled data").
    if _NEGATION.search(user_answer.replace("\u2019", "'")) and not _NEGATION.search(
        section.replace("\u2019", "'")
    ):
        return "the answer contains a negation the video does not have"

    return None


def _grounded_missing_concept(missing, item):
    """The LLM's missing idea, only if the video really says it."""
    if not missing:
        return None

    missing = " ".join(str(missing).split()).rstrip(".")

    if (
        not missing
        or missing.upper() == "NONE"
        or len(missing.split()) > MAX_MISSING_CONCEPT_WORDS
        or _is_meta(missing)
    ):
        return None

    section = item.get("section_text") or item.get("text", "")

    if not _is_grounded(missing, section + " " + item["answer"],
                        MISSING_CONCEPT_GROUNDING):
        return None

    return missing[0].upper() + missing[1:]


# ============================================================
# PUBLIC FUNCTION: CHECK AN ANSWER
# ============================================================

FEEDBACK = {
    "video": "Your answer matches what the video says.",
    "concept": (
        "Your answer is a correct explanation of the idea. The video "
        "explains it in its own way, shown below."
    ),
    "example": "Your answer correctly describes the video's example.",
    "general_only": (
        "Your explanation is correct in general, but this question asks "
        "what the video itself shows."
    ),
    "partial": "Your answer is on the right track but incomplete.",
    "incorrect": "Your answer does not match what the video says.",
    "off_topic": "Your answer is not about what this question asks.",
}

SIMILARITY_ONLY_NOTE = (
    "(Checked by similarity only: the language model is not available.)"
)


def check_answer(user_answer, item, embedding_model, llm=None):
    """
    Grade one answer against a viva question.

    Parameters
    ----------
    user_answer : str
        What the user typed.
    item : dict
        One question from generate_viva_questions().
    embedding_model :
        The already loaded SentenceTransformer.
    llm : callable, optional
        prompt -> text, the examiner. Defaults to the local Ollama model.
        When it is not available, similarity alone decides.

    Returns
    -------
    dict
        {
          "verdict": "correct" | "partial" | "incorrect",
          "score": 1.0 | 0.5 | 0.0,
          "similarity": float,     # best match in the section, 0..1
          "coverage": float,       # key words, 0..1
          "missing": [str, ...],   # key words not mentioned
          "missing_concept": str or None,
          "kind": "video" | "concept" | "example" | "general_only"
                  | "partial" | "incorrect" | "off_topic",
          "judged_by": "llm" | "similarity",
          "feedback": str
        }
    """
    user_answer = " ".join(str(user_answer or "").split())
    expected = item["answer"]

    def result(
        verdict,
        kind,
        feedback,
        similarity=0.0,
        coverage=0.0,
        missing=(),
        missing_concept=None,
        judged_by="similarity"
    ):
        return {
            "verdict": verdict,
            "score": VERDICT_SCORES[verdict],
            "similarity": similarity,
            "coverage": coverage,
            "missing": list(missing),
            "missing_concept": missing_concept,
            "kind": kind,
            "judged_by": judged_by,
            "feedback": feedback,
        }

    if not _content_words(user_answer):
        return result(
            "incorrect", "incorrect",
            "No answer was given."
        )

    if _only_repeats_question(user_answer, item["question"]):
        return result(
            "incorrect", "incorrect",
            "The answer only repeats the question."
        )

    # ---- 1) similarity with THIS part of the video ----------------
    references = _section_references(item)
    similarities = _similarities(user_answer, references, embedding_model)

    similarity_expected = max(0.0, similarities[0])
    similarity = max(0.0, max(similarities))
    coverage = _coverage(user_answer, expected, item["question"])
    missing = _missing_words(user_answer, expected, item["question"])

    if (
        similarity >= CORRECT_SIMILARITY
        or coverage >= CORRECT_COVERAGE
    ):
        verdict = "correct"

    elif (
        similarity >= PARTIAL_SIMILARITY
        or coverage >= PARTIAL_COVERAGE
    ):
        verdict = "partial"

    else:
        verdict = "incorrect"

    # Practically the video's own answer - but never when the answer
    # says the opposite: with a one-word key point ("feedback"), "does
    # NOT work on the principle of feedback" covers it completely.
    contradiction = _contradiction(user_answer, item)

    verbatim = (
        (
            similarity_expected >= VERBATIM_SIMILARITY
            or coverage >= VERBATIM_COVERAGE
        )
        and contradiction is None
    )

    # ---- 2) relevance gate ------------------------------------------
    on_topic = (
        len(_topic_words(user_answer, item)) >= MIN_TOPIC_WORDS
        or similarity >= PARTIAL_SIMILARITY
    )

    if not on_topic:
        return result(
            "incorrect", "off_topic",
            FEEDBACK["off_topic"],
            similarity, coverage
        )

    # ---- 3) the LLM examiner judges the meaning ---------------------
    judgement = _ask_examiner(llm or _judge_llm, user_answer, item)

    # Word hints only help when the answer already covers some of the
    # key words; otherwise they just list the example's words
    # ("Key points to mention: friend, million").
    if coverage == 0:
        missing = []

    if judgement is None:
        # Without the LLM: similarity decides, as before - but a
        # similar-sounding answer with an opposite meaning is not
        # marked correct.
        if verdict == "correct" and contradiction and not verbatim:
            verdict = "partial"

        kind = {"correct": "video"}.get(verdict, verdict)
        feedback = FEEDBACK[kind]

        if verdict != "correct" and missing:
            feedback += " Key points to mention: " + ", ".join(missing) + "."

        return result(
            verdict, kind,
            f"{feedback} {SIMILARITY_ONLY_NOTE}",
            similarity, coverage, missing
        )

    llm_verdict, answer_type, llm_contradicts, llm_missing = judgement
    missing_concept = _grounded_missing_concept(llm_missing, item)

    if llm_verdict == "correct":

        if (
            item.get("kind") == "example"
            and answer_type == "CONCEPT"
            and verdict != "correct"
        ):
            # A video-specific question answered from outside knowledge.
            final, kind = "partial", "general_only"

        elif answer_type == "EXAMPLE":
            final, kind = "correct", "example"

        elif answer_type == "CONCEPT" and verdict != "correct":
            final, kind = "correct", "concept"

        else:
            final, kind = "correct", "video"

    elif llm_verdict == "partial":
        final, kind = (
            ("correct", "video") if verbatim else ("partial", "partial")
        )

    elif verbatim:
        # The video's own answer is never failed by a flaky judgement.
        final, kind = "correct", "video"

    else:
        # SAFETY NET. llama3.2 is sometimes too strict (on the real video
        # it failed "rewards and penalties" for "the principle of
        # feedback"). An INCORRECT verdict is softened to PARTIAL only if
        # ALL of these hold:
        #   - the meaning is clearly close to this part of the video
        #     (MiniLM similarity, not shared words),
        #   - the LLM itself does not say the answer contradicts the
        #     video, and did not call it WRONG while also saying so,
        #   - the plain rules find no opposite word and no negation the
        #     video does not have.
        # Otherwise the LLM's INCORRECT stands.
        softened = (
            similarity >= CORRECT_SIMILARITY
            and not llm_contradicts
            and contradiction is None
        )

        # The LLM's own "incomplete" reading is partial, not wrong.
        incomplete = (
            answer_type == "INCOMPLETE"
            and not llm_contradicts
            and contradiction is None
        )

        if softened or incomplete:
            final, kind = "partial", "partial"
        else:
            final, kind = "incorrect", "incorrect"

    feedback = FEEDBACK[kind]

    if final != "correct":
        if missing_concept:
            feedback += f" Missing: {missing_concept}."
        elif missing:
            feedback += " Key points to mention: " + ", ".join(missing) + "."

    return result(
        final, kind, feedback,
        similarity, coverage,
        missing if final != "correct" else (),
        missing_concept if final != "correct" else None,
        judged_by="llm"
    )


# ============================================================
# PUBLIC FUNCTION: FINAL SCORE
# ============================================================

def viva_score(results, total_questions):
    """
    Total score of a viva.

    results : dict  question number -> check_answer() result
    Unanswered questions count as 0.

    Returns {"score", "total", "percent"}.
    """
    score = sum(
        result["score"]
        for result in results.values()
    )

    percent = (
        round(100 * score / total_questions)
        if total_questions
        else 0
    )

    return {
        "score": score,
        "total": total_questions,
        "percent": percent,
    }
