"""
Video summary for the AI Video Assistant.

Builds a short, transcript-only summary of a processed video.

Every summary point carries:
    title      short topic name
    summary    one sentence
    start/end  timestamps
    text       supporting transcript

IMPORTANT GUARANTEE
-------------------
The timestamps and the supporting text of a point come from the real
Whisper segments and from nowhere else: start is the start of one
segment, end is the end of the same or a later segment, and text is
exactly those segments joined in order. The LLM never supplies a
timestamp or evidence, so neither can be invented. The LLM only
phrases the title and the one-sentence summary, and each phrase is
checked against the evidence it was written from. If the LLM is
unavailable or produces something that is not supported by that
evidence, a plain extractive wording is used instead.

Chunks are 20-second windows, so they often begin or end in the
middle of a sentence ("higher will be the accuracy. There are many
ways ..."). The evidence of a point is therefore snapped to sentence
boundaries, in whole segments: a sentence that began in the previous
chunk is pulled in (the start moves back to the segment where it
began), and an unfinished sentence at the end is left to the next
window, which owns it.

The one-sentence description must also be a COMPLETE sentence
(see sentence_problem). Chunks are time windows, so they often
start or end in the middle of a sentence. A description that is
a fragment is replaced by the nearest complete sentence of the
real transcript. The timestamps do not change when that happens.

This module does not import Streamlit and does not touch the
FAISS index, the chunker or the question-answering code.
"""

import bisect
import math
import re
from collections import Counter

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Maximum number of summary points. Fewer are returned when the
# video does not contain that many distinct meaningful topics.
MAX_POINTS = 5

# A chunk is "meaningful" only if it has enough real words.
# This drops filler such as "Simple, right?".
MIN_WORDS = 8
MIN_CONTENT_WORDS = 4

# Two chunks describe the same topic if their embeddings are
# at least this similar (cosine) ...
DUPLICATE_SIMILARITY = 0.75

# ... or if they share this fraction of their content words.
DUPLICATE_OVERLAP = 0.60

# Balance between "central to the video" (1.0) and
# "different from what is already selected" (0.0).
MMR_LAMBDA = 0.6

# Local LLM, the same model the question answering uses.
LLM_MODEL = "llama3.2"

MAX_SUMMARY_CHARS = 220
MAX_TITLE_WORDS = 6

# Minimum share of the words in an LLM-written title / summary
# that must appear in the transcript it was written from.
SUMMARY_GROUNDING = 0.60
TITLE_GROUNDING = 0.50

# A description must have at least this many words to count as
# a sentence ("Higher accuracy." is not one).
MIN_SENTENCE_WORDS = 5

# Transcript sentences used as a fallback may be a little longer
# than an LLM-written description: people speak in long sentences.
MAX_FALLBACK_CHARS = 320

# A rejected draft counts as "a piece of" a transcript sentence when
# at least this share of its content words occur in that sentence.
DRAFT_MATCH = 0.60

# How far (seconds) the evidence of a point may be moved to reach a
# sentence boundary: back to where a sentence began, or forward to
# where one ends.
MAX_EXTEND_SECONDS = 15.0

# A silence at least this long (seconds) between two Whisper
# segments is treated as the end of a sentence, even when the
# text before it has no full stop.
SENTENCE_GAP_SECONDS = 2.0


# ============================================================
# TEXT HELPERS
# ============================================================

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "even",
    "few", "for", "from", "further", "get", "got", "going", "had",
    "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "let", "like", "many", "me", "more", "most", "much", "my", "no",
    "nor", "not", "now", "of", "off", "ok", "okay", "on", "once", "one",
    "only", "or", "other", "our", "ours", "out", "over", "own", "same",
    "say", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "us",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "yeah",
    "yes", "you", "your", "yours", "right", "well", "really",
    "let's", "that's", "it's", "i'll", "i'm", "i've", "you're",
    "you'll", "we're", "we'll", "we've", "don't", "doesn't", "didn't",
    "can't", "won't", "isn't", "aren't", "they're", "there's",
    "what's", "he's", "she's",
    "others", "thing", "things", "something", "anything", "everything",
    "today", "give", "gives", "make", "makes", "made", "know", "want",
    "need",
}

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def _tokens(text):
    return _TOKEN_PATTERN.findall(str(text).lower())


def _content_words(text):
    return [
        word
        for word in _tokens(text)
        if word not in STOPWORDS
        and len(word) > 2
        and not word.isdigit()
    ]


def _stem(word):
    """Very light stemming, only used to compare words."""
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def _stem_set(text):
    return {_stem(word) for word in _content_words(text)}


def _jaccard(first, second):
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _title_case(text):
    words = text.split()
    return " ".join(
        word if word[:1].isupper() else word.capitalize()
        for word in words
    )


def escape_markdown(text):
    """
    Escape characters that Streamlit markdown would interpret
    (bold, links, headings, LaTeX with $...) so that transcript
    and LLM text is always shown literally.
    """
    return re.sub(r"([\\`*_\[\]<>~$#|])", r"\\\1", str(text))


# ============================================================
# SENTENCE COMPLETENESS
# ============================================================
#
# Plain text rules (no language model) that decide whether a piece
# of text is a complete sentence. They exist to catch the fragments
# that time-window chunks produce, for example
# "higher will be the accuracy."

# Words that open a dependent clause: "Which is used by Uber."
# is a fragment, not a sentence.
FRAGMENT_STARTERS = {
    "and", "but", "or", "nor", "which", "whose", "whom",
    "because", "although", "though", "whereas", "than", "plus",
}

# Words a sentence cannot end on. A text that stops on one of
# them was cut off ("... is used in the.").
DANGLING_ENDINGS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "into", "onto", "about", "as", "than", "and",
    "or", "but", "nor", "because", "if", "when", "while", "where",
    "whose", "which", "will", "would", "can", "could", "should",
    "may", "might", "must", "its", "their", "our", "your", "my",
    "his",
}

# A full stop after these does not end a sentence.
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "e.g",
    "i.e", "approx", "fig", "eq", "inc", "ltd", "st",
}

_OPENING_MARKS = "\"'\u201c\u2018([{"
_CLOSING_MARKS = "\"'\u201d\u2019)]}"


def _starts_lowercase(text):
    """
    True if the text begins in the middle of a sentence (a lowercase
    first word). Words that are written that way on purpose (iPhone,
    eBay, macOS) have a capital letter inside them and are allowed.
    """
    body = " ".join(str(text or "").split()).lstrip(_OPENING_MARKS + " ")

    if not body or not body[0].isalnum():
        return False

    first_word = body.split(" ", 1)[0]

    return body[0].islower() and not any(
        character.isupper() for character in first_word[1:]
    )


def sentence_problem(text, max_chars=MAX_SUMMARY_CHARS):
    """
    Return None if `text` is a complete sentence, otherwise a short
    reason why it is not:

        empty, too_short, too_long, starts_with_punctuation,
        starts_lowercase, fragment_start, truncated,
        no_end_punctuation, dangling_end, unbalanced
    """
    sentence = " ".join(str(text or "").split())

    if not sentence:
        return "empty"

    if len(_tokens(sentence)) < MIN_SENTENCE_WORDS:
        return "too_short"

    if len(sentence) > max_chars:
        return "too_long"

    # ---- how it starts ------------------------------------------
    body = sentence.lstrip(_OPENING_MARKS + " ")

    if not body or not body[0].isalnum():
        return "starts_with_punctuation"

    first_word = body.split(" ", 1)[0]

    # A lowercase start means the text begins mid-sentence.
    if _starts_lowercase(sentence):
        return "starts_lowercase"

    if first_word.lower().strip(",.;:!?") in FRAGMENT_STARTERS:
        return "fragment_start"

    # ---- how it ends --------------------------------------------
    ending = sentence.rstrip(_CLOSING_MARKS + " ")

    if ending.endswith("...") or ending.endswith("\u2026"):
        return "truncated"

    if not ending or ending[-1] not in ".!?":
        return "no_end_punctuation"

    if _tokens(ending)[-1] in DANGLING_ENDINGS:
        return "dangling_end"

    # ---- quotes / brackets ----------------------------------------
    if (
        sentence.count('"') % 2
        or sentence.count("(") != sentence.count(")")
        or sentence.count("\u201c") != sentence.count("\u201d")
    ):
        return "unbalanced"

    return None


def is_complete_sentence(text, max_chars=MAX_SUMMARY_CHARS):
    return sentence_problem(text, max_chars) is None


# ============================================================
# TIMESTAMP FORMATTING
# ============================================================
#
# Display only. Seeking still uses seconds. Times are cut down to
# whole seconds, exactly like the Jump button's target, so the
# label and the position the player jumps to always agree.

def format_timestamp(seconds):
    """143.0 -> '2:23'   3725 -> '1:02:05'"""
    total = max(0, int(float(seconds)))

    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"


def format_time_range(start, end):
    """(143.0, 161.4) -> '2:23 \u2013 2:41'"""
    first = format_timestamp(start)
    last = format_timestamp(end)

    if first == last:
        return first

    return f"{first} \u2013 {last}"


# ============================================================
# TRANSCRIPT SENTENCES  (fallback wording)
# ============================================================
#
# Chunks are groups of whole Whisper segments, and a sentence can
# run across two chunks. Here the segments are taken back out of the
# chunks and the sentences of the WHOLE transcript are rebuilt; every
# sentence remembers which segments (and chunks) it spans. This is
# used for two things:
#
#   - snapping the evidence of a point to sentence boundaries, so it
#     never begins or ends in the middle of a sentence;
#   - finding a complete sentence to describe a point when the LLM
#     wording is rejected.
#
# Nothing is written here: every sentence is real transcript text,
# and every timestamp is the start or end of a real segment.

_SENTENCE_BOUNDARY = re.compile(
    r"([.!?][\"'\u201d\u2019)\]]*)\s+"
    r"(?=[\"'\u201c\u2018(\[]?[A-Z0-9])"
)


def _sentence_spans(text, breaks=()):
    """
    (begin, end) offsets of the sentences in `text`.
    `breaks` are offsets where a sentence must end (long silences).
    """
    edges = (
        [0]
        + sorted(b for b in set(breaks) if 0 < b < len(text))
        + [len(text)]
    )

    spans = []

    for left, right in zip(edges, edges[1:]):

        block = text[left:right]
        begin = 0

        for match in _SENTENCE_BOUNDARY.finditer(block):

            end = match.end(1)

            words = block[begin:end].split()
            last = (
                words[-1].lower().strip(".!?\"'\u201d\u2019)]([")
                if words else ""
            )

            # "Dr. Smith", "e.g. this", "J. Smith": not a sentence end.
            if (
                last in ABBREVIATIONS
                or (len(last) == 1 and last.isalpha())
            ):
                continue

            spans.append((left + begin, left + end))
            begin = match.end()

        if block[begin:].strip():
            spans.append((left + begin, right))

    return spans


def _seconds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _distance(later, earlier):
    try:
        return float(later) - float(earlier)
    except (TypeError, ValueError):
        return float("inf")


def _transcript_index(chunks):
    """
    Flatten the chunks back into the real Whisper segments and rebuild
    the sentences of the whole transcript.

    Returns a dict:
        segments      [{"start", "end", "text", "chunk"}], in video order
        sentences     [{"id", "text", "segments": (first, last),
                        "chunks": [...], "complete": bool}, ...]
        starts        indexes of the segments that begin a sentence
        ends          indexes of the segments that end a sentence
        chunk_ranges  {chunk index: (first segment, last segment)}
    """
    segments = []
    ranges = {}

    for number, chunk in enumerate(chunks or []):

        parts = []

        try:
            for segment in (chunk.get("segments") or []):
                parts.append((
                    str(segment.get("text", "")),
                    _seconds(segment.get("start")),
                    _seconds(segment.get("end")),
                ))
        except (AttributeError, TypeError):
            parts = []

        if not any(part[0].strip() for part in parts):
            try:
                parts = [(
                    str(chunk.get("text", "")),
                    _seconds(chunk.get("start")),
                    _seconds(chunk.get("end")),
                )]
            except AttributeError:
                parts = []

        for text, start, end in parts:

            text = " ".join(text.split())

            if not text:
                continue

            position = len(segments)

            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "chunk": number,
            })

            if number in ranges:
                ranges[number][1] = position
            else:
                ranges[number] = [position, position]

    # ---- one running text, remembering where every segment sits ----
    full = ""
    offsets = []
    breaks = []
    previous_end = None

    for segment in segments:

        if full:
            full += " "

            # A long silence ends the sentence, whatever the text says.
            if (
                previous_end is not None
                and segment["start"] is not None
                and segment["start"] - previous_end >= SENTENCE_GAP_SECONDS
            ):
                breaks.append(len(full))

        offsets.append((len(full), len(full) + len(segment["text"])))
        full += segment["text"]
        previous_end = segment["end"]

    # ---- sentences, and which segments begin / end one -------------
    begins = [offset[0] for offset in offsets]

    sentences = []
    starts = set()
    ends = set()

    for left, right in _sentence_spans(full, breaks):

        raw = full[left:right]
        text = raw.strip()

        if not text:
            continue

        begin = left + (len(raw) - len(raw.lstrip()))
        end = begin + len(text)

        first = bisect.bisect_right(begins, begin) - 1
        last = bisect.bisect_right(begins, end - 1) - 1

        if begin == offsets[first][0]:
            starts.add(first)

        if end == offsets[last][1]:
            ends.add(last)

        sentences.append({
            "id": len(sentences),
            "text": text,
            "segments": (first, last),
            "chunks": sorted({
                segments[i]["chunk"] for i in range(first, last + 1)
            }),
            "complete": (
                sentence_problem(text, MAX_FALLBACK_CHARS) is None
            ),
        })

    return {
        "segments": segments,
        "sentences": sentences,
        "starts": starts,
        "ends": ends,
        "chunk_ranges": {
            number: (span[0], span[1]) for number, span in ranges.items()
        },
    }


def _build_sentences(chunks):
    """The rebuilt sentences of the whole transcript."""
    return _transcript_index(chunks)["sentences"]


def _window_text(index, first, last):
    return " ".join(
        index["segments"][i]["text"] for i in range(first, last + 1)
    )


def _snap_windows(index):
    """
    Put the boundaries between neighbouring chunks on sentence
    boundaries, so that every sentence belongs to exactly one window and
    no window begins or ends in the middle of one.

    A boundary that falls inside a sentence is moved:
      - BACK to where that sentence begins, so the sentence is pulled
        into the later window (at most MAX_EXTEND_SECONDS back), or,
        when the earlier window would be left with too little,
      - FORWARD to where the sentence ends, so the earlier window keeps
        it (at most MAX_EXTEND_SECONDS forward).

    What still cannot be fixed (no sentence boundary close enough) is
    cleaned up inside each window: a leading fragment is dropped and an
    unfinished last sentence is left out. A fragment is never shown.

    Only whole segments change sides, so every window starts at the real
    start of a segment and ends at the real end of a segment.

    Returns {chunk index: (first segment, last segment)}.
    """
    segments = index["segments"]
    starts = index["starts"]
    ends = index["ends"]

    def words(first, last):
        return len(_tokens(_window_text(index, first, last)))

    order = sorted(index["chunk_ranges"])
    windows = [list(index["chunk_ranges"][number]) for number in order]

    # ---- 1. the boundaries between neighbouring windows ------------
    for position in range(len(windows) - 1):

        left = windows[position]
        right = windows[position + 1]
        boundary = right[0]

        # Already between two sentences (or not neighbouring segments).
        if boundary in starts or left[1] + 1 != boundary:
            continue

        moved = None

        # a) pull the whole sentence into the later window
        k = next(
            (k for k in range(boundary - 1, left[0], -1) if k in starts),
            None
        )

        if (
            k is not None
            and _distance(
                segments[boundary]["start"], segments[k]["start"]
            ) <= MAX_EXTEND_SECONDS
            and words(left[0], k - 1) >= MIN_WORDS
        ):
            moved = k

        # b) otherwise leave the whole sentence to the earlier window
        if moved is None:

            k = next(
                (
                    k for k in range(boundary + 1, right[1] + 1)
                    if k in starts
                ),
                None
            )

            if (
                k is not None
                and _distance(
                    segments[k - 1]["end"], segments[boundary - 1]["end"]
                ) <= MAX_EXTEND_SECONDS
            ):
                moved = k

        if moved is not None:
            left[1] = moved - 1
            right[0] = moved

    # ---- 2. what could not be fixed: no fragment at either end ----
    for window in windows:

        if window[0] not in starts:
            k = next(
                (
                    k for k in range(window[0] + 1, window[1] + 1)
                    if k in starts
                ),
                None
            )

            if k is not None:
                window[0] = k

        if window[1] not in ends:
            k = next(
                (
                    k for k in range(window[1] - 1, window[0] - 1, -1)
                    if k in ends
                ),
                None
            )

            if k is not None and words(window[0], k) >= MIN_WORDS:
                window[1] = k

    return {
        number: (window[0], window[1])
        for number, window in zip(order, windows)
    }


def _fallback_pool(index, first, last):
    """
    Complete transcript sentences that can describe the evidence
    window made of segments first..last.

    rank 0: lies entirely inside the window
    rank 1: overlaps the window (it starts or ends outside it)

    A sentence that does not touch the window is never used: it could
    describe something other than the evidence shown.
    """
    if first is None or last is None:
        return []

    pool = []

    for sentence in index["sentences"]:

        if not sentence["complete"]:
            continue

        begin, end = sentence["segments"]

        if end < first or begin > last:
            continue

        pool.append((0 if begin >= first and end <= last else 1, sentence))

    return pool


def _pick_fallback_sentence(pool, draft, used):
    """
    Choose the sentence that describes an evidence window.

    1. Inside the window first, then one that overlaps its edge: the
       description must be about the evidence that is shown.
    2. Statements before questions.
    3. The sentence the rejected draft was a piece of. The words of a
       fragment lie inside the sentence it was cut from, so this repairs
       "higher will be the accuracy." into the whole sentence. A draft
       that only shares a word or two with a sentence is ignored.
    4. The sentence carrying the most information.
    """
    draft_stems = _stem_set(draft) if draft else set()

    best = None
    best_key = None

    for rank, sentence in pool:

        if sentence["id"] in used:
            continue

        stems = _stem_set(sentence["text"])

        match = (
            len(draft_stems & stems) / len(draft_stems)
            if draft_stems else 0.0
        )

        key = (
            rank,
            sentence["text"].rstrip(_CLOSING_MARKS + " ").endswith("?"),
            -(match if match >= DRAFT_MATCH else 0.0),
            -len(stems),
            sentence["id"],
        )

        if best_key is None or key < best_key:
            best = sentence
            best_key = key

    return best


# ============================================================
# CANDIDATE CHUNKS
# ============================================================

def _prepare_candidates(chunks, min_words, min_content_words, index, windows):
    """
    Keep only chunks that are real, well-formed and meaningful, with
    the evidence snapped to sentence boundaries (see _snap_windows).

    start, end and text always describe the same run of real Whisper
    segments. When the chunk already begins and ends on sentence
    boundaries it is carried over untouched.
    """
    candidates = []

    for number, chunk in enumerate(chunks or []):

        try:
            text = str(chunk.get("text", "")).strip()
            start = float(chunk["start"])
            end = float(chunk["end"])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

        if not text:
            continue

        first = last = None
        span = index["chunk_ranges"].get(number)

        if span is not None:

            first, last = span
            snapped = windows.get(number, span)

            if snapped != span:

                new_start = index["segments"][snapped[0]]["start"]
                new_end = index["segments"][snapped[1]]["end"]

                if new_start is not None and new_end is not None:
                    first, last = snapped
                    start, end = new_start, new_end
                    text = _window_text(index, first, last)

        if (
            len(_tokens(text)) < min_words
            or len(_content_words(text)) < min_content_words
        ):
            continue

        candidates.append({
            "chunk_index": number,
            "first": first,
            "last": last,
            "start": start,
            "end": end,
            "text": text,
            "clean_start": not _starts_lowercase(text),
            "stems": _stem_set(text),
        })

    return candidates


def _embed(candidates, embedding_model):
    vectors = embedding_model.encode(
        [candidate["text"] for candidate in candidates],
        normalize_embeddings=True
    )

    vectors = np.asarray(vectors, dtype="float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return vectors / norms


# ============================================================
# POINT SELECTION  (relevance + diversity, duplicates removed)
# ============================================================

def _select_points(candidates, vectors, max_points):
    """
    Greedy maximal-marginal-relevance selection.

    Picks chunks that are central to the video but different from
    each other. A chunk that describes the same topic as an
    already selected chunk is discarded for good.

    Returns indices into `candidates`, in selection order.
    """
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)

    if norm > 0:
        centroid = centroid / norm

    relevance = vectors @ centroid

    remaining = list(range(len(candidates)))
    selected = []

    while remaining and len(selected) < max_points:

        best_index = None
        best_score = None
        still_valid = []

        for i in remaining:

            if selected:
                similarity = max(
                    float(vectors[i] @ vectors[j])
                    for j in selected
                )

                overlap = max(
                    _jaccard(
                        candidates[i]["stems"],
                        candidates[j]["stems"]
                    )
                    for j in selected
                )

                # Same topic as something already selected.
                if (
                    similarity >= DUPLICATE_SIMILARITY
                    or overlap >= DUPLICATE_OVERLAP
                ):
                    continue
            else:
                similarity = 0.0

            still_valid.append(i)

            score = (
                MMR_LAMBDA * float(relevance[i])
                - (1.0 - MMR_LAMBDA) * similarity
            )

            if best_score is None or score > best_score:
                best_score = score
                best_index = i

        remaining = still_valid

        if best_index is None:
            break

        selected.append(best_index)
        remaining.remove(best_index)

    return selected


# ============================================================
# EXTRACTIVE WORDING (used when the LLM is unavailable/unsafe)
# ============================================================

def _extractive_title(text, document_frequency, total_chunks):
    counts = Counter(_content_words(text))

    if not counts:
        return "Key Point"

    scored = []

    for word, count in counts.items():
        idf = math.log(
            1.0 + total_chunks / max(1, document_frequency.get(word, 1))
        )
        scored.append((count * idf * min(len(word), 10), word))

    scored.sort(key=lambda item: (-item[0], item[1]))

    chosen = [scored[0][1]]

    if len(scored) > 1 and scored[1][0] >= 0.75 * scored[0][0]:
        chosen.append(scored[1][1])

    # Show the words in the order they are spoken.
    spoken = [
        word for word in _content_words(text)
        if word in chosen
    ]

    ordered = []
    for word in spoken:
        if word not in ordered:
            ordered.append(word)

    return _title_case(" ".join(ordered))


def _first_sentences(text, limit=MAX_SUMMARY_CHARS):
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())

    sentence = pieces[0]

    if len(sentence.split()) < 5 and len(pieces) > 1:
        sentence = f"{sentence} {pieces[1]}"

    return _limit_length(sentence, limit)


def _limit_length(sentence, limit=MAX_SUMMARY_CHARS):
    sentence = " ".join(sentence.split())

    if len(sentence) <= limit:
        return sentence

    cut = sentence[:limit].rsplit(" ", 1)[0].rstrip(",;:")

    return cut + "..."


def _extractive_topic(texts):
    """Most repeated word pair (else word) across the transcript."""
    pairs = Counter()
    words = Counter()

    for text in texts:
        sequence = _tokens(text)

        for first, second in zip(sequence, sequence[1:]):
            if (
                first not in STOPWORDS and second not in STOPWORDS
                and len(first) > 2 and len(second) > 2
            ):
                pairs[(first, second)] += 1

        words.update(_content_words(text))

    if pairs:
        (first, second), count = pairs.most_common(1)[0]
        if count >= 2:
            return _title_case(f"{first} {second}")

    if words:
        best = max(words.items(), key=lambda item: (item[1], len(item[0])))
        return _title_case(best[0])

    return "Video"


# ============================================================
# LLM WORDING (checked against the transcript)
# ============================================================

POINT_PROMPT = """You are summarizing one part of a video using ONLY its transcript.
Do not use outside knowledge and do not add facts that are not in the excerpt.

Transcript excerpt:
__TEXT__

Reply in exactly this format and nothing else:
TITLE: <a 2 to 4 word topic name>
SUMMARY: <one complete sentence that starts with a capital letter and ends with a full stop, saying what this part of the video explains>
"""

TOPIC_PROMPT = """Below are short notes taken from one video.
Using ONLY these notes, name the single main topic of the video in 2 to 4 words.
Do not use outside knowledge.

Notes:
__NOTES__

Reply in exactly this format and nothing else:
MAIN TOPIC: <2 to 4 words>
"""


def _default_llm(prompt):
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
        ]
    )

    return response["message"]["content"]


def _extract_field(raw, label):
    """Read 'LABEL: value' from messy model output."""
    cleaned = re.sub(r"[*`#]", "", str(raw))

    match = re.search(
        rf"^\s*[-•>]*\s*{label}\s*[:\-–—]\s*(.+?)\s*$",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if not match:
        return None

    value = match.group(1).strip().strip("\"'“”‘’ ").strip()

    return value or None


def _is_grounded(text, source_text, minimum):
    """True if enough of the words in `text` occur in the source."""
    words = _stem_set(text)

    if not words:
        return False

    supported = words & _stem_set(source_text)

    return len(supported) / len(words) >= minimum


def _clean_title(value):
    if not value:
        return None

    value = value.rstrip(".:;,").strip()

    if not value or len(value.split()) > MAX_TITLE_WORDS:
        return None

    return value if not value.islower() else _title_case(value)


def _clean_summary(value):
    """
    First sentence of the model output, with spacing tidied.

    Nothing is added or shortened here: a missing capital letter,
    a missing full stop or a cut-off ending is a sign of a fragment,
    and sentence_problem() must still be able to see it.
    """
    if not value:
        return None

    sentence = re.split(r"(?<=[.!?])\s+", value.strip())[0]

    return " ".join(sentence.split()) or None


def _call_llm(llm, prompt, state):
    """One guarded LLM call. After a failure the LLM is not retried."""
    if not state["llm_ok"]:
        return None

    try:
        return llm(prompt)
    except Exception:
        state["llm_ok"] = False
        return None


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_video_summary(
    chunks,
    embedding_model,
    llm=None,
    max_points=MAX_POINTS
):
    """
    Build a transcript-grounded summary.

    Parameters
    ----------
    chunks : list
        The existing timestamped chunks
        (dicts with start, end, text, segments).
    embedding_model :
        The already loaded SentenceTransformer.
    llm : callable, optional
        prompt -> text. Defaults to the local Ollama model.
    max_points : int
        Upper limit for the number of summary points.

    Returns
    -------
    dict
        {
          "main_topic": str,
          "points": [
              {
                "title", "summary",
                "start", "end", "text",   # copied from a chunk
                "chunk_index",
                "segment_range",          # first/last Whisper segment
                "description_source"      # llm | transcript_sentence
                                          # | chunk_text
              }, ...                      # in video order
          ],
          "used_llm": bool
        }
    """
    empty = {
        "main_topic": "",
        "points": [],
        "used_llm": False
    }

    # The real segments and sentences of the transcript, and the
    # chunk boundaries moved onto sentence boundaries.
    index = _transcript_index(chunks)
    windows = _snap_windows(index)

    candidates = _prepare_candidates(
        chunks,
        MIN_WORDS,
        MIN_CONTENT_WORDS,
        index,
        windows
    )

    # Very short videos: relax the filter instead of returning nothing.
    if not candidates:
        candidates = _prepare_candidates(chunks, 4, 2, index, windows)

    if not candidates:
        return empty

    # Evidence that still begins in the middle of a sentence (the
    # beginning could not be found) is not shown: prefer the others.
    clean = [
        candidate for candidate in candidates
        if candidate["clean_start"]
    ]

    if clean:
        candidates = clean

    # Prefer parts of the video that a complete sentence can describe.
    describable = [
        candidate for candidate in candidates
        if _fallback_pool(index, candidate["first"], candidate["last"])
    ]

    if describable:
        candidates = describable

    llm = llm or _default_llm
    state = {"llm_ok": True}
    used_llm = False

    # ---- choose WHICH parts of the video become points --------
    vectors = _embed(candidates, embedding_model)

    selected = _select_points(candidates, vectors, max_points)

    # Present the points in the order they appear in the video.
    selected.sort(key=lambda i: candidates[i]["start"])

    document_frequency = Counter()
    for candidate in candidates:
        document_frequency.update(
            set(_content_words(candidate["text"]))
        )

    # ---- word each point ------------------------------------------
    points = []
    used_sentences = set()

    for i in selected:

        candidate = candidates[i]
        text = candidate["text"]

        raw = _call_llm(
            llm,
            POINT_PROMPT.replace("__TEXT__", text),
            state
        )

        title = None
        summary = None

        if raw:
            title = _clean_title(_extract_field(raw, "title"))
            summary = _clean_summary(_extract_field(raw, "summary"))

            # Model ignored the format: accept a single plain sentence.
            if (
                summary is None
                and _extract_field(raw, "title") is None
                and "\n" not in raw.strip()
            ):
                summary = _clean_summary(raw)

        if title and not _is_grounded(title, text, TITLE_GROUNDING):
            title = None

        if title:
            used_llm = True

        # ---- the one-sentence description ------------------------
        # 1) the LLM sentence, if it is a COMPLETE sentence that the
        #    transcript supports;
        # 2) otherwise the nearest complete sentence of the real
        #    transcript;
        # 3) last resort (a video with no complete sentence at all):
        #    the chunk text itself.
        if (
            summary
            and sentence_problem(summary) is None
            and _is_grounded(summary, text, SUMMARY_GROUNDING)
        ):
            description = summary
            description_source = "llm"
            used_llm = True

        else:
            pool = _fallback_pool(
                index,
                candidate["first"],
                candidate["last"]
            )

            fallback = _pick_fallback_sentence(
                pool,
                summary,
                used_sentences
            )

            if fallback is not None:
                description = fallback["text"]
                description_source = "transcript_sentence"
                used_sentences.add(fallback["id"])

            elif pool:
                # Every sentence that fits this chunk already describes
                # an earlier point: this would be a duplicate topic.
                continue

            else:
                description = _first_sentences(text)
                description_source = "chunk_text"

        points.append({
            "title": title or _extractive_title(
                text,
                document_frequency,
                len(candidates)
            ),
            "summary": description,
            "start": candidate["start"],
            "end": candidate["end"],
            "text": text,
            "chunk_index": candidate["chunk_index"],
            "segment_range": [candidate["first"], candidate["last"]],
            "description_source": description_source,
        })

    # ---- drop points whose final wording repeats an earlier one ----
    unique_points = []

    for point in points:
        words = _stem_set(point["summary"])

        if any(
            _jaccard(words, _stem_set(kept["summary"])) >= 0.8
            for kept in unique_points
        ):
            continue

        unique_points.append(point)

    points = unique_points

    # ---- main topic -----------------------------------------------
    transcript_texts = [candidate["text"] for candidate in candidates]

    main_topic = None

    notes = "\n".join(
        f"- {point['title']}: {point['summary']}"
        for point in points
    )

    raw_topic = _call_llm(
        llm,
        TOPIC_PROMPT.replace("__NOTES__", notes),
        state
    )

    if raw_topic:
        topic = _extract_field(raw_topic, "main topic")

        if topic:
            topic = topic.rstrip(".:;,").strip()

        if (
            topic
            and len(topic.split()) <= MAX_TITLE_WORDS
            and _is_grounded(
                topic,
                " ".join(transcript_texts),
                TITLE_GROUNDING
            )
        ):
            main_topic = (
                topic if not topic.islower() else _title_case(topic)
            )
            used_llm = True

    if not main_topic:
        main_topic = _extractive_topic(transcript_texts)

    return {
        "main_topic": main_topic,
        "points": points,
        "used_llm": used_llm
    }