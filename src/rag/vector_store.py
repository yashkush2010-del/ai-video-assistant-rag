import re

import faiss
import numpy as np


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_index(embeddings):

    vectors = np.asarray(
        embeddings,
        dtype="float32"
    )

    if vectors.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array."
        )

    if len(vectors) == 0:
        raise ValueError(
            "Cannot create FAISS index from empty embeddings."
        )

    # Normalize vectors so inner product becomes
    # cosine similarity.
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(vectors)

    return index


# ============================================================
# SEARCH FAISS INDEX
# ============================================================

def search_index(
    index,
    query_embedding,
    k=6
):

    query_vector = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    # Same normalization used for stored vectors.
    faiss.normalize_L2(
        query_vector
    )

    # Never ask FAISS for more vectors
    # than actually exist.
    k = min(
        k,
        index.ntotal
    )

    if k <= 0:
        return (
            np.array([[]]),
            np.array([[]], dtype=int)
        )

    similarities, indices = index.search(
        query_vector,
        k
    )

    return similarities, indices


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    index,
    chunks,
    query_embedding,
    k=6,
    threshold=0.25
):

    similarities, indices = search_index(
        index,
        query_embedding,
        k
    )

    results = []

    for similarity, chunk_index in zip(
        similarities[0],
        indices[0]
    ):

        if chunk_index == -1:
            continue

        similarity = float(
            similarity
        )

        # Higher cosine similarity = better match.
        if similarity < threshold:
            continue

        chunk = chunks[chunk_index]

        results.append({
            "score": similarity,
            # Its position in the video, so the pieces next to it
            # can be found (add_neighbouring_chunks).
            "chunk_index": int(
                chunk_index
            ),
            "start": float(
                chunk["start"]
            ),
            "end": float(
                chunk["end"]
            ),
            "text": chunk["text"],
            "segments": chunk.get(
                "segments",
                []
            )
        })

    # Highest similarity first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# ADD NEIGHBOURING CHUNKS
# ============================================================
#
# The video is cut into chunks of about 20 seconds, and the cut often
# falls in the middle of a sentence or an explanation. Real example:
#
#     chunk 1: "... Paul decides this on the basis of the song's tempo,"
#     chunk 2: "genre, intensity and the gender of voice. ..."
#
# The search found chunk 1 but not chunk 2, so the answer only named
# the tempo. For every chunk the search finds, the chunk just before
# and just after it are added too, so a sentence or a story is never
# cut in half. The best matches are expanded first, and the total is
# capped so the context stays a size the language model handles well.
# (12 made answers slower; 9 still kept every neighbour that fixed an
# answer in scripts/check_qa_real_video.py.)

MAX_CONTEXT_CHUNKS = 9


def add_neighbouring_chunks(
    results,
    chunks,
    max_chunks=MAX_CONTEXT_CHUNKS
):

    # Results without a position cannot be expanded: unchanged.
    if not any(
        "chunk_index" in result
        for result in results
    ):
        return list(results)

    chosen = []

    def add(chunk_index, score):

        if len(chosen) >= max_chunks:
            return

        if not 0 <= chunk_index < len(chunks):
            return

        if any(
            item["chunk_index"] == chunk_index
            for item in chosen
        ):
            return

        chunk = chunks[chunk_index]

        chosen.append({
            # A neighbour gets the score of the match it belongs to.
            "score": score,
            "chunk_index": chunk_index,
            "start": float(
                chunk["start"]
            ),
            "end": float(
                chunk["end"]
            ),
            "text": chunk["text"],
            "segments": chunk.get(
                "segments",
                []
            )
        })

    # Every match first (best first), so a neighbour can never push
    # out a real match ...
    for result in results:

        if "chunk_index" in result:

            add(
                result["chunk_index"],
                result["score"]
            )

    # ... then their neighbours, again best match first.
    for result in results:

        if "chunk_index" not in result:
            continue

        for neighbour in (
            result["chunk_index"] - 1,
            result["chunk_index"] + 1
        ):

            add(
                neighbour,
                result["score"]
            )

    # In the order of the video, so the language model reads the
    # transcript as it was spoken.
    chosen.sort(
        key=lambda item: item["start"]
    )

    return chosen


# ============================================================
# FIND RELEVANT TRANSCRIPT SEGMENTS
# ============================================================

def find_relevant_segments(
    results,
    query_embedding,
    embedding_model,
    max_segments=2
):

    relevant_segments = []

    if not results:
        return relevant_segments

    for result in results:

        segments = result.get(
            "segments",
            []
        )

        if not segments:
            continue

        segment_texts = [
            segment.get(
                "text",
                ""
            ).strip()
            for segment in segments
        ]

        segment_texts = [
            text
            for text in segment_texts
            if text
        ]

        if not segment_texts:
            continue

        segment_embeddings = embedding_model.encode(
            segment_texts,
            normalize_embeddings=True
        )

        query_vector = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_norm = np.linalg.norm(
            query_vector
        )

        if query_norm > 0:
            query_vector = (
                query_vector /
                query_norm
            )

        similarities = (
            np.asarray(
                segment_embeddings
            )
            @ query_vector
        )

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        for segment_index in ranked_indices[
            :max_segments
        ]:

            segment = segments[
                int(segment_index)
            ]

            relevant_segments.append({
                "score": float(
                    similarities[
                        segment_index
                    ]
                ),
                "start": float(
                    segment["start"]
                ),
                "end": float(
                    segment["end"]
                ),
                "text": segment.get(
                    "text",
                    ""
                ).strip()
            })

    relevant_segments.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return relevant_segments[
        :max_segments
    ]


# ============================================================
# FIND EVIDENCE FOR GENERATED ANSWER
# ============================================================

def find_answer_segments(
    results,
    answer,
    embedding_model,
    max_segments=2
):

    if not results:
        return []

    if not answer:
        return []

    answer_embedding = embedding_model.encode(
        answer,
        normalize_embeddings=True
    )

    answer_embedding = np.asarray(
        answer_embedding,
        dtype="float32"
    )

    all_segments = []

    for result in results:

        for segment in result.get(
            "segments",
            []
        ):

            text = segment.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            all_segments.append(
                segment
            )

    if not all_segments:
        return []

    segment_texts = [
        segment.get(
            "text",
            ""
        ).strip()
        for segment in all_segments
    ]

    segment_embeddings = embedding_model.encode(
        segment_texts,
        normalize_embeddings=True
    )

    segment_embeddings = np.asarray(
        segment_embeddings,
        dtype="float32"
    )

    similarities = (
        segment_embeddings
        @ answer_embedding
    )

    scored_segments = []

    for segment, similarity in zip(
        all_segments,
        similarities
    ):

        scored_segments.append({
            "score": float(
                similarity
            ),
            "start": float(
                segment["start"]
            ),
            "end": float(
                segment["end"]
            ),
            "text": segment.get(
                "text",
                ""
            ).strip()
        })

    # Highest similarity first
    scored_segments.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_segments[
        :max_segments
    ]


# ============================================================
# FIND EVIDENCE FOR A QUESTION AND ITS ANSWER
# ============================================================
#
# The evidence is the transcript text shown under the answer, with
# the Jump time. find_answer_segments() picks the ONE Whisper piece
# most similar to the whole answer. Two problems, measured with
# scripts/check_evidence.py on two videos (18 questions):
#
#   - Whisper cuts speech into very short pieces, often half a
#     sentence. "Well, that's called machine learning" and "which is
#     used by global taxi giant Uber" are separate pieces, and the one
#     piece alone rarely looks like the answer.
#   - A long answer mentions several things, so the most similar piece
#     can be about a side topic ("What is machine learning?" jumped to
#     "Hence, supervised learning uses labeled data").
#
# Here every 2 consecutive pieces are one candidate, scored half on
# the QUESTION and half on the answer sentence it matches best.
# Right place: 13 of 18 before, 15 of 18 with this. The text shown is
# also a complete thought instead of half a sentence.

EVIDENCE_PIECES = 2

# Pieces at most this far apart (seconds) are consecutive speech.
EVIDENCE_PIECE_GAP = 1.0

# Answer sentences shorter than this are not used to choose evidence.
MIN_ANSWER_SENTENCE_WORDS = 4

# Openers that are not the real first sentence of an answer.
_ANSWER_OPENER = re.compile(
    r"^(here'?s the answer|according to the video)[:.,]?\s*$",
    flags=re.IGNORECASE
)


def _answer_sentences(answer):

    parts = re.split(
        r"(?<=[.!?:])\s+|\n+",
        str(answer)
    )

    sentences = [
        part.strip(" -*\t")
        for part in parts
    ]

    return [
        sentence
        for sentence in sentences
        if len(sentence.split()) >= MIN_ANSWER_SENTENCE_WORDS
        and not _ANSWER_OPENER.match(sentence)
    ]


def _evidence_candidates(results, pieces):

    # Every piece once, in the order of the video.
    seen = set()
    ordered = []

    for result in results:

        for segment in result.get(
            "segments",
            []
        ):

            text = segment.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            key = (
                float(segment["start"]),
                text
            )

            if key in seen:
                continue

            seen.add(key)

            ordered.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text
            })

    ordered.sort(
        key=lambda piece: piece["start"]
    )

    groups = []

    for first in range(len(ordered) - pieces + 1):

        run = ordered[first:first + pieces]

        consecutive = all(
            later["start"] - earlier["end"] <= EVIDENCE_PIECE_GAP
            for earlier, later in zip(run, run[1:])
        )

        if consecutive:

            groups.append({
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "text": " ".join(
                    piece["text"]
                    for piece in run
                )
            })

    # Too few consecutive pieces: single pieces, as before.
    return groups or ordered


def find_evidence(
    results,
    question,
    answer,
    embedding_model,
    pieces=EVIDENCE_PIECES
):

    if not results or not answer:
        return []

    candidates = _evidence_candidates(
        results,
        pieces
    )

    if not candidates:
        return []

    sentences = (
        _answer_sentences(answer)
        or [str(answer)]
    )

    vectors = np.asarray(
        embedding_model.encode(
            [candidate["text"] for candidate in candidates],
            normalize_embeddings=True
        ),
        dtype="float32"
    )

    query_vectors = np.asarray(
        embedding_model.encode(
            [str(question)] + sentences,
            normalize_embeddings=True
        ),
        dtype="float32"
    )

    question_similarity = vectors @ query_vectors[0]

    # For each candidate: the answer sentence it matches best.
    answer_similarity = np.max(
        vectors @ query_vectors[1:].T,
        axis=1
    )

    scores = (
        0.5 * question_similarity
        + 0.5 * answer_similarity
    )

    best = int(np.argmax(scores))

    evidence = dict(candidates[best])
    evidence["score"] = float(scores[best])

    return [evidence]
