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