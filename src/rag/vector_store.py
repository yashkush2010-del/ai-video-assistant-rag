import faiss
import numpy as np


def create_index(embeddings):
    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def search_index(index, query_embedding, k=3):
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(
        query_vector,
        k
    )

    return distances, indices


def retrieve_chunks(
    index,
    chunks,
    query_embedding,
    k=3,
    threshold=1.0
):
    distances, indices = search_index(
        index,
        query_embedding,
        k
    )

    results = []

    for distance, i in zip(
        distances[0],
        indices[0]
    ):

        if i == -1:
            continue

        if distance <= threshold:

            results.append({
                "score": float(distance),
                "start": chunks[i]["start"],
                "end": chunks[i]["end"],
                "text": chunks[i]["text"],
                "segments": chunks[i]["segments"]
            })

    return results


def find_relevant_segments(
    results,
    query_embedding,
    embedding_model,
    max_segments=2
):
    relevant_segments = []

    for result in results:

        segments = result["segments"]

        segment_texts = [
            segment["text"].strip()
            for segment in segments
        ]

        segment_embeddings = embedding_model.encode(
            segment_texts
        )

        similarities = []

        for segment_embedding in segment_embeddings:

            distance = np.linalg.norm(
                query_embedding - segment_embedding
            )

            similarities.append(distance)

        ranked_indices = np.argsort(
            similarities
        )

        for index in ranked_indices[:max_segments]:

            segment = segments[index]

            relevant_segments.append({
                "score": float(similarities[index]),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })

    relevant_segments.sort(
        key=lambda x: x["score"]
    )

    return relevant_segments[:max_segments]
def find_answer_segments(
    results,
    answer,
    embedding_model,
    max_segments=2
):
    answer_embedding = embedding_model.encode(
        answer
    )

    all_segments = []

    for result in results:

        for segment in result["segments"]:

            all_segments.append(segment)

    segment_texts = [
        segment["text"].strip()
        for segment in all_segments
    ]

    segment_embeddings = embedding_model.encode(
        segment_texts
    )

    scored_segments = []

    for segment, segment_embedding in zip(
        all_segments,
        segment_embeddings
    ):

        distance = np.linalg.norm(
            answer_embedding - segment_embedding
        )

        scored_segments.append({
            "score": float(distance),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    scored_segments.sort(
        key=lambda x: x["score"]
    )

    return scored_segments[:max_segments]