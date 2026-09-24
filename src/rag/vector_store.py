import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def create_index(embeddings):
    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def search_index(index, query_embedding, k=3):
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, k)

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

    for distance, i in zip(distances[0], indices[0]):

        if i == -1:
            continue

        if distance <= threshold:

            results.append({
                "score": float(distance),
                "start": chunks[i]["start"],
                "end": chunks[i]["end"],
                "text": chunks[i]["text"]
            })

    return results