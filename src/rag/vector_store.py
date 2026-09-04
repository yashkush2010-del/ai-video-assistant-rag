import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.processing.chunker import create_chunks
import whisper


def create_index(embeddings):
    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index


if __name__ == "__main__":

    print("Transcribing audio...")
    whisper_model = whisper.load_model("base")

    result = whisper_model.transcribe("data/audio/sample.wav")
    segments = result["segments"]

    print("Creating chunks...")
    chunks = create_chunks(segments)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts)

    print("Creating FAISS index...")
    index = create_index(embeddings)

    print("\nFAISS index created successfully!")
    print("Number of vectors:", index.ntotal)
    print("Vector dimension:", index.d)
