import ollama
import whisper
from sentence_transformers import SentenceTransformer

from src.processing.chunker import create_chunks
from src.rag.vector_store import create_index, retrieve_chunks


def prepare_context(results):
    context = ""

    for result in results:
        context += (
            f"[{result['start']} - {result['end']}]\n"
            f"{result['text']}\n\n"
        )

    return context


def generate_answer(context, question):
    prompt = f"""
Answer the question using only the information provided in the context.

If the answer is not present in the context, say:
"I don't have enough information in the video."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


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

    question = input("\nEnter your question: ")

    print("\nSearching for relevant chunks...")
    query_embedding = embedding_model.encode(question)

    results = retrieve_chunks(
        index,
        chunks,
        query_embedding
    )

    context = prepare_context(results)

    answer = generate_answer(
        context,
        question
    )

    print("\nGenerated answer:")
    print(answer)