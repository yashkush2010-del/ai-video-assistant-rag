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


# ============================================================
# WHEN OLLAMA CANNOT ANSWER
# ============================================================
#
# If Ollama is not running, ollama.chat() raises a connection error
# ("Failed to connect to Ollama ..."), and if the model has not been
# downloaded it raises ollama.ResponseError (404). Without handling,
# that error reached Streamlit as a traceback. It is turned into
# LLMUnavailableError with a message the user can act on. No answer
# is made up: the caller decides what to show.

LLM_MODEL = "llama3.2"

OLLAMA_NOT_RUNNING_MESSAGE = (
    "The language model is not available: Ollama does not seem to be "
    "running, so this question could not be answered. Start Ollama "
    "(open the Ollama app, or run \"ollama serve\" in a terminal) and "
    "ask the question again."
)

MODEL_NOT_FOUND_MESSAGE = (
    f"The language model \"{LLM_MODEL}\" is not installed in Ollama, so "
    f"this question could not be answered. Download it with "
    f"\"ollama pull {LLM_MODEL}\" and ask the question again."
)


class LLMUnavailableError(Exception):
    """Ollama could not produce an answer. str(error) is shown to the user."""


def _connection_errors():
    """Errors that mean Ollama could not be reached."""
    errors = [ConnectionError, TimeoutError, OSError]

    # Older ollama versions let httpx's own errors through.
    try:
        import httpx
        errors.append(httpx.TransportError)
    except ImportError:
        pass

    return tuple(errors)


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

    response_error = getattr(ollama, "ResponseError", None)

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    except _connection_errors() as error:
        raise LLMUnavailableError(OLLAMA_NOT_RUNNING_MESSAGE) from error

    except Exception as error:
        # Only Ollama's own response errors are expected here; anything
        # else is a real bug and is not hidden.
        if response_error is None or not isinstance(error, response_error):
            raise

        if (
            getattr(error, "status_code", None) == 404
            or "not found" in str(error).lower()
        ):
            raise LLMUnavailableError(MODEL_NOT_FOUND_MESSAGE) from error

        raise LLMUnavailableError(
            f"The language model could not answer this question "
            f"(Ollama said: {error}). Please try again."
        ) from error

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
    
    print("\n--- CONTEXT SENT TO LLM ---")
    print(context)
    print("---------------------------")
    
    answer = generate_answer(
        context,
        question
    )

    print("\nGenerated answer:")
    print(answer)

    print("\nRelevant timestamps:")

    shown_timestamps = set()

    for result in results:
        timestamp = (result["start"], result["end"])

        if timestamp not in shown_timestamps:
           print(f"\n[{result['start']}s - {result['end']}s]")
        print(result["text"])
        shown_timestamps.add(timestamp)