import ollama
import whisper
from sentence_transformers import SentenceTransformer

from src.processing.chunker import create_chunks
from src.rag.vector_store import create_index, retrieve_chunks


# ============================================================
# CONTEXT FOR THE LANGUAGE MODEL
# ============================================================
#
# The transcript parts are given in the order of the video, with
# readable times ([2:23 - 2:41] instead of [143.0 - 161.4]). Parts that
# follow each other directly are joined into one passage, so a
# sentence that was cut between two chunks reads as one sentence again.
# A small model connects facts much better this way than when the
# parts are listed by search score.

# Two chunks closer than this (seconds) are one continuous passage.
JOIN_GAP = 0.5


def _clock(seconds):
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"


def prepare_context(results):
    passages = []

    for result in sorted(results, key=lambda item: float(item["start"])):
        start = float(result["start"])
        end = float(result["end"])
        text = str(result["text"]).strip()

        if passages and start - passages[-1]["end"] <= JOIN_GAP:
            passages[-1]["end"] = max(passages[-1]["end"], end)
            passages[-1]["text"] += " " + text
        else:
            passages.append({"start": start, "end": end, "text": text})

    context = ""

    for passage in passages:
        context += (
            f"[{_clock(passage['start'])} - {_clock(passage['end'])}]\n"
            f"{passage['text']}\n\n"
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


def model_not_found_message(model):
    """MODEL_NOT_FOUND_MESSAGE for any model (e.g. the larger one)."""
    return MODEL_NOT_FOUND_MESSAGE.replace(LLM_MODEL, model)


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


# ============================================================
# ANSWER
# ============================================================
#
# The earlier prompt only said "answer using only the context". With
# llama3.2 that gave answers that repeated one sentence containing the
# question's words ("Machine learning is a lot more than just
# learning"), left out facts that were in the context, or added things
# that were not. The prompt now asks for a complete answer from the
# transcript only.
#
# Measured with scripts/check_qa_real_video.py on two videos: a
# stricter "at most 4 sentences, only this question" version made
# llama3.2 leave out facts it had been given (8.5 instead of 10 of 10
# on the Internet video) and saved only 1-3 seconds, so this wording
# is kept. Answers take longer than before (about 15 s instead of
# 3-10 s on a laptop CPU) because the model reads more transcript.
#
# When the search found nothing, the model is not asked at all: with
# an empty context it replied "Go ahead and ask your question" instead
# of saying that the video has no answer.
#
# The app recognises "no answer" by the sentence NO_INFORMATION_ANSWER,
# so the prompt asks for exactly that sentence.
#
# Temperature 0: the same question about the same video always gets
# the same answer. num_ctx: room for up to MAX_CONTEXT_CHUNKS chunks
# (vector_store.py) plus the prompt.

NO_INFORMATION_ANSWER = "I don't have enough information in the video."

ANSWER_OPTIONS = {
    "temperature": 0,
    "num_ctx": 4096
}

ANSWER_PROMPT = """You answer a question about a video. You only have parts of the video's transcript, given below in the order they are spoken.

Transcript parts:
__CONTEXT__
Question: __QUESTION__

How to answer:
- Use only what the transcript parts say. Do not add facts, examples or items that are not in them.
- Give a complete answer: include every reason, factor, step or example the transcript gives for this question, even when it is spread over several parts.
- Explain it in your own words in 2 to 4 sentences, or as a short list when the question asks for several things. Do not just repeat a sentence that contains the words of the question.
- Talk about the video, not about "the context" or "the transcript".
- If the transcript parts do not answer the question, reply with exactly this sentence and nothing else:
__NO_INFORMATION__

Answer:"""


def generate_answer(context, question, model=LLM_MODEL):
    # Nothing in the video matched the question: there is nothing to
    # answer from, so the answer is known without asking the model.
    if not str(context or "").strip():
        return NO_INFORMATION_ANSWER

    # Plain replace (not str.format): a transcript can contain { }.
    prompt = (
        ANSWER_PROMPT
        .replace("__CONTEXT__", context)
        .replace("__QUESTION__", str(question).strip())
        .replace("__NO_INFORMATION__", NO_INFORMATION_ANSWER)
    )

    response_error = getattr(ollama, "ResponseError", None)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options=ANSWER_OPTIONS
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
            raise LLMUnavailableError(
                model_not_found_message(model)
            ) from error

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