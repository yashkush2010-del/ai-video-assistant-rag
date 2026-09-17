import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer

from src.processing.chunker import create_chunks
from src.rag.vector_store import create_index, retrieve_chunks
from src.rag.answer_generator import prepare_context, generate_answer


st.title("AI Video Assistant")

st.write("Ask questions about your video using RAG.")


@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return whisper_model, embedding_model


@st.cache_resource
def prepare_video():
    whisper_model, embedding_model = load_models()

    result = whisper_model.transcribe("data/audio/sample.wav")
    segments = result["segments"]

    chunks = create_chunks(segments)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts)

    index = create_index(embeddings)

    return embedding_model, index, chunks


embedding_model, index, chunks = prepare_video()


question = st.text_input("Enter your question:")


if st.button("Ask Question"):

    if question:

        st.write("### Answer")

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

        st.write(answer)

        st.write("### Relevant timestamps")

        shown_timestamps = set()

        for result in results:

            timestamp = (
                result["start"],
                result["end"]
            )

            if timestamp not in shown_timestamps:

                st.write(
                    f"**[{result['start']}s - {result['end']}s]**"
                )

                st.write(result["text"])

                shown_timestamps.add(timestamp)

    else:

        st.warning("Please enter a question.")