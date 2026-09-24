import os
import hashlib

import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer

from src.ingestion.audio_extractor import extract_audio
from src.processing.chunker import create_chunks
from src.rag.vector_store import create_index, retrieve_chunks
from src.rag.answer_generator import prepare_context, generate_answer


st.title("AI Video Assistant")

st.write("Ask questions about your video using RAG.")


@st.cache_resource
def load_models():

    whisper_model = whisper.load_model("base")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return whisper_model, embedding_model


uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"]
)


if uploaded_video is not None:

    video_data = uploaded_video.getvalue()

    video_hash = hashlib.md5(video_data).hexdigest()

    if st.session_state.get("video_hash") != video_hash:

        video_path = os.path.join(
            "data",
            "videos",
            uploaded_video.name
        )

        with open(video_path, "wb") as file:
            file.write(video_data)

        st.session_state.video_hash = video_hash

        st.success("Video uploaded successfully!")

        audio_path = os.path.join(
            "data",
            "audio",
            "sample.wav"
        )

        extract_audio(
            video_path,
            audio_path
        )

        st.success("Audio extracted successfully!")

        whisper_model, embedding_model = load_models()

        with st.spinner("Transcribing video..."):

            result = whisper_model.transcribe(
                audio_path
            )

        segments = result["segments"]

        st.success("Transcription completed!")

        chunks = create_chunks(segments)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_model.encode(
            texts
        )

        index = create_index(
            embeddings
        )

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.embedding_model = embedding_model

    else:

        st.success("Video uploaded successfully!")


    st.video(uploaded_video)


    if "index" in st.session_state:

        question = st.text_input(
            "Enter your question:"
        )


        if st.button("Ask Question"):

            if question:

                st.write("### Answer")

                embedding_model = (
                    st.session_state.embedding_model
                )

                index = st.session_state.index

                chunks = st.session_state.chunks

                query_embedding = embedding_model.encode(
                    question
                )

                results = retrieve_chunks(
                    index,
                    chunks,
                    query_embedding,
                    threshold=1.0
                )

                if not results:

                    st.info(
                        "I don't have enough information "
                        "in the video to answer this question."
                    )

                    st.write("### Relevant timestamps")

                    st.info(
                        "No relevant information found in the video."
                    )

                else:

                    context = prepare_context(
                        results
                    )

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
                                f"**[{result['start']}s - "
                                f"{result['end']}s]**"
                            )

                            st.write(
                                result["text"]
                            )

                            shown_timestamps.add(
                                timestamp
                            )

            else:

                st.warning(
                    "Please enter a question."
                )