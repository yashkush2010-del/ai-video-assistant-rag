
import os
import json
import hashlib

import streamlit as st
import whisper

from sentence_transformers import SentenceTransformer

from src.ingestion.audio_extractor import extract_audio
from src.processing.chunker import create_chunks

from src.processing.keyframe_extractor import (
    extract_keyframes,
    match_keyframes_with_transcript,
    create_multimodal_records
)

from src.rag.vector_store import (
    create_index,
    retrieve_chunks,
    find_answer_segments
)

from src.rag.answer_generator import (
    prepare_context,
    generate_answer
)


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

    # Process video only when a new video is uploaded
    if st.session_state.get("video_hash") != video_hash:

        # Create required directories
        os.makedirs("data/videos", exist_ok=True)
        os.makedirs("data/audio", exist_ok=True)
        os.makedirs("data/keyframes", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        video_path = os.path.join(
            "data",
            "videos",
            uploaded_video.name
        )

        with open(video_path, "wb") as file:
            file.write(video_data)

        st.session_state.video_hash = video_hash

        st.success("Video uploaded successfully!")

        # Extract audio
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

        # Load models
        whisper_model, embedding_model = load_models()

        # Transcription
        with st.spinner("Transcribing video..."):

            result = whisper_model.transcribe(
                audio_path
            )

        segments = result["segments"]

        st.success("Transcription completed!")

        # Extract keyframes
        with st.spinner("Extracting video keyframes..."):

            video_name = os.path.splitext(
                uploaded_video.name
            )[0]

            keyframe_dir = os.path.join(
                "data",
                "keyframes",
                video_name
            )

            keyframes = extract_keyframes(
                video_path,
                keyframe_dir,
                interval=30
            )

        st.success(
            f"Extracted {len(keyframes)} keyframes!"
        )

        # Match keyframes with transcript
        matched_frames = match_keyframes_with_transcript(
            keyframes,
            segments,
            tolerance=2.0
        )

        st.success(
            f"Matched {len(matched_frames)} keyframes with transcript segments!"
        )

        # Create structured multimodal records
        multimodal_records = create_multimodal_records(
            matched_frames
        )

        # Save records as JSON
        records_path = os.path.join(
            "data",
            "processed",
            f"{video_name}_multimodal_records.json"
        )

        with open(
            records_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                multimodal_records,
                file,
                indent=4,
                ensure_ascii=False
            )

        st.success(
            "Multimodal records saved successfully!"
        )

        # Create chunks
        chunks = create_chunks(
            segments
        )

        # Create embeddings
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_model.encode(
            texts
        )

        # Create FAISS index
        index = create_index(
            embeddings
        )

        # Store everything in session state
        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.embedding_model = embedding_model

        st.session_state.multimodal_records = multimodal_records

        st.session_state.keyframes = keyframes

        st.session_state.segments = segments

    else:

        st.success("Video uploaded successfully!")

    # Show video
    st.video(uploaded_video)

    # Display multimodal records
    if "multimodal_records" in st.session_state:

        records = st.session_state.multimodal_records

        st.write("### Multimodal Record Preview")

        st.write(
            f"Total structured records: {len(records)}"
        )

        if records:

            st.json(records[0])

            st.write("### Keyframe Preview")

            first_frame = records[0]

            if os.path.exists(first_frame["image_path"]):

                st.image(
                    first_frame["image_path"],
                    caption=(
                        f"Timestamp: "
                        f"{first_frame['timestamp']} seconds"
                    )
                )

            st.write("### Associated Transcript")

            st.write(
                first_frame["transcript_text"]
            )

        else:

            st.info(
                "No keyframes matched the transcript within the allowed tolerance."
            )

    # Question answering
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

                # Convert question into embedding
                query_embedding = embedding_model.encode(
                    question
                )

                # Retrieve relevant chunks
                results = retrieve_chunks(
                    index,
                    chunks,
                    query_embedding,
                    threshold=1.0
                )

                # Prepare context
                context = prepare_context(
                    results
                )

                # Generate answer using Llama
                answer = generate_answer(
                    context,
                    question
                )

                st.write(answer)

                # Check if video contains the answer
                if "don't have enough information" in answer.lower():

                    st.write("### Relevant timestamps")

                    st.info(
                        "No relevant information found in the video."
                    )

                else:

                    # Find precise evidence segments
                    evidence_segments = find_answer_segments(
                        results,
                        answer,
                        embedding_model,
                        max_segments=1
                    )

                    st.write("### Relevant timestamps")

                    for segment in evidence_segments:

                        st.write(
                            f"**[{segment['start']}s - "
                            f"{segment['end']}s]**"
                        )

                        st.write(
                            segment["text"]
                        )

            else:

                st.warning(
                    "Please enter a question."
                )