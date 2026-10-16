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
    create_multimodal_records,
)

from src.rag.vector_store import (
    create_index,
    retrieve_chunks,
    find_answer_segments,
)

from src.rag.answer_generator import (
    prepare_context,
    generate_answer,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🎥 AI Video Assistant")

st.write(
    "We know humans learn from their past experiences"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    whisper_model = whisper.load_model("base")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return whisper_model, embedding_model


# ============================================================
# SESSION STATE
# ============================================================

if "video_hash" not in st.session_state:
    st.session_state.video_hash = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_name" not in st.session_state:
    st.session_state.video_name = None

if "video_data" not in st.session_state:
    st.session_state.video_data = None

if "segments" not in st.session_state:
    st.session_state.segments = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "keyframes" not in st.session_state:
    st.session_state.keyframes = None

if "multimodal_records" not in st.session_state:
    st.session_state.multimodal_records = None

if "question" not in st.session_state:
    st.session_state.question = ""

if "answer" not in st.session_state:
    st.session_state.answer = None

if "evidence_segments" not in st.session_state:
    st.session_state.evidence_segments = []

# ------------------------------------------------------------
# This controls the timestamp at which the video starts.
# ------------------------------------------------------------

if "jump_time" not in st.session_state:
    st.session_state.jump_time = 0


# ============================================================
# JUMP CALLBACK
# ============================================================

def jump_to_timestamp(timestamp):

    st.session_state.jump_time = int(timestamp)


# ============================================================
# UPLOAD VIDEO
# ============================================================

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv"],
)


# ============================================================
# PROCESS VIDEO
# ============================================================

if uploaded_video is not None:

    video_data = uploaded_video.getvalue()

    video_hash = hashlib.md5(
        video_data
    ).hexdigest()


    # ========================================================
    # ONLY PROCESS A NEW VIDEO
    # ========================================================

    if st.session_state.video_hash != video_hash:

        # ----------------------------------------------------
        # DIRECTORIES
        # ----------------------------------------------------

        os.makedirs(
            "data/videos",
            exist_ok=True
        )

        os.makedirs(
            "data/audio",
            exist_ok=True
        )

        os.makedirs(
            "data/keyframes",
            exist_ok=True
        )

        os.makedirs(
            "data/processed",
            exist_ok=True
        )


        # ----------------------------------------------------
        # SAVE VIDEO
        # ----------------------------------------------------

        video_path = os.path.join(
            "data",
            "videos",
            uploaded_video.name
        )


        with open(
            video_path,
            "wb"
        ) as file:

            file.write(video_data)


        # ----------------------------------------------------
        # SAVE SESSION INFORMATION
        # ----------------------------------------------------

        st.session_state.video_hash = video_hash

        st.session_state.video_path = video_path

        st.session_state.video_name = os.path.splitext(
            uploaded_video.name
        )[0]

        st.session_state.video_data = video_data

        st.session_state.jump_time = 0

        st.session_state.question = ""

        st.session_state.answer = None

        st.session_state.evidence_segments = []


        # ====================================================
        # AUDIO
        # ====================================================

        audio_path = os.path.join(
            "data",
            "audio",
            "sample.wav"
        )


        with st.spinner(
            "Extracting audio..."
        ):

            extract_audio(
                video_path,
                audio_path
            )


        st.success(
            "Audio extracted successfully!"
        )


        # ====================================================
        # MODELS
        # ====================================================

        with st.spinner(
            "Loading AI models..."
        ):

            whisper_model, embedding_model = (
                load_models()
            )


        # ====================================================
        # TRANSCRIPTION
        # ====================================================

        with st.spinner(
            "Transcribing video..."
        ):

            result = whisper_model.transcribe(
                audio_path
            )


        segments = result["segments"]

        st.session_state.segments = segments


        st.success(
            "Transcription completed!"
        )


        # ====================================================
        # KEYFRAMES
        # ====================================================

        with st.spinner(
            "Extracting keyframes..."
        ):

            video_name = (
                st.session_state.video_name
            )

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


        st.session_state.keyframes = keyframes


        st.success(
            f"Extracted {len(keyframes)} keyframes!"
        )


        # ====================================================
        # MATCH KEYFRAMES WITH TRANSCRIPT
        # ====================================================

        with st.spinner(
            "Matching keyframes with transcript..."
        ):

            matched_frames = (
                match_keyframes_with_transcript(
                    keyframes,
                    segments,
                    tolerance=2.0
                )
            )


        # ====================================================
        # MULTIMODAL RECORDS
        # ====================================================

        multimodal_records = (
            create_multimodal_records(
                matched_frames
            )
        )


        st.session_state.multimodal_records = (
            multimodal_records
        )


        # ----------------------------------------------------
        # SAVE RECORDS
        # ----------------------------------------------------

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


        # ====================================================
        # CREATE CHUNKS
        # ====================================================

        chunks = create_chunks(
            segments
        )


        # ====================================================
        # EMBEDDINGS
        # ====================================================

        with st.spinner(
            "Creating embeddings..."
        ):

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = embedding_model.encode(
                texts
            )


        # ====================================================
        # FAISS INDEX
        # ====================================================

        index = create_index(
            embeddings
        )


        # ====================================================
        # SAVE RAG STATE
        # ====================================================

        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.embedding_model = (
            embedding_model
        )


        st.success(
            "Video processing completed!"
        )


# ============================================================
# INITIAL VIDEO
#
# Only show the normal video here when there is no answer yet.
# Once the user asks a question, the video moves directly
# underneath the timestamp section.
# ============================================================

if (
    st.session_state.video_data is not None
    and st.session_state.answer is None
):

    st.markdown(
        "## 🎬 Video"
    )

    st.video(
        st.session_state.video_data,
        format="video/mp4",
        start_time=st.session_state.jump_time,
    )


# ============================================================
# MULTIMODAL RECORD PREVIEW
# ============================================================

if (
    st.session_state.multimodal_records
    is not None
    and st.session_state.answer is None
):

    records = (
        st.session_state.multimodal_records
    )


    st.markdown(
        "## 🖼️ Multimodal Record Preview"
    )


    st.write(
        f"Total structured records: "
        f"{len(records)}"
    )


    if records:

        with st.expander(
            "View first multimodal record"
        ):

            st.json(
                records[0]
            )


        st.markdown(
            "### 🖼️ Keyframe Preview"
        )


        first_record = records[0]


        if os.path.exists(
            first_record["image_path"]
        ):

            st.image(
                first_record["image_path"],
                caption=(
                    "Timestamp: "
                    f"{first_record['timestamp']} seconds"
                ),
                use_container_width=True,
            )


# ============================================================
# QUESTION SECTION
# ============================================================

if st.session_state.index is not None:

    st.markdown(
        "## 💬 Ask Questions About Your Video"
    )


    question = st.text_input(
        "Enter your question:",
        value=st.session_state.question,
        placeholder=(
            "Example: What is mentioned "
            "about machine learning?"
        ),
    )


    ask_button = st.button(
        "🔍 Ask Question",
        key="ask_question",
    )


    # ========================================================
    # ASK QUESTION
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            st.session_state.question = question


            # ------------------------------------------------
            # RESET OLD RESULTS
            # ------------------------------------------------

            st.session_state.answer = None

            st.session_state.evidence_segments = []

            # Start video from beginning for a new question
            st.session_state.jump_time = 0


            # ------------------------------------------------
            # GET RAG DATA
            # ------------------------------------------------

            embedding_model = (
                st.session_state.embedding_model
            )

            index = (
                st.session_state.index
            )

            chunks = (
                st.session_state.chunks
            )


            # ------------------------------------------------
            # QUESTION EMBEDDING
            # ------------------------------------------------

            query_embedding = (
                embedding_model.encode(
                    question
                )
            )


            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            results = retrieve_chunks(
                index,
                chunks,
                query_embedding,
                threshold=2.0,
            )


            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            context = prepare_context(
                results
            )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    context,
                    question
                )


            st.session_state.answer = answer


            # =================================================
            # FIND EVIDENCE
            # =================================================

            if (
                "don't have enough information"
                not in answer.lower()
            ):

                evidence_segments = (
                    find_answer_segments(
                        results,
                        answer,
                        embedding_model,
                        max_segments=1,
                    )
                )


                st.session_state.evidence_segments = (
                    evidence_segments
                )


# ============================================================
# DISPLAY ANSWER
# ============================================================

if st.session_state.answer is not None:

    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        "## 🧠 Answer"
    )


    st.write(
        st.session_state.answer
    )


    # ========================================================
    # TIMESTAMPS
    # ========================================================

    st.markdown(
        "## ⏱️ Relevant Timestamps"
    )


    evidence_segments = (
        st.session_state.evidence_segments
    )


    if not evidence_segments:

        st.info(
            "No precise evidence segment was found."
        )


    else:

        # ====================================================
        # EVIDENCE
        # ====================================================

        for i, segment in enumerate(
            evidence_segments
        ):

            start_time = int(
                segment["start"]
            )

            end_time = int(
                segment["end"]
            )


            st.markdown(
                f"### ⏱️ "
                f"[{segment['start']:.1f}s - "
                f"{segment['end']:.1f}s]"
            )


            st.write(
                segment["text"]
            )


            # =================================================
            # JUMP BUTTON
            # =================================================

            st.button(
                f"▶️ Jump to {start_time}s",
                key=f"jump_{i}_{start_time}",
                on_click=jump_to_timestamp,
                args=(start_time,),
            )


            st.caption(
                f"Video evidence starts at approximately "
                f"{start_time} seconds and ends at "
                f"{end_time} seconds."
            )


    # ========================================================
    # THIS IS THE IMPORTANT PART
    #
    # The video is now immediately BELOW the timestamp.
    #
    # We do NOT put it at the top of the page.
    # ========================================================

    st.markdown(
        "## 🎬 Video Evidence"
    )


    current_time = int(
        st.session_state.jump_time
    )


    st.video(
        st.session_state.video_data,
        format="video/mp4",
        start_time=current_time,
    )


    # ========================================================
    # SHOW CURRENT TIMESTAMP
    # ========================================================

    if current_time > 0:

        st.success(
            f"🎯 Video opened at approximately "
            f"{current_time} seconds."
        )


        # ----------------------------------------------------
        # RETURN TO BEGINNING
        # ----------------------------------------------------

        if st.button(
            "↩️ Return Video to Beginning",
            key="return_video",
        ):

            st.session_state.jump_time = 0

            st.rerun()


    # ========================================================
    # RELATED KEYFRAME
    # ========================================================

    records = (
        st.session_state.multimodal_records
    )


    related_keyframe = None


    if records and evidence_segments:

        target_time = float(
            evidence_segments[0]["start"]
        )


        closest_distance = float(
            "inf"
        )


        for record in records:

            try:

                record_time = float(
                    record["timestamp"]
                )


                distance = abs(
                    record_time - target_time
                )


                if distance < closest_distance:

                    closest_distance = distance

                    related_keyframe = record


            except (
                KeyError,
                ValueError,
                TypeError,
            ):

                continue


    # ========================================================
    # KEYFRAME
    # ========================================================

    if related_keyframe:

        st.markdown(
            "## 🖼️ Related Keyframe"
        )


        if os.path.exists(
            related_keyframe["image_path"]
        ):

            st.image(
                related_keyframe["image_path"],
                caption=(
                    "Keyframe at "
                    f"{related_keyframe['timestamp']} "
                    "seconds"
                ),
                use_container_width=True,
            )


        st.markdown(
            "### 📝 Related Transcript"
        )


        st.write(
            related_keyframe["transcript_text"]
        )