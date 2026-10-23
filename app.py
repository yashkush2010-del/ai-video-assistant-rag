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

from src.rag.video_summary import (
    generate_video_summary,
    escape_markdown,
    format_timestamp,
    format_time_range
)

from src.ingestion.youtube_loader import (
    DEFAULT_CACHE_DIR as YOUTUBE_CACHE_DIR,
    YouTubeError,
    YouTubeVideoSource,
    describe_progress,
    fetch_youtube_video
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎥 AI Video Assistant")

st.write(
    "Upload a video and ask questions about its content using "
    "RAG, timestamped transcripts, and keyframes."
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    whisper_model = whisper.load_model(
        "base"
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return (
        whisper_model,
        embedding_model
    )


# ============================================================
# VIDEO SEEK HELPER
# ============================================================
#
# WHY THIS EXISTS
# ---------------
# st.video(..., start_time=X) only sets the INITIAL playback
# position of the player. It is a declarative argument, not a
# command. Two things follow from that:
#
#   1. If the value is the same as in the previous run (for
#      example the user scrubbed the player and then clicks the
#      same "Jump to Xs" button again), the browser receives no
#      change and the player is NOT moved.
#
#   2. The player is rendered at the TOP of the page while the
#      Jump buttons are far below it, so even when the player
#      does move, the user never sees or hears it happen.
#
# This helper sends an explicit command to the real <video>
# element in the browser: seek to the requested time, bring the
# player into view and start playback.
#
# st.video() stays exactly as it is. This is only added on top.

# Start playing after a jump (browsers may block this silently).
AUTOPLAY_AFTER_JUMP = True

# Scroll the page so the player is visible after a jump.
SCROLL_TO_PLAYER_ON_JUMP = True


def seek_video_player(seconds, request_id):

    script = """
<script>
(function () {
    // request id: __REQUEST_ID__
    // This number changes on every click, so the iframe content
    // is different every time and the script runs again even
    // when the same timestamp is requested twice.

    var TARGET = __SECONDS__;
    var AUTOPLAY = __AUTOPLAY__;
    var SCROLL = __SCROLL__;
    var attempts = 0;

    function findVideo() {
        var doc = window.parent.document;
        return (
            doc.querySelector('video[data-testid="stVideo"]') ||
            doc.querySelector('video')
        );
    }

    function seek(video) {

        function apply() {
            video.currentTime = TARGET;

            if (SCROLL) {
                video.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
            }

            if (AUTOPLAY) {
                var playing = video.play();
                if (playing && playing.catch) {
                    playing.catch(function () {});
                }
            }
        }

        // readyState 0 = no metadata yet, seeking has to wait.
        if (video.readyState >= 1) {
            apply();
        } else {
            video.addEventListener(
                "loadedmetadata",
                apply,
                { once: true }
            );
        }
    }

    function waitForVideo() {
        try {
            var video = findVideo();

            if (video) {
                seek(video);
                return;
            }
        } catch (error) {
            console.error("Video seek failed:", error);
            return;
        }

        attempts += 1;

        if (attempts < 50) {
            setTimeout(waitForVideo, 100);
        }
    }

    waitForVideo();
})();
</script>
"""

    script = (
        script
        .replace("__REQUEST_ID__", str(int(request_id)))
        .replace("__SECONDS__", str(float(seconds)))
        .replace("__AUTOPLAY__", "true" if AUTOPLAY_AFTER_JUMP else "false")
        .replace("__SCROLL__", "true" if SCROLL_TO_PLAYER_ON_JUMP else "false")
        .strip()
    )

    # Newer Streamlit versions provide st.iframe and mark
    # components.v1.html as deprecated (it would show a warning
    # banner in the app). Older versions only have components.html.
    if hasattr(st, "iframe"):
        st.iframe(script, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(script, height=0)


# ============================================================
# JUMP REQUEST HELPER
# ============================================================
#
# Performs exactly the same three state updates as the Jump
# buttons in the evidence section. The actual seeking is done
# by seek_video_player() at the end of the script, so every
# Jump button shares one working seek/autoplay mechanism.

def request_video_jump(start_time):

    st.session_state.jump_time = start_time

    st.session_state.seek_counter += 1

    st.session_state.seek_request = {
        "time": start_time,
        "id": st.session_state.seek_counter,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "video_hash" not in st.session_state:
    st.session_state.video_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "multimodal_records" not in st.session_state:
    st.session_state.multimodal_records = None

if "keyframes" not in st.session_state:
    st.session_state.keyframes = None

if "segments" not in st.session_state:
    st.session_state.segments = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_name" not in st.session_state:
    st.session_state.video_name = None

if "video_data" not in st.session_state:
    st.session_state.video_data = None

# Timestamp used by the video player
if "jump_time" not in st.session_state:
    st.session_state.jump_time = None

# One-shot command for the browser: seek the real <video> element.
# Holds {"time": seconds, "id": counter} until it has been sent.
if "seek_request" not in st.session_state:
    st.session_state.seek_request = None

if "seek_counter" not in st.session_state:
    st.session_state.seek_counter = 0

# Cached video summary: {"video_hash": ..., "summary": {...}}
if "video_summary" not in st.session_state:
    st.session_state.video_summary = None

# True for one rerun after a summary Jump, so the summary
# section stays open instead of collapsing.
if "summary_open" not in st.session_state:
    st.session_state.summary_open = False

# YouTube input: the downloaded video, which source (upload or YouTube)
# the user used last, the last uploaded file, and the last error shown.
if "youtube_source" not in st.session_state:
    st.session_state.youtube_source = None

if "active_source" not in st.session_state:
    st.session_state.active_source = None

if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

if "youtube_error" not in st.session_state:
    st.session_state.youtube_error = None

# Store the last question/answer so a jump rerun
# does not remove the previous result
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_evidence_segments" not in st.session_state:
    st.session_state.last_evidence_segments = []

if "last_related_keyframe" not in st.session_state:
    st.session_state.last_related_keyframe = None


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)


# ============================================================
# YOUTUBE URL  (an alternative to the upload above)
# ============================================================
#
# A YouTube link is turned into a small local video file by
# src/ingestion/youtube_loader.py. From there it goes through exactly
# the same processing as an uploaded video: the block below only sees
# a "video source" with a name and some bytes.

youtube_url = st.text_input(
    "Or paste a YouTube link",
    key="youtube_url",
    placeholder="https://www.youtube.com/watch?v=..."
)

process_youtube = st.button(
    "Process YouTube Video",
    key="process_youtube"
)

# The video used is the one the user chose last: a newly uploaded
# file, or a YouTube video that was just processed.
upload_id = (
    (uploaded_video.name, uploaded_video.size)
    if uploaded_video is not None
    else None
)

if upload_id != st.session_state.last_upload_id:

    st.session_state.last_upload_id = upload_id

    if upload_id is not None:
        st.session_state.active_source = "upload"
        st.session_state.youtube_source = None
        st.session_state.youtube_error = None

if process_youtube:

    st.session_state.youtube_error = None

    progress_text = st.empty()
    progress_bar = st.progress(0.0)

    def show_youtube_progress(stage, fraction=None):

        text, value = describe_progress(stage, fraction)

        progress_text.write(text)
        progress_bar.progress(value)

    try:
        youtube_video = fetch_youtube_video(
            youtube_url,
            cache_dir=YOUTUBE_CACHE_DIR,
            on_progress=show_youtube_progress
        )

    except YouTubeError as error:
        st.session_state.youtube_error = error.user_message
        progress_text.empty()

    else:
        st.session_state.youtube_source = (
            YouTubeVideoSource(youtube_video)
        )
        st.session_state.active_source = "youtube"

    progress_bar.empty()

if st.session_state.youtube_error:
    st.error(st.session_state.youtube_error)

video_source = uploaded_video

if st.session_state.active_source == "youtube":
    video_source = st.session_state.youtube_source


# ============================================================
# PROCESS VIDEO
# ============================================================

if video_source is not None:

    video_data = video_source.getvalue()

    video_hash = hashlib.md5(
        video_data
    ).hexdigest()


    # ========================================================
    # PROCESS ONLY IF NEW VIDEO
    # ========================================================

    if st.session_state.video_hash != video_hash:

        # ----------------------------------------------------
        # CREATE DIRECTORIES
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

        video_path = getattr(video_source, "path", None)

        if video_path is None:

            # An uploaded file: save it under data/videos.
            video_path = os.path.join(
                "data",
                "videos",
                video_source.name
            )

            with open(
                video_path,
                "wb"
            ) as file:

                file.write(
                    video_data
                )

        # (A YouTube video is already on disk, in data/youtube.
        # It is used where it is: nothing is copied.)


        # ----------------------------------------------------
        # STORE VIDEO INFORMATION
        # ----------------------------------------------------

        st.session_state.video_hash = video_hash

        st.session_state.video_path = (
            video_path
        )

        st.session_state.video_name = (
            os.path.splitext(
                video_source.name
            )[0]
        )

        st.session_state.video_data = (
            video_data
        )

        # Reset previous question state
        st.session_state.jump_time = None

        st.session_state.seek_request = None

        st.session_state.last_question = ""

        st.session_state.last_answer = None

        st.session_state.last_evidence_segments = []

        st.session_state.last_related_keyframe = None


        st.success(
            getattr(
                video_source,
                "ready_message",
                "Video uploaded successfully!"
            )
        )


        # ====================================================
        # EXTRACT AUDIO
        # ====================================================

        audio_path = os.path.join(
            "data",
            "audio",
            "sample.wav"
        )

        with st.spinner(
            "Extracting audio from video..."
        ):

            extract_audio(
                video_path,
                audio_path
            )

        st.success(
            "Audio extracted successfully!"
        )


        # ====================================================
        # LOAD MODELS
        # ====================================================

        with st.spinner(
            "Loading AI models..."
        ):

            (
                whisper_model,
                embedding_model
            ) = load_models()


        # ====================================================
        # TRANSCRIPTION
        # ====================================================

        with st.spinner(
            "Transcribing video..."
        ):

            result = whisper_model.transcribe(
                audio_path
            )

        segments = result[
            "segments"
        ]

        st.session_state.segments = (
            segments
        )

        st.success(
            "Transcription completed!"
        )


        # ====================================================
        # KEYFRAME EXTRACTION
        # ====================================================

        with st.spinner(
            "Extracting video keyframes..."
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

        st.session_state.keyframes = (
            keyframes
        )

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

        st.success(
            f"Matched {len(matched_frames)} "
            "keyframes with transcript segments!"
        )


        # ====================================================
        # CREATE MULTIMODAL RECORDS
        # ====================================================

        multimodal_records = (
            create_multimodal_records(
                matched_frames
            )
        )

        st.session_state.multimodal_records = (
            multimodal_records
        )


        # ====================================================
        # SAVE MULTIMODAL RECORDS
        # ====================================================

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


        # ====================================================
        # CREATE CHUNKS
        # ====================================================

        chunks = create_chunks(
            segments
        )


        # ====================================================
        # CREATE EMBEDDINGS
        # ====================================================

        with st.spinner(
            "Creating embeddings..."
        ):

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = (
                embedding_model.encode(
                    texts,
                    normalize_embeddings=True
                )
            )


        # ====================================================
        # CREATE FAISS INDEX
        # ====================================================

        index = create_index(
            embeddings
        )


        # ====================================================
        # STORE IN SESSION STATE
        # ====================================================

        st.session_state.chunks = (
            chunks
        )

        st.session_state.index = (
            index
        )

        st.session_state.embedding_model = (
            embedding_model
        )

        st.success(
            "Video processing completed!"
        )


# ============================================================
# VIDEO PLAYER
# ============================================================

if st.session_state.video_data is not None:

    st.markdown(
        "## 🎬 Video Player"
    )


    # ========================================================
    # IMPORTANT:
    # USE jump_time AS THE VIDEO START TIME
    # ========================================================

    current_start_time = (
        st.session_state.jump_time
        if st.session_state.jump_time is not None
        else 0
    )

    st.video(
        st.session_state.video_data,
        start_time=current_start_time
    )


    # ========================================================
    # SHOW CURRENT JUMP INFORMATION
    # ========================================================

    if st.session_state.jump_time is not None:

        jump_time = (
            st.session_state.jump_time
        )

        st.success(
            f"🎯 Video opened at approximately "
            f"{int(jump_time)} seconds."
        )

        if st.button(
            "⬅️ Return to Beginning"
        ):

            st.session_state.jump_time = None

            st.rerun()


# ============================================================
# VIDEO SUMMARY
# ============================================================
#
# Isolated feature. The logic is in src/rag/video_summary.py.
#
# Every summary point uses the start, end and text of a real
# transcript chunk (nothing is invented), and its Jump button
# uses the same seek/autoplay mechanism as the other Jump buttons.

if (
    st.session_state.chunks is not None
    and st.session_state.video_data is not None
):

    # --------------------------------------------------------
    # GENERATE ONCE PER VIDEO
    # Reused on every rerun (for example after a Jump click).
    # --------------------------------------------------------
    cached_summary = st.session_state.video_summary

    if (
        cached_summary is None
        or cached_summary["video_hash"]
        != st.session_state.video_hash
    ):

        with st.spinner(
            "Generating video summary..."
        ):

            # A failure here must never break the rest of the app.
            try:
                new_summary = generate_video_summary(
                    st.session_state.chunks,
                    st.session_state.embedding_model
                )
            except Exception:
                new_summary = {
                    "main_topic": "",
                    "points": [],
                    "used_llm": False,
                    "error": True
                }

        st.session_state.video_summary = {
            "video_hash": st.session_state.video_hash,
            "summary": new_summary
        }

    video_summary = (
        st.session_state.video_summary["summary"]
    )

    # --------------------------------------------------------
    # DISPLAY (collapsible, collapsed by default)
    # --------------------------------------------------------
    # A rerun would otherwise collapse the section every time a
    # Jump button is clicked, so it is kept open for the rerun
    # that follows a summary Jump.
    summary_expanded = (
        st.session_state.summary_open
    )

    st.session_state.summary_open = False

    with st.expander(
        "📋 Video Summary",
        expanded=summary_expanded
    ):

        if not video_summary["points"]:

            if video_summary.get("error"):
                st.warning(
                    "The video summary could not be generated."
                )
            else:
                st.info(
                    "Not enough spoken content in this "
                    "video to build a summary."
                )

        else:

            st.markdown(
                f"**Main Topic:** "
                f"{escape_markdown(video_summary['main_topic'])}"
            )

            total_points = len(video_summary["points"])

            for number, point in enumerate(
                video_summary["points"],
                start=1
            ):

                start_time = int(
                    point["start"]
                )

                # Title
                st.markdown(
                    f"#### {number}. "
                    f"{escape_markdown(point['title'])}"
                )

                # One-sentence summary
                st.markdown(
                    escape_markdown(
                        point["summary"]
                    )
                )

                # Timestamp (taken from the transcript chunk),
                # shown as minutes:seconds
                st.markdown(
                    f"⏱ "
                    f"{format_time_range(point['start'], point['end'])}"
                )

                # Supporting transcript evidence
                st.markdown(
                    "📝 "
                    + escape_markdown(point["text"])
                )

                # Jump button (unique key per summary point)
                if st.button(
                    f"▶️ Jump to {format_timestamp(point['start'])}",
                    key=(
                        f"summary_jump_"
                        f"{number}_"
                        f"{start_time}"
                    )
                ):
                    request_video_jump(
                        start_time
                    )

                    st.session_state.summary_open = True

                    st.rerun()

                if number < total_points:
                    st.divider()

            if not video_summary["used_llm"]:
                st.caption(
                    "Wording is taken directly from the "
                    "transcript (the language model was "
                    "not used)."
                )


# ============================================================
# MULTIMODAL RECORD PREVIEW
# ============================================================

if (
    st.session_state.multimodal_records
    is not None
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

        # ----------------------------------------------------
        # FIRST RECORD
        # ----------------------------------------------------

        first_record = records[0]


        with st.expander(
            "View first multimodal record"
        ):

            st.json(
                first_record
            )


        # ----------------------------------------------------
        # KEYFRAME PREVIEW
        # ----------------------------------------------------

        st.markdown(
            "### 🖼️ Keyframe Preview"
        )


        if os.path.exists(
            first_record["image_path"]
        ):

            st.image(
                first_record["image_path"],
                caption=(
                    "Timestamp: "
                    f"{first_record['timestamp']} "
                    "seconds"
                ),
                use_container_width=True
            )


        # ----------------------------------------------------
        # ASSOCIATED TRANSCRIPT
        # ----------------------------------------------------

        st.markdown(
            "### 📝 Associated Transcript"
        )


        st.write(
            first_record[
                "transcript_text"
            ]
        )


    else:

        st.info(
            "No keyframes matched the transcript "
            "within the allowed tolerance."
        )


# ============================================================
# QUESTION ANSWERING
# ============================================================

if (
    st.session_state.index
    is not None
):

    st.markdown(
        "## 💬 Ask Questions About Your Video"
    )


    question = st.text_input(
        "Enter your question:",
        value=st.session_state.last_question,
        placeholder=(
            "Example: What is mentioned about "
            "machine learning?"
        )
    )


    ask_button = st.button(
        "🔍 Ask Question"
    )


    # ========================================================
    # ASK QUESTION
    # ========================================================

    if ask_button:

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # GET STORED MODELS / DATA
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
                    question,
                    normalize_embeddings=True
                )
            )


            # ------------------------------------------------
            # RETRIEVE RELEVANT CHUNKS
            # ------------------------------------------------

            results = retrieve_chunks(
                index,
                chunks,
                query_embedding,
                k=6,
                threshold=0.25
            )


            # ------------------------------------------------
            # PREPARE CONTEXT
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


            # ------------------------------------------------
            # SAVE ANSWER IN SESSION STATE
            # ------------------------------------------------

            st.session_state.last_question = (
                question
            )

            st.session_state.last_answer = (
                answer
            )


            # Clear previous jump when asking a new question
            st.session_state.jump_time = None


            # =================================================
            # FIND EVIDENCE SEGMENTS
            # =================================================

            if (
                "don't have enough information"
                in answer.lower()
            ):

                st.session_state.last_evidence_segments = []

                st.session_state.last_related_keyframe = None

            else:

                evidence_segments = (
                    find_answer_segments(
                        results,
                        answer,
                        embedding_model,
                        max_segments=1
                    )
                )


                st.session_state.last_evidence_segments = (
                    evidence_segments
                )


                # =================================================
                # FIND RELATED KEYFRAME
                # =================================================

                related_keyframe = None


                records = (
                    st.session_state.multimodal_records
                )


                if (
                    records
                    and evidence_segments
                ):

                    start_time = (
                        float(
                            evidence_segments[0][
                                "start"
                            ]
                        )
                    )


                    closest_distance = (
                        float("inf")
                    )


                    for record in records:

                        try:

                            record_time = float(
                                record[
                                    "timestamp"
                                ]
                            )


                            distance = abs(
                                record_time
                                - start_time
                            )


                            if (
                                distance
                                < closest_distance
                            ):

                                closest_distance = (
                                    distance
                                )

                                related_keyframe = (
                                    record
                                )


                        except (
                            KeyError,
                            ValueError,
                            TypeError
                        ):

                            continue


                st.session_state.last_related_keyframe = (
                    related_keyframe
                )


            # =================================================
            # FORCE RERUN SO RESULTS ARE DISPLAYED
            # =================================================

            st.rerun()


    # ========================================================
    # DISPLAY STORED ANSWER
    # ========================================================

    if (
        st.session_state.last_answer
        is not None
    ):

        st.markdown(
            "## 🧠 Answer"
        )


        st.write(
            st.session_state.last_answer
        )


        answer = (
            st.session_state.last_answer
        )


        # ====================================================
        # NO ANSWER / NO INFORMATION
        # ====================================================

        if (
            "don't have enough information"
            in answer.lower()
        ):

            st.markdown(
                "## ⏱️ Relevant Timestamps"
            )

            st.info(
                "No relevant information "
                "was found in the video."
            )


        # ====================================================
        # DISPLAY EVIDENCE
        # ====================================================

        else:

            evidence_segments = (
                st.session_state.last_evidence_segments
            )


            st.markdown(
                "## ⏱️ Relevant Timestamps"
            )


            if not evidence_segments:

                st.info(
                    "No precise evidence "
                    "segment was found."
                )


            else:

                for segment in evidence_segments:

                    start_time = int(
                        segment["start"]
                    )

                    end_time = int(
                        segment["end"]
                    )


                    # ----------------------------------------
                    # TIMESTAMP
                    # ----------------------------------------

                    st.markdown(
                        f"### ⏱️ "
                        f"{format_time_range(segment['start'], segment['end'])}"
                    )


                    # ----------------------------------------
                    # TRANSCRIPT EVIDENCE
                    # ----------------------------------------

                    st.write(
                        segment["text"]
                    )


                    # ----------------------------------------
                    # JUMP BUTTON
                    # ----------------------------------------

                    if st.button(
                        f"▶️ Jump to {format_timestamp(segment['start'])}",
                        key=(
                            f"jump_"
                            f"{start_time}"
                        )
                    ):

                        # THIS IS THE IMPORTANT PART
                        st.session_state.jump_time = (
                            start_time
                        )

                        # Command the browser to seek the real
                        # <video> element (sent at the end of
                        # the script, after the player exists).
                        st.session_state.seek_counter += 1

                        st.session_state.seek_request = {
                            "time": start_time,
                            "id": st.session_state.seek_counter,
                        }

                        st.rerun()


                    # ----------------------------------------
                    # VIDEO EVIDENCE INFO
                    # ----------------------------------------

                    st.caption(
                        f"Video evidence starts "
                        f"at approximately "
                        f"{start_time} seconds "
                        f"and ends at "
                        f"{end_time} seconds."
                    )


            # =================================================
            # RELATED KEYFRAME
            # =================================================

            related_keyframe = (
                st.session_state.last_related_keyframe
            )


            if related_keyframe:

                st.markdown(
                    "### 🖼️ Related Keyframe"
                )


                if os.path.exists(
                    related_keyframe[
                        "image_path"
                    ]
                ):

                    st.image(
                        related_keyframe[
                            "image_path"
                        ],
                        caption=(
                            "Keyframe at "
                            f"{related_keyframe['timestamp']} "
                            "seconds"
                        ),
                        use_container_width=True
                    )


                # ---------------------------------------------
                # RELATED TRANSCRIPT
                # ---------------------------------------------

                st.markdown(
                    "### 📝 Related Transcript"
                )


                st.write(
                    related_keyframe[
                        "transcript_text"
                    ]
                )


# ============================================================
# SEND PENDING SEEK COMMAND TO THE BROWSER
#
# Runs last, so the <video> element already exists on the page.
# The request is cleared right away: it is sent exactly once per
# click and never repeats on later reruns.
# ============================================================

if st.session_state.seek_request is not None:

    request = st.session_state.seek_request

    st.session_state.seek_request = None

    seek_video_player(
        request["time"],
        request["id"]
    )