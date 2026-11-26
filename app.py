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
    add_neighbouring_chunks,
    find_evidence
)

from src.rag.answer_generator import (
    prepare_context,
    generate_answer,
    LLMUnavailableError
)

from src.rag.video_summary import (
    generate_video_summary,
    escape_markdown,
    format_timestamp,
    format_time_range
)

from src.rag.viva import (
    generate_viva_questions,
    check_answer,
    viva_score
)

from src.processing import processing_cache

from src.processing.evidence_frame import (
    extract_frame_at,
    frame_time
)

from src.rag.report_export import (
    summary_report,
    viva_report,
    report_file_name
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

st.caption(
    "Ask questions about a video, get a timestamped summary and test "
    "yourself in a viva. Every answer points to the moment in the "
    "video it comes from."
)


# ============================================================
# LOOK
# ============================================================
#
# Colours and fonts are in .streamlit/config.toml. These few rules
# only shape what the theme cannot (title size, caption contrast,
# button hover, rounded and sticky video). If a future Streamlit
# version changes its page structure, a rule simply stops applying.

st.markdown(
    """
    <style>
    /* A slightly smaller title, so the video is visible at once. */
    h1 {
        font-size: 2.2rem !important;
    }
    /* Secondary text (captions) a little brighter on the dark theme. */
    div[data-testid="stCaptionContainer"] {
        color: rgba(230, 230, 239, 0.78);
    }
    /* Buttons: a subtle violet border on hover. */
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        border-color: #8B5CF6;
        color: #C4B5FD;
    }
    /* Rounded video. */
    div[data-testid="stVideo"] video {
        border-radius: 12px;
    }
    /* The video stays in view while a long section is scrolled. */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stVideo"]) {
        align-items: flex-start;
    }
    div[data-testid="stColumn"]:has(div[data-testid="stVideo"]) {
        position: sticky;
        top: 3.75rem;
        z-index: 1;
    }
    /* On a narrow screen the columns are stacked: no sticky video. */
    @media (max-width: 640px) {
        div[data-testid="stColumn"]:has(div[data-testid="stVideo"]) {
            position: static;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Filled with the start screen only when no video is loaded. Created
# this early so that, when a video starts processing, the start screen
# of the previous run disappears at once instead of staying faded
# under the progress box.
welcome_slot = st.empty()


# ============================================================
# SECTIONS
# ============================================================

SECTION_ASK = "💬 Ask Questions"

SECTION_SUMMARY = "📋 Summary"

SECTION_VIVA = "🎓 Viva"

SECTIONS = [
    SECTION_ASK,
    SECTION_SUMMARY,
    SECTION_VIVA
]


# ============================================================
# ANSWER MODEL (Q&A)
# ============================================================
#
# Q&A answers can use a larger model. llama3.1 (8B) writes more
# complete answers but is about 2-3x slower on a laptop CPU and has
# to be downloaded once ("ollama pull llama3.1"). The Summary and
# Viva always use llama3.2 (many calls; the larger model would make
# them too slow). If a chosen model is missing, Q&A says how to
# install it (the same message as for llama3.2).

ANSWER_MODELS = {
    "llama3.2 (fast)": "llama3.2",
    "llama3.1 8B (better, slower)": "llama3.1"
}


# ============================================================
# LOAD MODELS
# ============================================================

# The model names are also part of the processing cache key: a result
# made with another model is never reused.
WHISPER_MODEL_NAME = "base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# show_spinner=False: the processing box already says "Loading AI
# models..." (Streamlit's own message overlapped it).
@st.cache_resource(show_spinner=False)
def load_models():

    whisper_model = whisper.load_model(
        WHISPER_MODEL_NAME
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
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
# EXAMPLE QUESTIONS
# ============================================================
#
# Up to three questions built from the titles of the Video Summary
# points ("Types of Machine Learning" -> "What does the video say
# about types of machine learning?"), so the Q&A can be tried with
# one click. Only titles the language model wrote are used.

MAX_EXAMPLE_QUESTIONS = 3


def make_example_questions(video_summary):

    if not video_summary:
        return []

    summary = video_summary.get("summary") or {}

    if not summary.get("used_llm"):
        return []

    examples = []

    for point in summary.get("points", []):

        title = " ".join(
            str(point.get("title", "")).split()
        ).rstrip(".?!")

        if len(title.split()) < 2:
            continue

        # "Types Of Data" -> "types of data"; "AI" stays "AI".
        title = " ".join(
            word if len(word) > 1 and word.isupper() else word.lower()
            for word in title.split()
        )

        example = f"What does the video say about {title}?"

        if example.lower() not in (e.lower() for e in examples):
            examples.append(example)

        if len(examples) == MAX_EXAMPLE_QUESTIONS:
            break

    return examples


# ============================================================
# SHORTEN TEXT
# ============================================================
#
# Summary cards show a short piece of the transcript (the full
# passage is in the video). Cut at a word, never in the middle of one.

SHORT_TEXT_CHARS = 220


def shorten_text(text, limit=SHORT_TEXT_CHARS):

    text = " ".join(
        str(text).split()
    )

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0] + " …"


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
# The open section (Ask Questions / Summary / Viva).
if "section" not in st.session_state:
    st.session_state.section = SECTION_ASK

# Clicking the open section again un-selects it; then the last one
# is shown again.
if "last_section" not in st.session_state:
    st.session_state.last_section = SECTION_ASK

if st.session_state.section is None:
    st.session_state.section = st.session_state.last_section

if "summary_open" not in st.session_state:
    st.session_state.summary_open = False

# Viva Mode: {"video_hash": ..., "questions": [...]}, the number of
# the current question, and the result of every checked answer.
if "viva" not in st.session_state:
    st.session_state.viva = None

if "viva_index" not in st.session_state:
    st.session_state.viva_index = 0

if "viva_results" not in st.session_state:
    st.session_state.viva_results = {}

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

# Message shown when the last question could not be answered
# because Ollama was not available.
if "qa_error" not in st.session_state:
    st.session_state.qa_error = None


# The video is chosen in the sidebar, so the main page starts with
# the video itself. st.file_uploader / st.text_input are called
# inside "with st.sidebar:" (not st.sidebar.file_uploader), so
# they stay the same Streamlit calls as before.

with st.sidebar:

    st.markdown(
        "### 📂 Choose a video"
    )

    # Upload and YouTube link together in one box.
    source_box = st.container(border=True)

with source_box:

    # ============================================================
    # VIDEO UPLOAD
    # ============================================================

    uploaded_video = st.file_uploader(
        "Upload a video file",
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

    st.caption(
        "or"
    )

    youtube_url = st.text_input(
        "Paste a YouTube link",
        key="youtube_url",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    process_youtube = st.button(
        "Process YouTube Video",
        key="process_youtube",
        use_container_width=True
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

# A fixed place for the processing box, created on EVERY run (empty
# when nothing is processed). Without it, the video and the panel
# move up one place on the run after processing, and Streamlit then
# shows the old page faded below the new one until the run ends.
processing_slot = st.container()

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

        st.session_state.qa_error = None


        # All processing steps are shown in one status box that
        # collapses to "Video ready" when everything is done.
        with processing_slot.status(
            "Processing video...",
            expanded=True
        ) as processing_status:

            st.write(
                "✅ "
                + getattr(
                    video_source,
                    "ready_message",
                    "Video uploaded successfully!"
                )
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
            # CACHED TRANSCRIPT
            # ====================================================
            #
            # Whisper is the slowest step. If this exact video was
            # transcribed before (with the same Whisper model), its
            # saved transcript is used and audio extraction and
            # Whisper are skipped. See src/processing/processing_cache.py.

            segments = processing_cache.load_transcript(
                video_hash,
                WHISPER_MODEL_NAME
            )

            if segments is not None:

                st.write(
                    "✅ Transcript loaded from cache "
                    "(this video was processed before)."
                )

            else:

                # ================================================
                # EXTRACT AUDIO
                # ================================================

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

                st.write(
                    "✅ Audio extracted successfully!"
                )


                # ================================================
                # TRANSCRIPTION
                # ================================================

                with st.spinner(
                    "Transcribing video..."
                ):

                    result = whisper_model.transcribe(
                        audio_path
                    )

                segments = result[
                    "segments"
                ]

                processing_cache.save_transcript(
                    video_hash,
                    WHISPER_MODEL_NAME,
                    segments
                )

                st.write(
                    "✅ Transcription completed!"
                )

            st.session_state.segments = (
                segments
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

            st.write(
                f"✅ Extracted {len(keyframes)} keyframes!"
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

            st.write(
                f"✅ Matched {len(matched_frames)} "
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

            st.write(
                "✅ Multimodal records saved successfully!"
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

            st.write(
                "✅ Video processing completed!"
            )

            processing_status.update(
                label="✅ Video ready",
                state="complete",
                expanded=False
            )


# ============================================================
# VIDEO SUMMARY (GENERATION)
# ============================================================
#
# Made right after processing (or loaded from the cache), whichever
# section is open, so it is ready when the Summary or Viva section
# is opened. Displayed further below, in the Summary section.

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

        # A summary Ollama wrote earlier for exactly these chunks
        # is reused instead of asking Ollama again.
        new_summary = processing_cache.load_summary(
            st.session_state.video_hash,
            EMBEDDING_MODEL_NAME,
            st.session_state.chunks
        )

        if new_summary is None:

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

            # (Only kept when Ollama really wrote it.)
            processing_cache.save_summary(
                st.session_state.video_hash,
                EMBEDDING_MODEL_NAME,
                st.session_state.chunks,
                new_summary
            )

        st.session_state.video_summary = {
            "video_hash": st.session_state.video_hash,
            "summary": new_summary
        }


# ============================================================
# VIDEO PLAYER
# ============================================================

if st.session_state.video_data is not None:

    # The video on the left, the open section on the right, so an
    # answer, a summary point or a viva question and its Jump time
    # are always next to the video.
    video_column, panel_column = st.columns(
        [1.15, 1],
        gap="large"
    )

    with video_column:

        st.markdown(
            f"#### 🎬 {escape_markdown(str(st.session_state.video_name))}"
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

            # jump_time is the POSITION in the video (in seconds), shown
            # as m:ss like every other timestamp: 173 -> "2:53".
            st.success(
                f"🎯 Video opened at "
                f"{format_timestamp(jump_time)}."
            )

            if st.button(
                "⬅️ Return to Beginning"
            ):

                st.session_state.jump_time = None

                st.rerun()


    # ========================================================
    # SECTIONS
    # ========================================================
    #
    # A segmented control that remembers the open section (key
    # "section"). Not st.tabs: a Jump click reruns the page and
    # st.tabs can then fall back to its first tab.

    with panel_column:

        st.segmented_control(
            "Section",
            SECTIONS,
            key="section",
            label_visibility="collapsed"
        )

        section = (
            st.session_state.section
            or st.session_state.last_section
        )

        st.session_state.last_section = section

else:

    section = None

    # ========================================================
    # NO VIDEO YET
    # ========================================================

    with welcome_slot.container():

        st.info(
            "👈 Upload a video or paste a YouTube link "
            "in the sidebar to start."
        )

        for column, (title, text) in zip(
            st.columns(3),
            [
                (
                    "💬 Ask Questions",
                    "Answers come only from what the video says, "
                    "with a Jump button to the exact moment."
                ),
                (
                    "📋 Summary",
                    "The main points of the video, each with its "
                    "time and the words spoken there."
                ),
                (
                    "🎓 Viva",
                    "Questions about the video; your answers are "
                    "checked against what the video says."
                ),
            ]
        ):

            with column:

                with st.container(border=True):

                    st.markdown(
                        f"**{title}**"
                    )

                    st.caption(
                        text
                    )


# ============================================================
# CURRENT VIDEO (SIDEBAR)
# ============================================================

if st.session_state.video_data is not None:

    current_box = st.sidebar.container(border=True)

    current_box.markdown(
        f"🎬 **Current video**  \n"
        f"{escape_markdown(str(st.session_state.video_name))}"
    )

    # ========================================================
    # REPROCESS VIDEO
    # ========================================================
    #
    # A video that was processed before is loaded from the cache.
    # This button deletes its cached transcript and summary and
    # processes the video again from the beginning.
    #
    # Only offered while the video can still be read again (not
    # after the uploaded file was removed from the uploader).

    if video_source is not None and current_box.button(
        "🔄 Reprocess video",
        key="reprocess_video",
        type="tertiary",
        help=(
            "Run transcription and the summary again "
            "instead of using the saved results."
        )
    ):

        processing_cache.clear_video(
            st.session_state.video_hash
        )

        # A different hash makes the block above process the
        # video again on the next run.
        st.session_state.video_hash = None

        st.session_state.video_summary = None

        st.session_state.viva = None

        st.session_state.viva_index = 0

        st.session_state.viva_results = {}

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
    st.session_state.video_summary is not None
    and st.session_state.video_data is not None
    and section == SECTION_SUMMARY
):

    video_summary = (
        st.session_state.video_summary["summary"]
    )

    # --------------------------------------------------------
    # DISPLAY (in the Summary section, next to the video)
    # --------------------------------------------------------
    with panel_column:

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
                f"#### 📋 "
                f"{escape_markdown(video_summary['main_topic'])}"
            )

            # A printable page (Print -> Save as PDF for a PDF).
            st.download_button(
                "⬇️ Download summary",
                data=summary_report(
                    st.session_state.video_name,
                    video_summary
                ),
                file_name=report_file_name(
                    st.session_state.video_name,
                    "summary"
                ),
                mime="text/html",
                key="download_summary",
                help="A printable page. Use Print > Save as PDF for a PDF."
            )

            for number, point in enumerate(
                video_summary["points"],
                start=1
            ):

                start_time = int(
                    point["start"]
                )

                # One card per summary point: title, summary, a
                # short piece of the transcript, then time and Jump
                # always on one row at the bottom.
                with st.container(border=True):

                    st.markdown(
                        f"**{number}. "
                        f"{escape_markdown(point['title'])}**"
                    )

                    st.markdown(
                        escape_markdown(
                            point["summary"]
                        )
                    )

                    st.caption(
                        "📝 "
                        + escape_markdown(
                            shorten_text(point["text"])
                        )
                    )

                    time_column, jump_column = st.columns(
                        [1, 1],
                        vertical_alignment="center"
                    )

                    time_column.markdown(
                        f"⏱ "
                        f"{format_time_range(point['start'], point['end'])}"
                    )

                    # Jump button (unique key per summary point)
                    if jump_column.button(
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

                        st.rerun()

            if not video_summary["used_llm"]:
                st.caption(
                    "Wording is taken directly from the "
                    "transcript (the language model was "
                    "not used)."
                )


# ============================================================
# VIVA MODE
# ============================================================
#
# Isolated feature. The logic is in src/rag/viva.py.
#
# Questions are made from the Video Summary points, so every
# question has the real timestamps and transcript of a summary
# point. Its Jump button uses the same seek/autoplay mechanism as
# the other Jump buttons. Nothing above this section is changed.

if (
    st.session_state.video_summary is not None
    and st.session_state.video_data is not None
    and st.session_state.video_summary["video_hash"]
    == st.session_state.video_hash
    and section == SECTION_VIVA
):

    viva = st.session_state.viva

    # --------------------------------------------------------
    # A NEW VIDEO STARTS A NEW VIVA
    # --------------------------------------------------------
    if (
        viva is not None
        and viva["video_hash"]
        != st.session_state.video_hash
    ):

        viva = None

        st.session_state.viva = None

        st.session_state.viva_index = 0

        st.session_state.viva_results = {}

    # In the Viva section, next to the video.
    with panel_column:

        # ----------------------------------------------------
        # NOT STARTED YET
        # ----------------------------------------------------
        if viva is None:

            st.write(
                "Practise for a viva: answer questions about this "
                "video and get your answers checked against the "
                "transcript."
            )

            if st.button(
                "🎓 Generate Viva Questions",
                key="viva_generate"
            ):

                with st.spinner(
                    "Generating viva questions..."
                ):

                    # A failure here must never break the rest of the app.
                    try:
                        questions = generate_viva_questions(
                            st.session_state.video_summary[
                                "summary"
                            ]["points"],
                            segments=st.session_state.segments,
                            embedding_model=(
                                st.session_state.embedding_model
                            )
                        )
                        failed = False

                    except Exception:
                        questions = []
                        failed = True

                st.session_state.viva = {
                    "video_hash": st.session_state.video_hash,
                    "questions": questions,
                    "attempt": 1,
                    "error": failed
                }

                st.session_state.viva_index = 0

                st.session_state.viva_results = {}

                st.rerun()

        # ----------------------------------------------------
        # NO QUESTIONS
        # ----------------------------------------------------
        elif not viva["questions"]:

            if viva.get("error"):
                st.warning(
                    "The viva questions could not be generated."
                )
            else:
                st.info(
                    "Not enough spoken content in this "
                    "video to build viva questions."
                )

            if st.button(
                "🔁 Try Again",
                key="viva_retry"
            ):

                st.session_state.viva = None

                st.rerun()

        else:

            questions = viva["questions"]

            total_questions = len(questions)

            number = st.session_state.viva_index

            results = st.session_state.viva_results

            # ------------------------------------------------
            # ONE QUESTION
            # ------------------------------------------------
            if number < total_questions:

                item = questions[number]

                st.markdown(
                    f"**Question {number + 1} of {total_questions}**"
                )

                st.progress(
                    number / total_questions
                )

                st.markdown(
                    f"#### {escape_markdown(item['question'])}"
                )

                # The key changes per question and per attempt, so
                # every question starts with an empty answer box.
                user_answer = st.text_area(
                    "Your answer:",
                    key=(
                        f"viva_answer_"
                        f"{viva['attempt']}_"
                        f"{number}"
                    )
                )

                if st.button(
                    "✅ Check Answer",
                    key=f"viva_check_{number}"
                ):

                    if not user_answer.strip():

                        st.warning(
                            "Please type an answer first."
                        )

                    else:

                        # The LLM examiner can take a few seconds.
                        with st.spinner(
                            "Checking answer..."
                        ):

                            results[number] = check_answer(
                                user_answer,
                                item,
                                st.session_state.embedding_model
                            )

                        st.rerun()

                # --------------------------------------------
                # RESULT OF THIS QUESTION
                # --------------------------------------------
                result = results.get(number)

                if result is not None:

                    feedback = escape_markdown(
                        result["feedback"]
                    )

                    if result["verdict"] == "correct":
                        st.success(
                            f"✅ Correct. {feedback}"
                        )
                    elif result["verdict"] == "partial":
                        st.warning(
                            f"🟡 Partly correct. {feedback}"
                        )
                    else:
                        st.error(
                            f"❌ Incorrect. {feedback}"
                        )

                    # Expected answer, time and evidence in one card.
                    with st.container(border=True):

                        # Expected answer
                        st.markdown(
                            "**Expected answer:** "
                            + escape_markdown(item["answer"])
                        )

                        # Timestamp, shown as minutes:seconds
                        st.markdown(
                            f"⏱ "
                            f"{format_time_range(item['start'], item['end'])}"
                        )

                        # Supporting transcript evidence
                        st.markdown(
                            "📝 "
                            + escape_markdown(item["text"])
                        )

                        # Jump button (same mechanism as everywhere else)
                        if st.button(
                            f"▶️ Jump to {format_timestamp(item['start'])}",
                            key=(
                                f"viva_jump_"
                                f"{number}_"
                                f"{int(item['start'])}"
                            )
                        ):
                            request_video_jump(
                                int(item["start"])
                            )

                            st.rerun()

                    next_label = (
                        "➡️ Next Question"
                        if number + 1 < total_questions
                        else "🏁 Finish Viva"
                    )

                    if st.button(
                        next_label,
                        key=f"viva_next_{number}"
                    ):

                        st.session_state.viva_index += 1

                        st.rerun()

            # ------------------------------------------------
            # FINAL SCORE
            # ------------------------------------------------
            else:

                final = viva_score(
                    results,
                    total_questions
                )

                st.markdown(
                    f"### 🏁 Final Score: "
                    f"{final['score']:g} / {final['total']} "
                    f"({final['percent']}%)"
                )

                # How many answers were correct, partly correct or
                # incorrect (unanswered questions count as neither).
                counts = {
                    verdict: sum(
                        1
                        for result in results.values()
                        if result["verdict"] == verdict
                    )
                    for verdict in ("correct", "partial", "incorrect")
                }

                for column, (label, verdict) in zip(
                    st.columns(3),
                    [
                        ("✅ Correct", "correct"),
                        ("🟡 Partly", "partial"),
                        ("❌ Incorrect", "incorrect"),
                    ]
                ):
                    column.metric(
                        label,
                        counts[verdict]
                    )

                verdict_icons = {
                    "correct": "✅",
                    "partial": "🟡",
                    "incorrect": "❌"
                }

                for position, item in enumerate(questions):

                    result = results.get(position)

                    icon = (
                        verdict_icons[result["verdict"]]
                        if result is not None
                        else "➖"
                    )

                    st.markdown(
                        f"{icon} **Q{position + 1}.** "
                        f"{escape_markdown(item['question'])} "
                        f"(⏱ {format_timestamp(item['start'])})"
                    )

                # What the student typed, for the report.
                typed_answers = {
                    position: st.session_state.get(
                        f"viva_answer_{viva['attempt']}_{position}"
                    )
                    for position in range(total_questions)
                }

                st.download_button(
                    "⬇️ Download results",
                    data=viva_report(
                        st.session_state.video_name,
                        questions,
                        results,
                        final,
                        typed_answers
                    ),
                    file_name=report_file_name(
                        st.session_state.video_name,
                        "viva_results"
                    ),
                    mime="text/html",
                    key="download_viva",
                    help="A printable page. Use Print > Save as PDF for a PDF."
                )

                restart_column, new_column, _ = st.columns(
                    [1, 1, 1]
                )

                with restart_column:

                    if st.button(
                        "🔁 Restart Viva",
                        key="viva_restart"
                    ):

                        viva["attempt"] += 1

                        st.session_state.viva_index = 0

                        st.session_state.viva_results = {}

                        st.rerun()

                with new_column:

                    if st.button(
                        "🆕 New Questions",
                        key="viva_new"
                    ):

                        st.session_state.viva = None

                        st.session_state.viva_index = 0

                        st.session_state.viva_results = {}

                        st.rerun()

            if any(
                item["source"] == "transcript"
                for item in questions
            ):
                st.caption(
                    "Some questions use a simple form because the "
                    "language model was not available or its wording "
                    "was not supported by the transcript."
                )


# ============================================================
# QUESTION ANSWERING
# ============================================================

if (
    st.session_state.index
    is not None
    and section == SECTION_ASK
):

    with panel_column:

        st.markdown(
            "#### 💬 Ask Questions About Your Video"
        )


        # A form, so pressing Enter asks the question too.
        with st.form(
            "ask_form",
            border=False
        ):

            question = st.text_input(
                "Enter your question:",
                value=st.session_state.last_question,
                placeholder=(
                    "Example: What is mentioned about "
                    "machine learning?"
                )
            )

            ask_button = st.form_submit_button(
                "🔍 Ask Question"
            )


        # ----------------------------------------------------
        # EXAMPLE QUESTIONS (from the summary point titles)
        # ----------------------------------------------------
        #
        # One click asks it. Only shown before the first question,
        # so they do not push the answer down.

        example_questions = make_example_questions(
            st.session_state.video_summary
        )

        if (
            example_questions
            and st.session_state.last_answer is None
        ):

            st.caption(
                "Try one of these:"
            )

            for position, example in enumerate(example_questions):

                if st.button(
                    example,
                    key=f"example_question_{position}"
                ):

                    question = example

                    ask_button = True


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

                # Also the chunk before and after every match, so a
                # sentence cut between two chunks is complete again.
                results = add_neighbouring_chunks(
                    results,
                    chunks
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

                    try:

                        answer = generate_answer(
                            context,
                            question,
                            # Read from the dropdown's own state: the
                            # Settings box is drawn later in the page.
                            model=ANSWER_MODELS.get(
                                st.session_state.get("answer_model_label"),
                                "llama3.2"
                            )
                        )

                    except LLMUnavailableError as error:

                        # No answer is shown (not even the previous one):
                        # only the message, and the question stays in
                        # the box so it can be asked again.
                        st.session_state.last_question = (
                            question
                        )

                        st.session_state.last_answer = None

                        st.session_state.last_evidence_segments = []

                        st.session_state.last_related_keyframe = None

                        st.session_state.qa_error = str(error)

                        st.rerun()


                # A successful answer clears an earlier error message.
                st.session_state.qa_error = None


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

                    # Two consecutive transcript pieces, chosen by the
                    # question and the answer (see find_evidence).
                    evidence_segments = (
                        find_evidence(
                            results,
                            question,
                            answer,
                            embedding_model
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


                    # Better: a picture from the exact moment of the
                    # evidence (keyframes are only every 30 s, so the
                    # nearest one can show another scene). The nearest
                    # keyframe above is kept only if this fails.
                    if (
                        evidence_segments
                        and st.session_state.video_path
                    ):

                        moment = frame_time(
                            evidence_segments[0]["start"],
                            evidence_segments[0]["end"]
                        )

                        frame_path = extract_frame_at(
                            st.session_state.video_path,
                            moment,
                            os.path.join(
                                "data",
                                "keyframes",
                                str(st.session_state.video_name)
                            )
                        )

                        if frame_path:

                            related_keyframe = {
                                "image_path": frame_path,
                                "timestamp": moment,
                                "exact": True
                            }


                    st.session_state.last_related_keyframe = (
                        related_keyframe
                    )


                # =================================================
                # FORCE RERUN SO RESULTS ARE DISPLAYED
                # =================================================

                st.rerun()


        # ========================================================
        # LANGUAGE MODEL NOT AVAILABLE
        # ========================================================

        if st.session_state.qa_error:

            st.error(
                f"⚠️ {st.session_state.qa_error}"
            )


        # ========================================================
        # DISPLAY STORED ANSWER
        # ========================================================

        if (
            st.session_state.last_answer
            is not None
        ):

            st.markdown(
                "#### 🧠 Answer"
            )


            with st.container(border=True):

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
                    "#### ⏱️ Relevant Timestamps"
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
                    "#### ⏱️ Relevant Timestamps"
                )


                if not evidence_segments:

                    st.info(
                        "No precise evidence "
                        "segment was found."
                    )


                else:

                    # Where the video says it: separate from the answer.
                    evidence_box = st.container(border=True)

                    with evidence_box:

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
                                f"##### ⏱️ "
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
                                f"Video evidence: "
                                f"{format_time_range(segment['start'], segment['end'])}"
                            )


                # =================================================
                # RELATED KEYFRAME
                # =================================================

                related_keyframe = (
                    st.session_state.last_related_keyframe
                )


                if related_keyframe:

                    exact_frame = related_keyframe.get(
                        "exact",
                        False
                    )

                    st.markdown(
                        "##### 🖼️ At this moment in the video"
                        if exact_frame
                        else "##### 🖼️ Related Keyframe"
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
                                (
                                    "Video frame at "
                                    if exact_frame
                                    else "Nearest keyframe at "
                                )
                                + format_timestamp(
                                    related_keyframe["timestamp"]
                                )
                            ),
                            use_container_width=True
                        )


                    # ---------------------------------------------
                    # RELATED TRANSCRIPT
                    # ---------------------------------------------
                    # Only for the nearest stored keyframe: an exact
                    # frame belongs to the evidence shown above.

                    if not exact_frame:

                        st.markdown(
                            "##### 📝 Related Transcript"
                        )


                        st.write(
                            related_keyframe[
                                "transcript_text"
                            ]
                        )


# ============================================================
# DEVELOPER DETAILS (MULTIMODAL RECORD PREVIEW)
# ============================================================
#
# What the pipeline produced (keyframes matched with the transcript),
# for checking and for the project report. Collapsed at the bottom.

if (
    st.session_state.multimodal_records
    is not None
    and st.session_state.video_data is not None
):

    with st.expander(
        "🛠 Developer details",
        expanded=False
    ):

        records = (
            st.session_state.multimodal_records
        )


        st.markdown(
            "##### 🖼️ Multimodal Record Preview"
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


            st.caption(
                "First multimodal record:"
            )

            st.json(
                first_record,
                expanded=False
            )


            # ----------------------------------------------------
            # KEYFRAME PREVIEW
            # ----------------------------------------------------

            st.markdown(
                "##### 🖼️ Keyframe Preview"
            )


            if os.path.exists(
                first_record["image_path"]
            ):

                st.image(
                    first_record["image_path"],
                    caption=(
                        "Keyframe at "
                        f"{format_timestamp(first_record['timestamp'])}"
                    ),
                    use_container_width=True
                )


            # ----------------------------------------------------
            # ASSOCIATED TRANSCRIPT
            # ----------------------------------------------------

            st.markdown(
                "##### 📝 Associated Transcript"
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
# ABOUT (SIDEBAR)
# ============================================================

with st.sidebar:

    with st.expander(
        "⚙️ Settings"
    ):

        st.selectbox(
            "Model for answers",
            list(ANSWER_MODELS),
            key="answer_model_label",
            help=(
                "Used for Ask Questions. The larger model gives more "
                "complete answers but is slower. Install it once with: "
                "ollama pull llama3.1"
            )
        )

    with st.expander(
        "ℹ️ About this app"
    ):

        st.markdown(
            "- **Transcript:** Whisper (base)\n"
            "- **Search:** MiniLM embeddings + FAISS\n"
            "- **Answers:** llama3.2, or llama3.1 8B (Settings)\n"
            "- **Summary and viva:** llama3.2 via Ollama\n"
            "- Everything runs **locally** on this computer."
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