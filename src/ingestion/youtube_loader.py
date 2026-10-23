"""
YouTube ingestion for the AI Video Assistant.

    YouTube URL  ->  a small local video file  ->  the EXISTING pipeline

This module only gets a video onto the disk. Audio extraction, Whisper,
chunking, embeddings, FAISS, Q&A, summary, keyframes and Jump all work
on the local file exactly as they do for an uploaded video.

What it does
------------
  - validates the URL and extracts the 11-character video ID
    (only youtube.com / youtu.be links; one video, never a playlist)
  - caches the download under  data/youtube/<video id>.mp4  (a folder of
    its own, so it never mixes with the uploaded videos in data/videos)
  - downloads with yt-dlp at about 480p, preferring H.264 + AAC, so the
    file stays small and works with OpenCV, ffmpeg and the browser player
  - converts to H.264 / AAC mp4 only if yt-dlp delivered something else
  - never shows yt-dlp console output; every failure becomes a short,
    readable message (see YouTubeError.user_message)

The download is written to a temporary name and moved into the cache only
when it is complete, so a cache hit is always a whole file.
"""

import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


log = logging.getLogger(__name__)


# ============================================================
# SETTINGS
# ============================================================

# Next to data/videos (uploads), never inside it.
DEFAULT_CACHE_DIR = os.path.join("data", "youtube")

# Aim for about this height (pixels). Smaller than this is accepted;
# larger is only used when nothing smaller exists.
TARGET_HEIGHT = 480

# Whisper time grows with the length, so long videos are refused.
MAX_VIDEO_SECONDS = 60 * 60

# A "cached" file smaller than this is treated as broken.
MIN_CACHED_BYTES = 1024

# yt-dlp format choice: best video + best audio, ranked by
#   1. H.264 (works with OpenCV and every browser, no re-encoding)
#   2. height up to TARGET_HEIGHT (taller only if nothing smaller)
#   3. AAC audio   4. mp4 / m4a containers
FORMAT = "bv*+ba/b"
FORMAT_SORT = [
    "vcodec:h264",
    f"res:{TARGET_HEIGHT}",
    "acodec:aac",
    "ext:mp4:m4a",
]


# ============================================================
# ERRORS  (user_message is safe to show; detail is for the log)
# ============================================================

class YouTubeError(Exception):

    def __init__(self, user_message, detail=""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class InvalidYouTubeURL(YouTubeError):
    """The text is not a link to a single YouTube video."""


class VideoUnavailable(YouTubeError):
    """Private, removed, age-restricted, members-only or region-blocked."""


class UnsupportedVideo(YouTubeError):
    """A live stream / premiere, or a video that is too long."""


class DownloadFailed(YouTubeError):
    """Network problems, YouTube blocking the request, bad formats ..."""


class DownloaderMissing(YouTubeError):
    """yt-dlp is not installed."""


# ============================================================
# URL  ->  VIDEO ID
# ============================================================

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
}

_SHORT_HOSTS = {
    "youtu.be",
    "www.youtu.be",
}

# /embed/ID, /shorts/ID, /live/ID, /v/ID
_ID_IN_PATH = ("embed", "shorts", "live", "v")

_NOT_A_VIDEO = InvalidYouTubeURL(
    "That doesn't look like a YouTube video link. Paste a link like "
    "https://www.youtube.com/watch?v=... or https://youtu.be/..."
)


def extract_video_id(url):
    """
    Return the 11-character video ID of a YouTube video link.

    Accepts watch?v=, youtu.be/, /shorts/, /embed/, /live/ links (with or
    without https://, with any extra parameters such as &t=387 or &list=).
    Anything else raises InvalidYouTubeURL: other websites, look-alike
    domains, playlists, channels, search pages, and non-http(s) schemes.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidYouTubeURL("Paste a YouTube link first.")

    text = url.strip()

    if "://" not in text:
        text = "https://" + text

    try:
        parsed = urlparse(text)
        port = parsed.port
    except ValueError:
        raise _NOT_A_VIDEO

    # "https://youtube.com@evil.com/..." and similar tricks
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise _NOT_A_VIDEO

    if port not in (None, 80, 443):
        raise _NOT_A_VIDEO

    host = (parsed.hostname or "").lower()
    segments = [part for part in parsed.path.split("/") if part]

    if host in _SHORT_HOSTS:
        candidate = segments[0] if segments else ""

    elif host in _YOUTUBE_HOSTS:

        if parsed.path.rstrip("/") == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]

        elif len(segments) >= 2 and segments[0] in _ID_IN_PATH:
            candidate = segments[1]

        elif parsed.path.rstrip("/") == "/playlist":
            raise InvalidYouTubeURL(
                "Playlist links aren't supported. "
                "Paste the link of a single video."
            )

        else:
            raise _NOT_A_VIDEO

    else:
        raise _NOT_A_VIDEO

    if not _VIDEO_ID.match(candidate):
        raise _NOT_A_VIDEO

    return candidate


def canonical_url(video_id):
    """The only URL ever handed to yt-dlp (no playlist, no tracking)."""
    return f"https://www.youtube.com/watch?v={video_id}"


# ============================================================
# CACHE
# ============================================================

def cache_path(cache_dir, video_id):
    return os.path.join(cache_dir, f"{video_id}.mp4")


def _meta_path(cache_dir, video_id):
    return os.path.join(cache_dir, f"{video_id}.json")


def find_cached_video(cache_dir, video_id):
    """Path of the cached file, or None (missing, or empty / broken)."""
    path = cache_path(cache_dir, video_id)

    try:
        if os.path.isfile(path) and os.path.getsize(path) >= MIN_CACHED_BYTES:
            return path
    except OSError:
        pass

    return None


def _read_meta(cache_dir, video_id):
    try:
        with open(_meta_path(cache_dir, video_id), encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_meta(cache_dir, video_id, meta):
    target = _meta_path(cache_dir, video_id)
    temporary = target + f".{uuid.uuid4().hex[:8]}.tmp"

    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(meta, file)
        os.replace(temporary, target)
    except OSError:
        # The title is only a nicety; never fail a download for it.
        try:
            os.remove(temporary)
        except OSError:
            pass


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass
class YouTubeVideo:
    video_id: str
    path: str
    title: str
    duration: float
    from_cache: bool


class YouTubeVideoSource:
    """
    Makes a cached YouTube file look like an uploaded file to the
    existing processing block in app.py.

        name       the "file name" (youtube_<id>.mp4)
        getvalue() the bytes, like UploadedFile.getvalue()
        path       where the file already is on disk, so it is used in
                   place and not copied into data/videos
    """

    def __init__(self, video):
        self.video = video
        self.path = video.path
        self.name = f"youtube_{video.video_id}.mp4"
        self._data = None

        self.ready_message = (
            "YouTube video found in the cache, no download needed!"
            if video.from_cache
            else "YouTube video downloaded successfully!"
        )

    def getvalue(self):
        if self._data is None:
            with open(self.path, "rb") as file:
                self._data = file.read()

        return self._data


# ============================================================
# yt-dlp SETUP  (silent, small, one video)
# ============================================================

class _QuietLogger:
    """
    yt-dlp prints to the console even with quiet=True (errors go to
    stderr). Giving it a logger that prints nothing keeps the console
    and the app free of raw yt-dlp output; the messages are only kept
    for the Python log.
    """

    def debug(self, message):
        log.debug("yt-dlp: %s", message)

    def info(self, message):
        log.debug("yt-dlp: %s", message)

    def warning(self, message):
        log.debug("yt-dlp warning: %s", message)

    def error(self, message):
        log.debug("yt-dlp error: %s", message)


def _base_options():
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
    }


def build_download_options(output_template, progress_hook=None):
    options = _base_options()

    options.update({
        "format": FORMAT,
        "format_sort": list(FORMAT_SORT),
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "restrictfilenames": True,
        "concurrent_fragment_downloads": 4,
    })

    if progress_hook is not None:
        options["progress_hooks"] = [progress_hook]

    return options


def _load_yt_dlp():
    try:
        import yt_dlp
    except ImportError as error:
        raise DownloaderMissing(
            "YouTube support needs yt-dlp. Install it with:  "
            "pip install yt-dlp"
        ) from error

    return yt_dlp.YoutubeDL


# ============================================================
# ERRORS FROM yt-dlp  ->  readable messages
# ============================================================

def _normalise(text):
    return (
        str(text)
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .lower()
    )


# (words that identify the problem, exception class, message)
# The first row that matches wins, so the specific ones come first.
_ERROR_RULES = [
    (("private video",),
     VideoUnavailable,
     "This video is private, so it can't be downloaded."),

    (("members-only", "members only", "join this channel"),
     VideoUnavailable,
     "This video is only for channel members, so it can't be downloaded."),

    (("confirm your age", "age-restricted", "age restricted"),
     VideoUnavailable,
     "This video is age-restricted and needs a signed-in account, "
     "which this app doesn't use."),

    (("available in your country", "blocked it in your country",
      "geo restrict", "geo-restrict"),
     VideoUnavailable,
     "This video isn't available in your country."),

    (("video unavailable", "this video is unavailable",
      "removed by the uploader", "no longer available",
      "has been terminated", "video has been removed",
      "does not exist"),
     VideoUnavailable,
     "This video is unavailable. It may have been removed or made private."),

    (("not a bot",),
     DownloadFailed,
     "YouTube asked for a sign-in check and blocked the automated "
     "download. Try again later, or download the video yourself and "
     "upload the file."),

    (("live event will begin", "premieres in", "is live", "live stream"),
     UnsupportedVideo,
     "Live streams and premieres aren't supported. Try again once the "
     "recording is available."),

    (("http error 429", "too many requests"),
     DownloadFailed,
     "YouTube is temporarily limiting downloads (too many requests). "
     "Wait a few minutes and try again, or upload the file instead."),

    (("http error 403", "forbidden"),
     DownloadFailed,
     "YouTube refused the download (access was forbidden). Wait a little "
     "and try again, update yt-dlp (pip install -U yt-dlp), or upload "
     "the file instead."),

    (("failed to resolve", "transporterror", "proxyerror",
      "getaddrinfo", "name or service not known",
      "temporary failure in name resolution", "timed out",
      "connection refused", "connection reset", "network is unreachable",
      "urlopen error", "connectionerror"),
     DownloadFailed,
     "Couldn't reach YouTube. Check your internet connection and try again."),

    (("requested format is not available", "no video formats",
      "only images are available", "no supported javascript runtime"),
     DownloadFailed,
     "YouTube didn't offer a downloadable video format. Update yt-dlp "
     "(pip install -U yt-dlp), make sure Deno is installed, or upload "
     "the file instead."),
]

_GENERIC_FAILURE = (
    "The download failed. You can try again, update yt-dlp "
    "(pip install -U yt-dlp), or upload the video file instead."
)


def classify_download_error(error):
    """Turn whatever yt-dlp (or the disk) raised into a YouTubeError."""
    if isinstance(error, YouTubeError):
        return error

    text = _normalise(error)

    for words, error_class, message in _ERROR_RULES:
        if any(word in text for word in words):
            return error_class(message, detail=str(error))

    if isinstance(error, OSError):
        return DownloadFailed(
            "The video couldn't be saved. Check that there is free disk "
            "space and that the data/youtube folder can be written to.",
            detail=str(error),
        )

    return DownloadFailed(_GENERIC_FAILURE, detail=str(error))


# ============================================================
# COMPATIBILITY  (OpenCV, ffmpeg and the browser player need H.264)
# ============================================================

def probe_media(path):
    """Container and codecs of a file, or None if ffprobe is not usable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", path,
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
        data = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

    streams = data.get("streams", [])

    def codec_of(kind):
        for stream in streams:
            if stream.get("codec_type") == kind:
                return stream.get("codec_name")
        return None

    return {
        "container": (data.get("format") or {}).get("format_name", ""),
        "video": codec_of("video"),
        "audio": codec_of("audio"),
    }


def is_compatible(probe):
    return (
        "mp4" in probe["container"].split(",")
        and probe["video"] == "h264"
        and probe["audio"] in (None, "aac")
    )


def ensure_compatible_mp4(path, on_convert=None):
    """
    Return a path to an H.264 / AAC mp4.

    yt-dlp normally already delivers that (see FORMAT_SORT), and then the
    file is used as it is. Otherwise it is converted, which takes a while.
    """
    probe = probe_media(path)

    if probe is None or is_compatible(probe):
        return path

    if on_convert is not None:
        on_convert()

    converted = f"{path}.h264.mp4"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", path,
                "-vf", f"scale=-2:min({TARGET_HEIGHT}\\,ih)",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                converted,
            ],
            capture_output=True, timeout=60 * 60, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DownloadFailed(
            "The video was downloaded but couldn't be converted to a "
            "format this app can use. Upload the video file instead.",
            detail=str(error),
        ) from error

    return converted


# ============================================================
# PROGRESS
# ============================================================

class _DownloadProgress:
    """
    yt-dlp reports every stream (video, then audio) from 0 to 100%.
    This turns that into one overall fraction and only reports when the
    whole-number percentage changes.
    """

    def __init__(self, streams, emit):
        self.streams = max(1, streams)
        self.finished = 0
        self.emit = emit
        self.last_percent = -1

    def __call__(self, event):
        status = event.get("status")

        if status == "finished":
            self.finished += 1
            fraction = self.finished / self.streams

        elif status == "downloading":
            total = (
                event.get("total_bytes")
                or event.get("total_bytes_estimate")
            )
            done = event.get("downloaded_bytes") or 0
            part = min(done / total, 1.0) if total else 0.0
            fraction = (self.finished + part) / self.streams

        else:
            return

        fraction = min(fraction, 1.0)
        percent = int(fraction * 100)

        if percent != self.last_percent:
            self.last_percent = percent
            self.emit("downloading", fraction)


def describe_progress(stage, fraction=None):
    """
    (text, bar value 0..1) for a stage reported by fetch_youtube_video.
    The download takes the middle of the bar; the rest is set-up.
    """
    if stage == "validating":
        return "Validating URL...", 0.02

    if stage == "fetching_info":
        return "Checking the video...", 0.05

    if stage == "downloading":
        part = 0.0 if fraction is None else min(max(fraction, 0.0), 1.0)
        return f"Downloading video... {int(part * 100)}%", 0.05 + 0.85 * part

    if stage == "converting":
        return "Converting the video to a compatible format...", 0.92

    if stage == "cached":
        return "Found a saved copy of this video, no download needed.", 1.0

    return "Download complete.", 1.0


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def _check_video_info(info, max_seconds):
    if not info:
        raise DownloadFailed(_GENERIC_FAILURE, detail="no video information")

    if (
        info.get("is_live")
        or info.get("live_status") in ("is_live", "is_upcoming", "post_live")
    ):
        raise UnsupportedVideo(
            "Live streams and premieres aren't supported. Try again once "
            "the recording is available."
        )

    duration = info.get("duration")

    if duration and duration > max_seconds:
        raise UnsupportedVideo(
            f"This video is {round(duration / 60)} minutes long. Videos "
            f"longer than {round(max_seconds / 60)} minutes aren't "
            "supported yet."
        )


def _find_download(cache_dir, temporary_stem):
    """The finished file yt-dlp wrote for this download (largest wins)."""
    candidates = []

    for name in os.listdir(cache_dir):

        if not name.startswith(temporary_stem + "."):
            continue

        if name.endswith((".part", ".ytdl", ".temp", ".tmp")):
            continue

        candidates.append(os.path.join(cache_dir, name))

    if not candidates:
        raise DownloadFailed(
            _GENERIC_FAILURE, detail="yt-dlp produced no file"
        )

    return max(candidates, key=os.path.getsize)


def _remove_temporary_files(cache_dir, temporary_stem):
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return

    for name in names:
        if name.startswith(temporary_stem):
            try:
                os.remove(os.path.join(cache_dir, name))
            except OSError:
                pass


def fetch_youtube_video(
    url,
    cache_dir=DEFAULT_CACHE_DIR,
    on_progress=None,
    ydl_factory=None,
    max_seconds=MAX_VIDEO_SECONDS,
    make_compatible=ensure_compatible_mp4,
):
    """
    Make a YouTube video available as a local file.

    on_progress(stage, fraction)   optional; stages: validating, cached,
                                   fetching_info, downloading, converting
    ydl_factory                    yt_dlp.YoutubeDL (default); tests pass
                                   a fake

    Returns a YouTubeVideo. Raises a YouTubeError subclass with a
    user_message that can be shown as it is.
    """
    def emit(stage, fraction=None):
        if on_progress is not None:
            on_progress(stage, fraction)

    emit("validating")

    video_id = extract_video_id(url)

    # ---- cache hit: no network, no yt-dlp --------------------------
    cached = find_cached_video(cache_dir, video_id)

    if cached:
        meta = _read_meta(cache_dir, video_id)

        emit("cached", 1.0)

        return YouTubeVideo(
            video_id=video_id,
            path=cached,
            title=meta.get("title") or video_id,
            duration=meta.get("duration") or 0,
            from_cache=True,
        )

    # ---- cache miss: download -----------------------------------
    factory = ydl_factory or _load_yt_dlp()

    os.makedirs(cache_dir, exist_ok=True)

    target = canonical_url(video_id)
    stem = f"tmp-{video_id}-{uuid.uuid4().hex[:8]}"
    template = os.path.join(cache_dir, stem + ".%(ext)s")

    try:
        emit("fetching_info")

        with factory(_base_options()) as ydl:
            info = ydl.extract_info(target, download=False)

        _check_video_info(info, max_seconds)

        progress = _DownloadProgress(
            len(info.get("requested_formats") or []) or 1,
            emit,
        )

        emit("downloading", 0.0)

        with factory(build_download_options(template, progress)) as ydl:
            ydl.download([target])

        downloaded = make_compatible(
            _find_download(cache_dir, stem),
            on_convert=lambda: emit("converting"),
        )

        final_path = cache_path(cache_dir, video_id)

        os.replace(downloaded, final_path)

    except YouTubeError:
        raise

    except Exception as error:
        classified = classify_download_error(error)
        log.info("YouTube download failed: %s", classified.detail)
        raise classified from error

    finally:
        _remove_temporary_files(cache_dir, stem)

    title = info.get("title") or video_id
    duration = info.get("duration") or 0

    _write_meta(cache_dir, video_id, {
        "video_id": video_id,
        "title": title,
        "duration": duration,
    })

    emit("ready", 1.0)

    return YouTubeVideo(
        video_id=video_id,
        path=final_path,
        title=title,
        duration=duration,
        from_cache=False,
    )