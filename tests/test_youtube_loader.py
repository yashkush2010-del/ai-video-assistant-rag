import os
import sys

import pytest

import youtube_fakes as fakes
from src.ingestion import youtube_loader as loader
from src.ingestion.youtube_loader import (
    DEFAULT_CACHE_DIR,
    DownloadFailed,
    DownloaderMissing,
    InvalidYouTubeURL,
    UnsupportedVideo,
    VideoUnavailable,
    YouTubeError,
    YouTubeVideoSource,
    canonical_url,
    classify_download_error,
    describe_progress,
    extract_video_id,
    fetch_youtube_video,
    find_cached_video,
)

VIDEO_ID = "aqz-KE-bpKQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"

FAKE_MP4 = b"not really a video, but big enough to count " * 100


def no_conversion(path, on_convert=None):
    """Skip the ffprobe / ffmpeg step: these tests only use fake bytes."""
    return path


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "downloaded.mp4"
    path.write_bytes(FAKE_MP4)
    return path


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "youtube")


def fetch(url, cache_dir, factory, **kwargs):
    kwargs.setdefault("make_compatible", no_conversion)
    return fetch_youtube_video(
        url, cache_dir=cache_dir, ydl_factory=factory, **kwargs
    )


# ============================================================
# valid URL / extracting the video ID
# ============================================================

@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VIDEO_ID}",
    f"http://www.youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}",
    f"https://m.youtube.com/watch?v={VIDEO_ID}",
    f"https://music.youtube.com/watch?v={VIDEO_ID}",
    f"www.youtube.com/watch?v={VIDEO_ID}",
    f"youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com/watch?v={VIDEO_ID}&t=387s",
    f"https://www.youtube.com/watch?feature=share&v={VIDEO_ID}",
    f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLabcdefghijklmnop&index=2",
    f"https://www.youtube.com/watch/?v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}?t=387",
    f"youtu.be/{VIDEO_ID}",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/embed/{VIDEO_ID}",
    f"https://www.youtube.com/live/{VIDEO_ID}",
    f"https://www.youtube.com/v/{VIDEO_ID}",
    f"  https://www.youtube.com/watch?v={VIDEO_ID}\n",
    f"HTTPS://WWW.YOUTUBE.COM/watch?v={VIDEO_ID}",
    f"https://www.youtube.com:443/watch?v={VIDEO_ID}",
])
def test_valid_youtube_urls_give_the_video_id(url):
    assert extract_video_id(url) == VIDEO_ID


def test_every_kind_of_valid_id_character_is_accepted():
    assert extract_video_id("https://youtu.be/AbC_-12345z") == "AbC_-12345z"


def test_the_canonical_url_carries_nothing_but_the_video():
    assert canonical_url(VIDEO_ID) == URL


# ============================================================
# invalid URL
# ============================================================

@pytest.mark.parametrize("url", [
    "",
    "   ",
    None,
    123,
    "hello",
    "not a url at all",
    f"https://vimeo.com/watch?v={VIDEO_ID}",
    f"https://example.com/{VIDEO_ID}",
    # look-alike hosts
    f"https://youtube.com.evil.com/watch?v={VIDEO_ID}",
    f"https://evilyoutube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com.evil.com/watch?v={VIDEO_ID}",
    f"https://youtu.be.evil.com/{VIDEO_ID}",
    # credentials in the URL
    f"https://user:pw@www.youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com@evil.com/watch?v={VIDEO_ID}",
    # wrong scheme / port
    f"ftp://www.youtube.com/watch?v={VIDEO_ID}",
    "file:///etc/passwd",
    "javascript:alert(1)",
    f"https://www.youtube.com:8080/watch?v={VIDEO_ID}",
    # not a single video
    "https://www.youtube.com/",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=",
    "https://www.youtube.com/@somechannel",
    "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv",
    "https://www.youtube.com/results?search_query=machine+learning",
    "https://youtu.be/",
    # a broken ID
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/watch?v=waytoolongforanid",
    "https://www.youtube.com/watch?v=aqz-KE-bpK!",
    "https://www.youtube.com/watch?v=aqz KE bpKQ",
    "https://www.youtube.com/shorts/",
])
def test_invalid_urls_are_rejected(url):
    with pytest.raises(InvalidYouTubeURL) as caught:
        extract_video_id(url)

    assert caught.value.user_message


def test_a_playlist_link_gets_a_specific_message():
    with pytest.raises(InvalidYouTubeURL) as caught:
        extract_video_id("https://www.youtube.com/playlist?list=PLabcdefghijklmnop")

    assert "Playlist" in caught.value.user_message


def test_an_empty_box_asks_for_a_link():
    with pytest.raises(InvalidYouTubeURL) as caught:
        extract_video_id("")

    assert "Paste a YouTube link" in caught.value.user_message


def test_an_invalid_url_never_reaches_yt_dlp_or_the_disk(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)

    with pytest.raises(InvalidYouTubeURL):
        fetch("https://vimeo.com/123456", cache_dir, factory)

    assert log["options"] == []
    assert not os.path.exists(cache_dir)


# ============================================================
# cache miss, then cache hit
# ============================================================

def test_cache_miss_downloads_and_stores_under_the_video_id(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)

    video = fetch(URL, cache_dir, factory)

    assert video.video_id == VIDEO_ID
    assert video.from_cache is False
    assert video.path == os.path.join(cache_dir, f"{VIDEO_ID}.mp4")
    assert open(video.path, "rb").read() == FAKE_MP4
    assert video.title == "A Machine Learning Lecture"
    assert video.duration == 240

    assert len(log["download"]) == 1


def test_cache_hit_does_not_download_again(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)

    first = fetch(URL, cache_dir, factory)
    second = fetch(URL, cache_dir, factory)

    assert second.from_cache is True
    assert second.path == first.path
    assert second.video_id == VIDEO_ID

    # the title is remembered, and yt-dlp was used exactly once
    assert second.title == "A Machine Learning Lecture"
    assert len(log["download"]) == 1
    assert len(log["extract"]) == 1


def test_cache_hit_needs_no_yt_dlp_and_no_network(cache_dir, source_file, monkeypatch):
    factory, _ = fakes.make_fake_ydl(source_file)
    fetch(URL, cache_dir, factory)

    # yt-dlp not installed at all: a cached video still works
    monkeypatch.setitem(sys.modules, "yt_dlp", None)

    video = fetch_youtube_video(URL, cache_dir=cache_dir)

    assert video.from_cache is True


def test_a_different_link_to_the_same_video_hits_the_same_cache(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)

    fetch(URL, cache_dir, factory)
    again = fetch(f"https://youtu.be/{VIDEO_ID}?t=100", cache_dir, factory)

    assert again.from_cache is True
    assert len(log["download"]) == 1


def test_different_videos_do_not_share_a_cache_entry(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)

    a = fetch(URL, cache_dir, factory)
    b = fetch("https://youtu.be/BBBBBBBBBBB", cache_dir, factory)

    assert a.path != b.path
    assert b.from_cache is False
    assert len(log["download"]) == 2


def test_a_missing_title_file_still_gives_a_cache_hit(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    fetch(URL, cache_dir, factory)

    os.remove(os.path.join(cache_dir, f"{VIDEO_ID}.json"))

    video = fetch(URL, cache_dir, factory)

    assert video.from_cache is True
    assert video.title == VIDEO_ID


def test_an_empty_cached_file_counts_as_a_miss(cache_dir, source_file):
    os.makedirs(cache_dir)
    open(os.path.join(cache_dir, f"{VIDEO_ID}.mp4"), "wb").close()

    assert find_cached_video(cache_dir, VIDEO_ID) is None

    factory, log = fakes.make_fake_ydl(source_file)
    video = fetch(URL, cache_dir, factory)

    assert video.from_cache is False
    assert len(log["download"]) == 1
    assert open(video.path, "rb").read() == FAKE_MP4


def test_a_deleted_cache_file_is_downloaded_again(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)
    first = fetch(URL, cache_dir, factory)

    os.remove(first.path)

    second = fetch(URL, cache_dir, factory)

    assert second.from_cache is False
    assert len(log["download"]) == 2


def test_only_the_video_and_its_title_are_left_in_the_cache(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)

    fetch(URL, cache_dir, factory)

    assert fakes.files_in(cache_dir) == [f"{VIDEO_ID}.json", f"{VIDEO_ID}.mp4"]


def test_the_cache_is_not_the_uploaded_videos_folder():
    cache = os.path.normpath(DEFAULT_CACHE_DIR)
    uploads = os.path.normpath(os.path.join("data", "videos"))

    assert cache != uploads
    assert not cache.startswith(uploads + os.sep)
    assert os.path.dirname(cache) == os.path.dirname(uploads)


# ============================================================
# how yt-dlp is called
# ============================================================

def test_yt_dlp_is_given_one_canonical_url_and_quiet_small_options(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file)

    # a link that also carries a playlist must still download ONE video
    fetch(
        f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLabcdefghijklmnop&index=3",
        cache_dir, factory,
    )

    assert log["extract"] == [(URL, False)]
    assert log["download"] == [[URL]]

    options = log["options"][-1]

    assert options["noplaylist"] is True
    assert options["quiet"] is True
    assert options["no_warnings"] is True
    assert options["noprogress"] is True
    assert options["merge_output_format"] == "mp4"
    assert options["format_sort"][0] == "vcodec:h264"
    assert "res:480" in options["format_sort"]
    assert os.path.dirname(options["outtmpl"]) == cache_dir


def test_yt_dlp_console_output_is_silenced(cache_dir, source_file, capfd):
    factory, log = fakes.make_fake_ydl(source_file)

    fetch(URL, cache_dir, factory)

    logger = log["options"][-1]["logger"]

    for method in ("debug", "info", "warning", "error"):
        getattr(logger, method)("[download]  42.0% of 85.10MiB at 1.2MiB/s")

    out, err = capfd.readouterr()

    assert out == ""
    assert err == ""


def test_every_option_we_use_is_a_real_yt_dlp_option():
    yt_dlp = pytest.importorskip("yt_dlp")

    options = loader.build_download_options("x.%(ext)s", progress_hook=lambda d: None)
    documented = yt_dlp.YoutubeDL.__doc__

    for name in options:
        assert name in documented, f"yt-dlp does not document the option {name!r}"


# ============================================================
# progress
# ============================================================

def test_progress_goes_through_the_expected_stages(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    events = []

    fetch(URL, cache_dir, factory, on_progress=lambda s, f=None: events.append((s, f)))

    stages = [stage for stage, _ in events]

    assert stages[0] == "validating"
    assert stages[1] == "fetching_info"
    assert "downloading" in stages
    assert stages[-1] == "ready"

    # never goes backwards, and ends complete
    fractions = [f for s, f in events if s == "downloading"]

    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0


def test_two_streams_are_reported_as_one_download(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    events = []

    fetch(URL, cache_dir, factory, on_progress=lambda s, f=None: events.append((s, f)))

    # video then audio, each 0-100%, shown as one 0-100% bar
    assert [f for s, f in events if s == "downloading"] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_a_cache_hit_reports_validating_then_cached(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    fetch(URL, cache_dir, factory)

    events = []
    fetch(URL, cache_dir, factory, on_progress=lambda s, f=None: events.append(s))

    assert events == ["validating", "cached"]


def test_describe_progress_gives_readable_text_and_a_valid_bar():
    assert describe_progress("validating")[0] == "Validating URL..."
    assert describe_progress("downloading", 0.37)[0] == "Downloading video... 37%"
    assert describe_progress("cached")[1] == 1.0

    for stage, fraction in [
        ("validating", None), ("fetching_info", None), ("downloading", 0.0),
        ("downloading", 0.5), ("downloading", 1.0), ("downloading", 7.0),
        ("downloading", -3.0), ("converting", None), ("cached", 1.0),
        ("ready", 1.0),
    ]:
        text, bar = describe_progress(stage, fraction)

        assert text
        assert 0.0 <= bar <= 1.0


# ============================================================
# failed download
# ============================================================

def test_a_network_failure_is_a_readable_error_and_leaves_nothing_behind(cache_dir, source_file):
    error = fakes.DownloadError(
        "ERROR: [youtube] aqz-KE-bpKQ: Unable to download API page: "
        "HTTPSConnection(host='www.youtube.com', port=443): Failed to "
        "resolve 'www.youtube.com' ([Errno -2] Name or service not known)"
    )
    factory, _ = fakes.make_fake_ydl(source_file, extract_error=error)

    with pytest.raises(DownloadFailed) as caught:
        fetch(URL, cache_dir, factory)

    assert "internet connection" in caught.value.user_message
    assert "ERROR" not in caught.value.user_message
    assert "HTTPSConnection" not in caught.value.user_message
    assert "HTTPSConnection" in caught.value.detail

    assert find_cached_video(cache_dir, VIDEO_ID) is None
    assert fakes.files_in(cache_dir) == []


def test_a_failure_in_the_middle_of_a_download_leaves_no_partial_file(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(
        source_file,
        download_error=fakes.DownloadError("ERROR: unable to download video data: HTTP Error 403: Forbidden"),
        leave_partial_file=True,
    )

    with pytest.raises(DownloadFailed):
        fetch(URL, cache_dir, factory)

    assert fakes.files_in(cache_dir) == []
    assert find_cached_video(cache_dir, VIDEO_ID) is None


def test_after_a_failed_download_the_next_try_downloads_again(cache_dir, source_file):
    failing, _ = fakes.make_fake_ydl(
        source_file, download_error=fakes.DownloadError("boom")
    )
    working, log = fakes.make_fake_ydl(source_file)

    with pytest.raises(DownloadFailed):
        fetch(URL, cache_dir, failing)

    video = fetch(URL, cache_dir, working)

    assert video.from_cache is False
    assert len(log["download"]) == 1


def test_a_download_that_produces_no_file_is_a_failure(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file, produce_file=False)

    with pytest.raises(DownloadFailed):
        fetch(URL, cache_dir, factory)

    assert find_cached_video(cache_dir, VIDEO_ID) is None


def test_missing_video_information_is_a_failure(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file, no_info=True)

    with pytest.raises(DownloadFailed):
        fetch(URL, cache_dir, factory)

    assert log["download"] == []


def test_a_disk_error_gets_its_own_message(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(
        source_file, download_error=OSError(28, "No space left on device")
    )

    with pytest.raises(DownloadFailed) as caught:
        fetch(URL, cache_dir, factory)

    assert "disk space" in caught.value.user_message


def test_yt_dlp_not_installed_says_how_to_install_it(cache_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)

    with pytest.raises(DownloaderMissing) as caught:
        fetch_youtube_video(URL, cache_dir=cache_dir)

    assert "pip install yt-dlp" in caught.value.user_message


def test_ui_control_exceptions_are_not_swallowed(cache_dir, source_file):
    """Streamlit stops a script with a BaseException; it must get through."""

    class StopScript(BaseException):
        pass

    def stop(stage, fraction=None):
        if stage == "downloading":
            raise StopScript()

    factory, _ = fakes.make_fake_ydl(source_file)

    with pytest.raises(StopScript):
        fetch(URL, cache_dir, factory, on_progress=stop)

    assert fakes.files_in(cache_dir) == []


def test_a_real_yt_dlp_failure_is_handled_and_stays_silent(cache_dir, capfd):
    """
    The real yt-dlp, pointed at a proxy that refuses connections, so it
    fails the same way a lost connection does, with or without internet.
    """
    yt_dlp = pytest.importorskip("yt_dlp")

    def offline(options):
        return yt_dlp.YoutubeDL({**options, "proxy": "http://127.0.0.1:9", "retries": 0})

    with pytest.raises(DownloadFailed) as caught:
        fetch_youtube_video(URL, cache_dir=cache_dir, ydl_factory=offline)

    out, err = capfd.readouterr()

    assert out == "" and err == ""
    assert "internet connection" in caught.value.user_message
    assert fakes.files_in(cache_dir) == []


# ============================================================
# unsupported / private / unavailable videos
# ============================================================

@pytest.mark.parametrize("message, error_class, expected_words", [
    ("ERROR: [youtube] aqz-KE-bpKQ: Private video. Sign in if you've been granted access to this video",
     VideoUnavailable, "private"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Video unavailable",
     VideoUnavailable, "unavailable"),
    ("ERROR: [youtube] aqz-KE-bpKQ: This video has been removed by the uploader",
     VideoUnavailable, "unavailable"),
    ("ERROR: [youtube] aqz-KE-bpKQ: This video is no longer available because the YouTube account associated with this video has been terminated.",
     VideoUnavailable, "unavailable"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Sign in to confirm your age. This video may be inappropriate for some users.",
     VideoUnavailable, "age-restricted"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Join this channel to get access to members-only content like this video, and other exclusive perks.",
     VideoUnavailable, "channel members"),
    ("ERROR: [youtube] aqz-KE-bpKQ: The uploader has not made this video available in your country",
     VideoUnavailable, "country"),
    ("ERROR: [youtube] aqz-KE-bpKQ: This live event will begin in 2 hours.",
     UnsupportedVideo, "Live streams"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Premieres in 3 hours",
     UnsupportedVideo, "premieres"),
    # note the curly apostrophe YouTube actually uses
    ("ERROR: [youtube] aqz-KE-bpKQ: Sign in to confirm you\u2019re not a bot. Use --cookies-from-browser or --cookies for the authentication.",
     DownloadFailed, "sign-in check"),
    ("ERROR: unable to download video data: HTTP Error 429: Too Many Requests",
     DownloadFailed, "too many requests"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Requested format is not available. Use --list-formats for a list of available formats",
     DownloadFailed, "pip install -U yt-dlp"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>",
     DownloadFailed, "internet connection"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Unable to download API page: HTTPSConnection(host='www.youtube.com', port=443): Failed to resolve 'www.youtube.com' ([Errno -2] Name or service not known) (caused by TransportError(\"HTTPSConnection\"))",
     DownloadFailed, "internet connection"),
    ("ERROR: [youtube] aqz-KE-bpKQ: Unable to download webpage: The read operation timed out",
     DownloadFailed, "internet connection"),
    # YouTube answered, but refused: NOT a connection problem
    ("ERROR: unable to download video data: HTTP Error 403: Forbidden",
     DownloadFailed, "refused the download"),
    ("something nobody has ever seen before",
     DownloadFailed, "try again"),
])
def test_unsupported_private_and_unavailable_videos_get_clear_messages(
    cache_dir, source_file, message, error_class, expected_words
):
    factory, log = fakes.make_fake_ydl(
        source_file, extract_error=fakes.DownloadError(message)
    )

    with pytest.raises(error_class) as caught:
        fetch(URL, cache_dir, factory)

    shown = caught.value.user_message

    assert expected_words.lower() in shown.lower()

    # never the raw yt-dlp text
    assert "ERROR:" not in shown
    assert "[youtube]" not in shown
    assert message not in shown
    assert caught.value.detail == message

    # nothing was downloaded or saved
    assert log["download"] == []
    assert fakes.files_in(cache_dir) == []


@pytest.mark.parametrize("info", [
    {"is_live": True},
    {"live_status": "is_live"},
    {"live_status": "is_upcoming"},
    {"live_status": "post_live"},
])
def test_live_streams_are_refused_before_downloading(cache_dir, source_file, info):
    factory, log = fakes.make_fake_ydl(source_file, info=info)

    with pytest.raises(UnsupportedVideo) as caught:
        fetch(URL, cache_dir, factory)

    assert "Live streams" in caught.value.user_message
    assert log["download"] == []


def test_a_video_longer_than_the_limit_is_refused_before_downloading(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file, info={"duration": 61 * 60})

    with pytest.raises(UnsupportedVideo) as caught:
        fetch(URL, cache_dir, factory)

    assert "61 minutes" in caught.value.user_message
    assert "60 minutes" in caught.value.user_message
    assert log["download"] == []


def test_a_video_exactly_at_the_limit_is_accepted(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file, info={"duration": 60 * 60})

    assert fetch(URL, cache_dir, factory).from_cache is False


def test_the_length_limit_can_be_changed(cache_dir, source_file):
    factory, log = fakes.make_fake_ydl(source_file, info={"duration": 11 * 60})

    with pytest.raises(UnsupportedVideo):
        fetch(URL, cache_dir, factory, max_seconds=10 * 60)

    assert log["download"] == []


def test_every_error_is_a_youtube_error_with_a_message():
    for error_class in (
        InvalidYouTubeURL, VideoUnavailable, UnsupportedVideo,
        DownloadFailed, DownloaderMissing,
    ):
        assert issubclass(error_class, YouTubeError)

        error = error_class("something readable", detail="raw")

        assert error.user_message == "something readable"
        assert error.detail == "raw"
        assert str(error) == "something readable"


def test_classify_leaves_our_own_errors_alone():
    original = VideoUnavailable("already readable")

    assert classify_download_error(original) is original


# ============================================================
# format choice: yt-dlp's own engine against realistic format lists
# ============================================================

def _format(format_id, ext, height=None, video="none", audio="none", bitrate=500):
    data = {
        "format_id": format_id, "ext": ext, "vcodec": video, "acodec": audio,
        "tbr": bitrate, "protocol": "https",
        "url": f"https://example.invalid/{format_id}",
    }

    if height:
        data.update(height=height, width=int(height * 16 / 9))

    return data


def _ladder(*heights, h264=True, vp9=True):
    formats = []

    for height in heights:
        if h264:
            formats.append(_format(f"avc-{height}", "mp4", height, "avc1.640028", bitrate=height * 2))
        if vp9:
            formats.append(_format(f"vp9-{height}", "webm", height, "vp09.00.40.08", bitrate=height * 1.2))

    return formats


AUDIO = [
    _format("140", "m4a", audio="mp4a.40.2", bitrate=130),
    _format("251", "webm", audio="opus", bitrate=130),
]


def _choose(formats):
    yt_dlp = pytest.importorskip("yt_dlp")

    options = {
        "format": loader.FORMAT,
        "format_sort": list(loader.FORMAT_SORT),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    info = {
        "id": VIDEO_ID, "title": "t", "extractor": "youtube",
        "extractor_key": "Youtube", "webpage_url": URL, "duration": 100,
        "formats": [dict(f) for f in formats],
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            result = ydl.process_video_result(info, download=False)
    except (AttributeError, TypeError) as error:
        pytest.skip(f"this yt-dlp version changed its internals: {error}")

    picked = result.get("requested_formats") or [result]

    return {f["format_id"] for f in picked}


def test_a_normal_lecture_gets_480p_h264_with_aac_audio():
    assert _choose(_ladder(144, 240, 360, 480, 720, 1080, 1440, 2160) + AUDIO) == {"avc-480", "140"}


def test_a_video_with_nothing_at_or_below_480p_gets_the_smallest_larger_one():
    assert _choose(_ladder(720, 1080) + AUDIO) == {"avc-720", "140"}


def test_a_low_resolution_video_gets_what_exists():
    assert _choose(_ladder(240, 360) + AUDIO) == {"avc-360", "140"}


def test_h264_is_preferred_over_a_higher_vp9_so_no_conversion_is_needed():
    formats = [f for f in _ladder(360, 480, 720) if f["format_id"] != "avc-480"]

    assert _choose(formats + AUDIO) == {"avc-360", "140"}


def test_a_vp9_only_video_still_downloads():
    chosen = _choose(_ladder(360, 480, 720, h264=False) + AUDIO)

    assert chosen == {"vp9-480", "140"}


# ============================================================
# compatibility: real ffmpeg / OpenCV
# ============================================================

def test_a_compatible_file_is_left_exactly_as_it_is(tiny_video):
    assert loader.ensure_compatible_mp4(str(tiny_video)) == str(tiny_video)


def test_a_vp9_webm_is_converted_to_a_small_h264_mp4_that_opencv_can_read(vp9_video, tmp_path):
    cv2 = pytest.importorskip("cv2")

    copy = tmp_path / "from_youtube.webm"
    copy.write_bytes(vp9_video.read_bytes())

    converted = []

    result = loader.ensure_compatible_mp4(
        str(copy), on_convert=lambda: converted.append(True)
    )

    probe = loader.probe_media(result)

    assert converted == [True]
    assert result != str(copy)
    assert loader.is_compatible(probe)
    assert probe["video"] == "h264"

    capture = cv2.VideoCapture(result)

    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) <= loader.TARGET_HEIGHT


def test_a_downloaded_webm_ends_up_as_a_compatible_cached_mp4(vp9_video, cache_dir):
    factory, _ = fakes.make_fake_ydl(vp9_video, output_extension="webm")
    stages = []

    video = fetch_youtube_video(
        URL, cache_dir=cache_dir, ydl_factory=factory,
        on_progress=lambda s, f=None: stages.append(s),
    )

    assert video.path == os.path.join(cache_dir, f"{VIDEO_ID}.mp4")
    assert loader.is_compatible(loader.probe_media(video.path))
    assert "converting" in stages
    assert fakes.files_in(cache_dir) == [f"{VIDEO_ID}.json", f"{VIDEO_ID}.mp4"]


def test_a_failed_conversion_is_a_readable_error(vp9_video, cache_dir, monkeypatch):
    factory, _ = fakes.make_fake_ydl(vp9_video, output_extension="webm")

    real_run = loader.subprocess.run

    def broken_ffmpeg(command, *args, **kwargs):
        if command[0] == "ffmpeg":
            raise OSError("ffmpeg crashed")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(loader.subprocess, "run", broken_ffmpeg)

    with pytest.raises(DownloadFailed) as caught:
        fetch_youtube_video(URL, cache_dir=cache_dir, ydl_factory=factory)

    assert "couldn't be converted" in caught.value.user_message
    assert fakes.files_in(cache_dir) == []


def test_without_ffprobe_the_file_is_used_as_it_is(monkeypatch):
    monkeypatch.setattr(loader, "probe_media", lambda path: None)

    assert loader.ensure_compatible_mp4("whatever.mp4") == "whatever.mp4"


# ============================================================
# the object the existing pipeline receives
# ============================================================

def test_the_source_looks_like_an_uploaded_file(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    video = fetch(URL, cache_dir, factory)

    source = YouTubeVideoSource(video)

    assert source.name == f"youtube_{VIDEO_ID}.mp4"
    assert source.path == video.path
    assert source.getvalue() == FAKE_MP4
    assert "downloaded" in source.ready_message


def test_the_source_reads_the_file_only_once(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    source = YouTubeVideoSource(fetch(URL, cache_dir, factory))

    first = source.getvalue()
    os.remove(source.path)

    assert source.getvalue() is first


def test_a_cached_source_says_so(cache_dir, source_file):
    factory, _ = fakes.make_fake_ydl(source_file)
    fetch(URL, cache_dir, factory)

    source = YouTubeVideoSource(fetch(URL, cache_dir, factory))

    assert "cache" in source.ready_message


@pytest.mark.parametrize("message", [
    "ERROR: [youtube] aqz-KE-bpKQ: Unable to download webpage: HTTP Error 404: Not Found",
    "ERROR: [youtube] aqz-KE-bpKQ: Unable to download webpage: HTTP Error 500: Internal Server Error",
])
def test_a_server_error_is_not_blamed_on_the_users_internet_connection(message):
    """YouTube answered, so the connection is fine: only real network failures say otherwise."""
    shown = classify_download_error(fakes.DownloadError(message)).user_message

    assert "internet connection" not in shown
    assert "try again" in shown