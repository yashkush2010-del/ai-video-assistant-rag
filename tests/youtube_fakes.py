"""
A stand-in for yt_dlp.YoutubeDL, so the YouTube tests never touch the
network. It follows the real interface the loader uses:

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        ydl.download([url])           # writes options["outtmpl"], calls
                                      # the progress hooks, raises on error
"""

import os
import shutil
import types


class DownloadError(Exception):
    """Same name as yt_dlp.utils.DownloadError."""


DEFAULT_INFO = {
    "id": "aqz-KE-bpKQ",
    "title": "A Machine Learning Lecture",
    "duration": 240,
    "is_live": False,
    "live_status": "not_live",
    # video stream + audio stream, as for a normal 480p download
    "requested_formats": [{"format_id": "v"}, {"format_id": "a"}],
}


def make_fake_ydl(
    source_file=None,
    info=None,
    no_info=False,
    extract_error=None,
    download_error=None,
    produce_file=True,
    leave_partial_file=False,
    output_extension="mp4",
):
    """Returns (FakeYoutubeDL class, log). The log records every call."""

    log = {"options": [], "extract": [], "download": []}

    class FakeYoutubeDL:

        def __init__(self, options):
            self.options = options
            log["options"].append(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, url, download=False):
            log["extract"].append((url, download))

            if extract_error is not None:
                raise extract_error

            if no_info:
                return None

            return {**DEFAULT_INFO, **(info or {})}

        def download(self, urls):
            log["download"].append(list(urls))

            template = self.options["outtmpl"]
            hooks = self.options.get("progress_hooks", [])

            if leave_partial_file:
                with open(template.replace("%(ext)s", "mp4.part"), "wb") as f:
                    f.write(b"half a download")

            for hook in hooks:
                hook({"status": "downloading",
                      "downloaded_bytes": 50, "total_bytes": 100})

            if download_error is not None:
                raise download_error

            for hook in hooks:
                hook({"status": "finished"})

            if produce_file:
                shutil.copyfile(
                    source_file,
                    template.replace("%(ext)s", output_extension),
                )

            for hook in hooks:
                hook({"status": "downloading",
                      "downloaded_bytes": 50, "total_bytes": 100})
                hook({"status": "finished"})

            return 0

    return FakeYoutubeDL, log


def fake_yt_dlp_module(factory):
    """A module that can be put in sys.modules as 'yt_dlp'."""
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = factory
    return module


def files_in(directory):
    return sorted(os.listdir(directory)) if os.path.isdir(directory) else []