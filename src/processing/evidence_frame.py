"""
A picture from the exact moment of the Q&A evidence.

The stored keyframes are taken every 30 seconds, so the keyframe
closest to an evidence sentence can be up to 15 seconds away and show
something else (real case: evidence at 0:17 about "the basics of
machine learning", nearest keyframe 0:30 "Suppose Paul is listening
to songs"). Instead, one frame is read from the video at the evidence
time. It is saved, so the same moment is never read twice.
"""

import os

import cv2


# A little after the start of the evidence: the very first frame of a
# sentence often still shows the end of the previous scene.
FRAME_OFFSET_SECONDS = 1.0


def frame_time(start, end):
    """The moment to show: start + 1 s, but never past the middle."""
    start = float(start)
    end = float(end)

    return min(start + FRAME_OFFSET_SECONDS, (start + end) / 2)


def extract_frame_at(video_path, seconds, output_dir):
    """
    Saves the frame at `seconds` as a JPG in output_dir and returns its
    path, or None if the video cannot be read there. Never raises.
    """
    seconds = max(0.0, float(seconds))

    path = os.path.join(
        output_dir,
        f"evidence_frame_{int(round(seconds * 10)):06d}.jpg"
    )

    if os.path.exists(path):
        return path

    capture = None

    try:
        os.makedirs(output_dir, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            return None

        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

        success, frame = capture.read()

        if not success or frame is None:
            return None

        if not cv2.imwrite(path, frame):
            return None

        return path

    except (cv2.error, OSError, ValueError, TypeError):
        return None

    finally:
        if capture is not None:
            capture.release()
