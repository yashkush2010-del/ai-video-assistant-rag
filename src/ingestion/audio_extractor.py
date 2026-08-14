import subprocess
from pathlib import Path


def extract_audio(video_path, audio_path):
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
            "-y",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise ValueError("Audio extraction failed")

    return str(audio_path)


if __name__ == "__main__":
    audio = extract_audio(
        "data/videos/sample.mp4",
        "data/audio/sample.wav",
    )

    print(f"Audio extracted: {audio}")