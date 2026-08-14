import cv2


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps else 0

    cap.release()

    return {
        "path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }

if __name__ == "__main__":
    video = load_video(
        "data/videos/sample.mp4"
    )
    print(video)