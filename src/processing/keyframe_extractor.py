
import cv2
import os


def extract_keyframes(video_path, output_dir, interval=5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps else 0

    keyframes = []
    current_time = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

        success, frame = cap.read()

        if not success:
            break

        filename = f"frame_{int(current_time):04d}.jpg"
        frame_path = os.path.join(output_dir, filename)

        saved = cv2.imwrite(frame_path, frame)

        if not saved:
            raise IOError(f"Could not save frame: {frame_path}")

        keyframes.append({
            "video_name": video_name,
            "timestamp": current_time,
            "image_path": frame_path,
            "filename": filename
        })

        current_time += interval

    cap.release()

    return keyframes



def match_keyframes_with_transcript(
    keyframes,
    segments,
    tolerance=2.0
):
    matched_frames = []

    for frame in keyframes:
        timestamp = frame["timestamp"]

        best_segment = None
        smallest_gap = float("inf")

        for segment in segments:
            start = segment["start"]
            end = segment["end"]

            if start <= timestamp < end:
                best_segment = segment
                smallest_gap = 0
                break

            if timestamp < start:
                gap = start - timestamp
            else:
                gap = timestamp - end

            if gap < smallest_gap:
                smallest_gap = gap
                best_segment = segment

        if best_segment is not None and smallest_gap <= tolerance:
            matched_frames.append({
                **frame,
                "transcript_text": best_segment["text"].strip(),
                "segment_start": best_segment["start"],
                "segment_end": best_segment["end"],
                "match_gap": smallest_gap
            })

    return matched_frames

if __name__ == "__main__":
    frames = extract_keyframes(
        "data/videos/machine_learning.mp4",
        "data/keyframes/machine_learning",
        interval=30
    )

    print("Total keyframes extracted:", len(frames))

    print("\nFirst keyframe metadata:")
    if frames:
        print(frames[0])