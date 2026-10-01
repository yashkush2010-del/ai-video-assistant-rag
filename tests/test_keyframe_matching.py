
from src.processing.keyframe_extractor import match_keyframes_with_transcript


def test_keyframe_matches_transcript():
    keyframes = [
        {
            "timestamp": 10,
            "filename": "frame_0010.jpg"
        }
    ]

    segments = [
        {
            "start": 11,
            "end": 14,
            "text": "Machine learning uses data."
        }
    ]

    matched = match_keyframes_with_transcript(
        keyframes,
        segments,
        tolerance=2.0
    )

    assert len(matched) == 1
    assert matched[0]["transcript_text"] == "Machine learning uses data."
    assert matched[0]["match_gap"] == 1

    print("Keyframe matching test passed!")


if __name__ == "__main__":
    test_keyframe_matches_transcript()