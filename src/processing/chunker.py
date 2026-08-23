from transcriber import transcribe_audio


def create_chunks(segments, chunk_size=3):
    chunks = []

    for i in range(0, len(segments), chunk_size):
        chunk_segments = segments[i:i + chunk_size]

        start = chunk_segments[0]["start"]
        end = chunk_segments[-1]["end"]

        text = " ".join(
            segment["text"].strip()
            for segment in chunk_segments
        )

        chunks.append({
            "start": start,
            "end": end,
            "text": text
        })

    return chunks


if __name__ == "__main__":

    audio_file = "data/audio/sample.wav"

    print("Transcribing audio...")

    result = transcribe_audio(audio_file)

    segments = result["segments"]

    chunks = create_chunks(segments)

    print("\nReal Chunks:\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}")
        print(f"Time: {chunk['start']:.2f}s - {chunk['end']:.2f}s")
        print(f"Text: {chunk['text']}")
        print()