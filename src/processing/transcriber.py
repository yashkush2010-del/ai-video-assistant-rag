import whisper


def transcribe_audio(audio_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    result = model.transcribe(audio_path)

    return result


if __name__ == "__main__":
    audio_file = "data/audio/sample.wav"

    result = transcribe_audio(audio_file)

    print("\nTimestamped Transcript:")

    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        print(f"[{start:.2f}s - {end:.2f}s] {text}")