from sentence_transformers import SentenceTransformer
from chunker import create_chunks
import whisper


def create_embeddings(texts):
    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text into numerical vectors
    embeddings = model.encode(texts)

    return embeddings


if __name__ == "__main__":

    # Load Whisper model
    print("Transcribing audio...")
    whisper_model = whisper.load_model("base")

    # Transcribe the extracted audio
    result = whisper_model.transcribe("data/audio/sample.wav")

    # Get timestamped segments
    segments = result["segments"]

    # Create chunks from the transcript
    chunks = create_chunks(segments)

    # Take only the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Create embeddings
    embeddings = create_embeddings(texts)

    print("\nEmbedding Results:")
    print("Number of chunks:", len(texts))
    print("Embedding size:", len(embeddings[0]))

    print("\nFirst chunk:")
    print(texts[0])

    print("\nFirst embedding:")
    print(embeddings[0])