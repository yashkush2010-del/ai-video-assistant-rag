"""
Transcript chunking for the AI Video Assistant.

The chunker groups Whisper timestamped segments into
small time-based chunks while preserving their original
start/end timestamps.

This keeps the RAG pipeline timestamp-aware.
"""


def create_chunks(
    segments,
    max_duration=20.0
):
    """
    Convert Whisper segments into timestamped chunks.

    Parameters
    ----------
    segments : list
        Whisper timestamped segments.

    max_duration : float
        Maximum approximate duration of one chunk.

    Returns
    -------
    list
        List of dictionaries containing:
        start
        end
        text
        segments
    """

    if not segments:
        return []

    chunks = []

    current_segments = []
    chunk_start = None
    chunk_end = None

    for segment in segments:

        text = segment.get("text", "").strip()

        if not text:
            continue

        start = float(segment.get("start", 0.0))
        end = float(
            segment.get(
                "end",
                start
            )
        )

        # Start a new chunk
        if not current_segments:

            current_segments = [segment]

            chunk_start = start
            chunk_end = end

            continue

        proposed_duration = end - chunk_start

        # Keep adding segments while inside
        # the desired chunk duration.
        if proposed_duration <= max_duration:

            current_segments.append(segment)
            chunk_end = end

        else:

            # Save current chunk
            chunk_text = " ".join(
                item.get("text", "").strip()
                for item in current_segments
                if item.get("text", "").strip()
            )

            if chunk_text:

                chunks.append({
                    "start": float(chunk_start),
                    "end": float(chunk_end),
                    "text": chunk_text,
                    "segments": current_segments.copy()
                })

            # Start new chunk
            current_segments = [segment]

            chunk_start = start
            chunk_end = end

    # Save final chunk
    if current_segments:

        chunk_text = " ".join(
            item.get("text", "").strip()
            for item in current_segments
            if item.get("text", "").strip()
        )

        if chunk_text:

            chunks.append({
                "start": float(chunk_start),
                "end": float(chunk_end),
                "text": chunk_text,
                "segments": current_segments.copy()
            })

    return chunks