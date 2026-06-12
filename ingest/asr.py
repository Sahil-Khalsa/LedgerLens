"""
Whisper-based ASR transcription for earnings calls and investor presentations.

Transcribes audio/video using the OpenAI Whisper API, then indexes the
transcript as searchable text chunks alongside the related filing.

Usage:
  uv run python -m ingest.asr --file path/to/call.mp3 --accn 0001045810-24-000029
  uv run python -m ingest.asr --url https://example.com/call.mp3 --accn <accn>
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Whisper's maximum upload size is 25 MB.
_MAX_BYTES = 25 * 1024 * 1024


def transcribe_audio(
    url: str | None = None,
    file_path: str | Path | None = None,
    language: str = "en",
) -> str:
    """
    Transcribe an audio file using the OpenAI Whisper API.

    Either url or file_path must be provided.
    Large files are truncated to the first 25 MB before upload.
    Returns the raw transcript string.
    """
    from openai import OpenAI

    if url is None and file_path is None:
        raise ValueError("Either url or file_path must be provided")

    local_path: Path

    if file_path is not None:
        local_path = Path(file_path)
    else:
        import tempfile

        import ingest.edgar as edgar_mod

        logger.info("Downloading audio from %s", url)
        resp = edgar_mod.get(str(url))
        suffix = Path(str(url)).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(resp.content)
            local_path = Path(tmp.name)
        logger.info("Audio saved to %s (%d bytes)", local_path, local_path.stat().st_size)

    size = local_path.stat().st_size
    if size > _MAX_BYTES:
        logger.warning(
            "Audio file %d bytes exceeds Whisper 25 MB limit — truncating", size
        )

    client = OpenAI()
    logger.info("Transcribing %s ...", local_path.name)
    with local_path.open("rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="text",
        )

    transcript = str(response)
    logger.info("Transcribed %d chars from %s", len(transcript), local_path.name)
    return transcript


def transcribe_and_index(
    url: str | None = None,
    file_path: str | Path | None = None,
    filing_accn: str | None = None,
    language: str = "en",
) -> int:
    """
    Transcribe audio and index the transcript as searchable text chunks.

    The transcript is stored in the same chunks table as filing text, scoped
    to the provided accn. If no accn is provided, a synthetic one is generated
    from the file/url name so the transcript is still retrievable.

    Returns the number of chunks indexed.
    """
    from index.text_baseline import index_raw_text

    transcript = transcribe_audio(url=url, file_path=file_path, language=language)
    if not transcript.strip():
        logger.warning("Empty transcript — nothing indexed")
        return 0

    accn = filing_accn or f"asr_{Path(str(url or file_path or 'unknown')).stem}"
    n = index_raw_text(accn, transcript)
    logger.info("Indexed %d transcript chunks under accn=%s", n, accn)
    return n


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio and index as text chunks")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="URL of audio/video file")
    src.add_argument("--file", dest="file_path", help="Local audio/video file path")
    parser.add_argument("--accn", default=None, help="Filing accession number to associate")
    parser.add_argument("--language", default="en", help="Audio language code (default: en)")
    parser.add_argument("--index", action="store_true", help="Index transcript after transcription")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.index:
        n = transcribe_and_index(
            url=args.url,
            file_path=args.file_path,
            filing_accn=args.accn,
            language=args.language,
        )
        print(f"Indexed {n} chunks")
    else:
        transcript = transcribe_audio(
            url=args.url,
            file_path=args.file_path,
            language=args.language,
        )
        print(transcript)


if __name__ == "__main__":
    _cli()
