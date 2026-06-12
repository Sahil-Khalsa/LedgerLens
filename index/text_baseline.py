"""
Text-only baseline index — deliberately naive.

Purpose: produce a comparison number that the visual pipeline beats.
Do NOT optimize this. Its inaccuracy on numeric extraction is the demonstration
of the failure mode that justifies the whole project.

Pipeline: HTML → extract text → chunk with overlap → embed → store in pgvector.
Retrieval: cosine top-k over chunk embeddings.
"""

import logging
import re
from pathlib import Path
from typing import Any

import tiktoken
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from config import settings
from index.store import Chunk, Filing, SessionLocal

logger = logging.getLogger(__name__)

CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
_enc = tiktoken.get_encoding("cl100k_base")


# ── Embedding model (lazy singleton) ─────────────────────────────────────────

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.embedding_model)
    return _embed_model


# ── Text extraction ───────────────────────────────────────────────────────────


def extract_text_from_html(html_path: str) -> str:
    """
    Extract visible text from an SEC filing HTML.
    Strips scripts, styles, and excessive whitespace.
    """
    import warnings

    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    html = Path(html_path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "meta"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping token-bounded chunks.
    Overlap ensures a value near a chunk boundary is retrievable.
    """
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_tokens_slice = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens_slice))
        if end == len(tokens):
            break
        start += chunk_tokens - overlap_tokens
    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────


def index_filing(accn: str, html_path: str, db: Session | None = None) -> int:
    """
    Extract text, chunk, embed, and store in the chunks table.
    Returns the number of chunks inserted.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    assert db is not None

    try:
        # Clear any existing chunks for this filing (re-index safe)
        db.query(Chunk).filter(Chunk.filing_accn == accn).delete()

        text = extract_text_from_html(html_path)
        if not text:
            logger.warning("No text extracted from %s", html_path)
            return 0

        raw_chunks = chunk_text(text)
        logger.info("Chunked %s into %d chunks", accn, len(raw_chunks))

        model = _get_embed_model()
        embeddings = model.encode(raw_chunks, show_progress_bar=False, normalize_embeddings=True)

        for i, (content, emb) in enumerate(zip(raw_chunks, embeddings, strict=True)):
            db.add(
                Chunk(
                    filing_accn=accn,
                    chunk_idx=i,
                    content=content,
                    embedding=emb.tolist(),
                )
            )

        # Mark filing as text-indexed
        filing = db.query(Filing).filter(Filing.accn == accn).first()
        if filing:
            filing.is_indexed_text = True

        db.commit()
        logger.info("Indexed %d chunks for accn=%s", len(raw_chunks), accn)
        return len(raw_chunks)

    except Exception:
        db.rollback()
        raise
    finally:
        if own_session:
            db.close()


def index_raw_text(accn: str, text: str, db: Session | None = None) -> int:
    """
    Chunk, embed, and store a plain-text document (e.g. earnings-call transcript).
    Unlike index_filing(), this takes raw text instead of an HTML path.
    Returns the number of chunks inserted.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    assert db is not None

    try:
        db.query(Chunk).filter(Chunk.filing_accn == accn).delete()

        if not text.strip():
            logger.warning("Empty text for accn=%s — nothing indexed", accn)
            return 0

        raw_chunks = chunk_text(text)
        logger.info("Chunked transcript %s into %d chunks", accn, len(raw_chunks))

        model = _get_embed_model()
        embeddings = model.encode(raw_chunks, show_progress_bar=False, normalize_embeddings=True)

        for i, (content, emb) in enumerate(zip(raw_chunks, embeddings, strict=True)):
            db.add(
                Chunk(
                    filing_accn=accn,
                    chunk_idx=i,
                    content=content,
                    embedding=emb.tolist(),
                )
            )

        db.commit()
        logger.info("Indexed %d transcript chunks for accn=%s", len(raw_chunks), accn)
        return len(raw_chunks)

    except Exception:
        db.rollback()
        raise
    finally:
        if own_session:
            db.close()


# ── Retrieval ─────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    accn: str | None = None,
    top_k: int = 5,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """
    Cosine top-k retrieval from the text baseline index.
    Optionally scope to a single filing via accn.
    Returns list of {content, accn, chunk_idx, score}.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()
    assert db is not None

    try:
        model = _get_embed_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

        q = db.query(
            Chunk,
            Chunk.embedding.cosine_distance(q_emb).label("distance"),
        )
        if accn:
            q = q.filter(Chunk.filing_accn == accn)
        rows = q.order_by("distance").limit(top_k).all()

        return [
            {
                "content": chunk.content,
                "accn": chunk.filing_accn,
                "chunk_idx": chunk.chunk_idx,
                "score": float(1 - distance),  # cosine similarity
            }
            for chunk, distance in rows
        ]
    finally:
        if own_session:
            db.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the text baseline index")
    parser.add_argument("--accn", help="Accession number to index")
    parser.add_argument("--html", help="Path to HTML file (required with --accn)")
    parser.add_argument("--query", help="Retrieval query")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.accn and args.html:
        n = index_filing(args.accn, args.html)
        print(f"Indexed {n} chunks for {args.accn}")

    if args.query:
        results = retrieve(args.query, accn=args.accn, top_k=args.topk)
        for i, r in enumerate(results):
            print(f"\n--- Result {i + 1} (score={r['score']:.3f}, accn={r['accn']}) ---")
            print(r["content"][:500])


if __name__ == "__main__":
    _cli()
