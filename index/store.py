"""
SQLAlchemy ORM models and pgvector schema.
Schema is managed by Alembic — never ALTER TABLE by hand.

Tables:
  filings      — one row per filing (10-K, 10-Q, amendments)
  pages        — one row per page image
  chunks       — text chunks with embeddings (text baseline)
  xbrl_facts   — normalized XBRL companyfacts rows
  queries      — query log for cost/latency tracking
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from config import settings

logger = logging.getLogger(__name__)

# ── Engine & session ──────────────────────────────────────────────────────────

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI routes. Use as a context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── ORM models ────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class Filing(Base):
    __tablename__ = "filings"

    accn: Any = Column(String(25), primary_key=True)
    cik: Any = Column(String(10), nullable=False, index=True)
    ticker: Any = Column(String(10), nullable=False, index=True)
    form: Any = Column(String(10), nullable=False)
    filing_date: Any = Column(String(10))
    report_date: Any = Column(String(10))
    primary_doc: Any = Column(Text)
    html_path: Any = Column(Text)
    filing_scale: Any = Column(Integer, default=1)
    page_count: Any = Column(Integer, default=0)
    is_amendment: Any = Column(Boolean, default=False)
    amends_accn: Any = Column(String(25), ForeignKey("filings.accn"), nullable=True)
    is_indexed_text: Any = Column(Boolean, default=False)
    is_indexed_visual: Any = Column(Boolean, default=False)

    pages: Any = relationship("Page", back_populates="filing", cascade="all, delete-orphan")
    chunks: Any = relationship("Chunk", back_populates="filing", cascade="all, delete-orphan")
    xbrl_facts: Any = relationship(
        "XbrlFact", back_populates="filing", cascade="all, delete-orphan"
    )


TEXT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
VISUAL_EMBEDDING_DIM = 128  # ColPali mean-pool dimension


class Page(Base):
    __tablename__ = "pages"

    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    filing_accn: Any = Column(String(25), ForeignKey("filings.accn"), nullable=False, index=True)
    page_idx: Any = Column(Integer, nullable=False)
    png_path: Any = Column(Text)
    is_scanned: Any = Column(Boolean, default=False)
    continues_from_page: Any = Column(Integer, nullable=True)
    continued_on_page: Any = Column(Integer, nullable=True)
    # Mean-pooled ColPali vector for stage-1 retrieval
    pooled_embedding: Any = Column(Vector(VISUAL_EMBEDDING_DIM), nullable=True)

    filing: Any = relationship("Filing", back_populates="pages")

    __table_args__ = (
        UniqueConstraint("filing_accn", "page_idx", name="uq_pages_accn_idx"),
        Index(
            "ix_pages_pooled_hnsw",
            "pooled_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"pooled_embedding": "vector_cosine_ops"},
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    filing_accn: Any = Column(String(25), ForeignKey("filings.accn"), nullable=False, index=True)
    page_idx: Any = Column(Integer, nullable=True)
    chunk_idx: Any = Column(Integer, nullable=False)
    content: Any = Column(Text, nullable=False)
    embedding: Any = Column(Vector(TEXT_EMBEDDING_DIM), nullable=True)

    filing: Any = relationship("Filing", back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class XbrlFact(Base):
    __tablename__ = "xbrl_facts"

    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    cik: Any = Column(String(10), nullable=False, index=True)
    accn: Any = Column(String(25), ForeignKey("filings.accn"), nullable=False, index=True)
    concept: Any = Column(String(200), nullable=False, index=True)
    unit: Any = Column(String(50))
    value: Any = Column(Float, nullable=False)
    start_date: Any = Column(String(10), nullable=True)
    end_date: Any = Column(String(10), nullable=True)
    fy: Any = Column(Integer, nullable=True)
    fp: Any = Column(String(5), nullable=True)
    form: Any = Column(String(10), nullable=True)
    is_flow: Any = Column(Boolean, default=False)

    filing: Any = relationship("Filing", back_populates="xbrl_facts")

    __table_args__ = (Index("ix_xbrl_cik_concept_end", "cik", "concept", "end_date"),)


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Any = Column(Integer, primary_key=True, autoincrement=True)
    thread_id: Any = Column(String(36), nullable=True, index=True)
    question: Any = Column(Text, nullable=False)
    answer: Any = Column(Text, nullable=True)
    answer_status: Any = Column(String(20), nullable=True)
    route: Any = Column(String(20), nullable=True)
    pipeline: Any = Column(String(20), nullable=True)
    cost_usd: Any = Column(Float, nullable=True)
    latency_ms: Any = Column(Integer, nullable=True)
    retries: Any = Column(Integer, default=0)
    created_at: Any = Column(DateTime, server_default=func.now())


# ── Schema helpers ────────────────────────────────────────────────────────────


def create_schema() -> None:
    """Create the pgvector extension and all tables. Idempotent."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    logger.info("Schema created / verified")


def drop_schema() -> None:
    """Drop all tables. Dev only."""
    Base.metadata.drop_all(engine)


# ── Upsert helpers ───────────────────────────────────────────────────────────


def upsert_filing(meta: dict[str, object], db: Session | None = None) -> Filing:
    """
    Insert or update a filing row from ingest metadata.
    meta keys: accn, cik, ticker, form, filing_date, report_date,
               primary_doc, html_path, filing_scale, is_amendment, amends_accn
    """
    own = db is None
    if own:
        db = SessionLocal()
    assert db is not None
    try:
        filing = db.query(Filing).filter(Filing.accn == meta["accn"]).first()
        if filing is None:
            filing = Filing()
            db.add(filing)
        for key, val in meta.items():
            if hasattr(filing, key):
                setattr(filing, key, val)
        db.commit()
        db.refresh(filing)
        return filing
    except Exception:
        db.rollback()
        raise
    finally:
        if own:
            db.close()


def upsert_pages(
    accn: str,
    pages: list[dict[str, object]],
    db: Session | None = None,
) -> None:
    """
    Insert or update page rows from render pipeline output.
    pages: list of {page_idx, png_path, is_scanned}
    Also updates filing.page_count.
    """
    own = db is None
    if own:
        db = SessionLocal()
    assert db is not None
    try:
        for p in pages:
            existing = (
                db.query(Page)
                .filter(Page.filing_accn == accn, Page.page_idx == p["page_idx"])
                .first()
            )
            if existing is None:
                existing = Page(filing_accn=accn)
                db.add(existing)
            existing.page_idx = int(p["page_idx"])  # type: ignore[arg-type]
            existing.png_path = str(p["png_path"])
            existing.is_scanned = bool(p.get("is_scanned", False))

        filing = db.query(Filing).filter(Filing.accn == accn).first()
        if filing:
            filing.page_count = len(pages)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        if own:
            db.close()


def upsert_xbrl_facts(
    cik: str,
    accn: str,
    facts: list[dict[str, object]],
    db: Session | None = None,
) -> int:
    """
    Bulk-insert XBRL facts for a filing, replacing any existing rows for that accn.
    Returns count of rows inserted.
    """
    own = db is None
    if own:
        db = SessionLocal()
    assert db is not None
    try:
        db.query(XbrlFact).filter(XbrlFact.accn == accn).delete()
        rows = [
            XbrlFact(
                cik=cik,
                accn=accn,
                concept=str(f["concept"]),
                unit=str(f.get("unit", "")),
                value=float(f["value"]),  # type: ignore[arg-type]
                start_date=str(f["start"]) if f.get("start") else None,
                end_date=str(f["end"]) if f.get("end") else None,
                fy=int(f["fy"]) if f.get("fy") else None,  # type: ignore[arg-type]
                fp=str(f["fp"]) if f.get("fp") else None,
                form=str(f["form"]) if f.get("form") else None,
                is_flow=bool(f.get("is_flow", False)),
            )
            for f in facts
        ]
        db.bulk_save_objects(rows)
        db.commit()
        return len(rows)
    except Exception:
        db.rollback()
        raise
    finally:
        if own:
            db.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Database schema management")
    parser.add_argument("command", choices=["migrate", "check", "drop"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "migrate":
        create_schema()
        print("Schema ready")
    elif args.command == "check":
        with engine.connect() as conn:
            tables = ["filings", "pages", "chunks", "xbrl_facts", "query_log"]
            for t in tables:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                print(f"  {t}: {count} rows")
    elif args.command == "drop":
        confirm = input("Drop all tables? [yes/N] ")
        if confirm == "yes":
            drop_schema()
            print("Dropped")


if __name__ == "__main__":
    _cli()
