"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TEXT_EMBEDDING_DIM = 384
VISUAL_EMBEDDING_DIM = 128


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "filings",
        sa.Column("accn", sa.String(25), primary_key=True),
        sa.Column("cik", sa.String(10), nullable=False),
        sa.Column("ticker", sa.String(10), nullable=False),
        sa.Column("form", sa.String(10), nullable=False),
        sa.Column("filing_date", sa.String(10)),
        sa.Column("report_date", sa.String(10)),
        sa.Column("primary_doc", sa.Text),
        sa.Column("html_path", sa.Text),
        sa.Column("filing_scale", sa.Integer, server_default="1"),
        sa.Column("page_count", sa.Integer, server_default="0"),
        sa.Column("is_amendment", sa.Boolean, server_default="false"),
        sa.Column("amends_accn", sa.String(25), sa.ForeignKey("filings.accn"), nullable=True),
        sa.Column("is_indexed_text", sa.Boolean, server_default="false"),
        sa.Column("is_indexed_visual", sa.Boolean, server_default="false"),
    )
    op.create_index("ix_filings_cik", "filings", ["cik"])
    op.create_index("ix_filings_ticker", "filings", ["ticker"])

    op.create_table(
        "pages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("filing_accn", sa.String(25), sa.ForeignKey("filings.accn"), nullable=False),
        sa.Column("page_idx", sa.Integer, nullable=False),
        sa.Column("png_path", sa.Text),
        sa.Column("is_scanned", sa.Boolean, server_default="false"),
        sa.Column("continues_from_page", sa.Integer, nullable=True),
        sa.Column("continued_on_page", sa.Integer, nullable=True),
        sa.Column("pooled_embedding", Vector(VISUAL_EMBEDDING_DIM), nullable=True),
    )
    op.create_index("ix_pages_filing_accn", "pages", ["filing_accn"])
    op.create_unique_constraint("uq_pages_accn_idx", "pages", ["filing_accn", "page_idx"])
    op.execute(
        """
        CREATE INDEX ix_pages_pooled_hnsw ON pages
        USING hnsw (pooled_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("filing_accn", sa.String(25), sa.ForeignKey("filings.accn"), nullable=False),
        sa.Column("page_idx", sa.Integer, nullable=True),
        sa.Column("chunk_idx", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(TEXT_EMBEDDING_DIM), nullable=True),
    )
    op.create_index("ix_chunks_filing_accn", "chunks", ["filing_accn"])
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding_hnsw ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    op.create_table(
        "xbrl_facts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(10), nullable=False),
        sa.Column("accn", sa.String(25), sa.ForeignKey("filings.accn"), nullable=False),
        sa.Column("concept", sa.String(200), nullable=False),
        sa.Column("unit", sa.String(50)),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("start_date", sa.String(10), nullable=True),
        sa.Column("end_date", sa.String(10), nullable=True),
        sa.Column("fy", sa.Integer, nullable=True),
        sa.Column("fp", sa.String(5), nullable=True),
        sa.Column("form", sa.String(10), nullable=True),
        sa.Column("is_flow", sa.Boolean, server_default="false"),
    )
    op.create_index("ix_xbrl_cik", "xbrl_facts", ["cik"])
    op.create_index("ix_xbrl_accn", "xbrl_facts", ["accn"])
    op.create_index("ix_xbrl_concept", "xbrl_facts", ["concept"])
    op.create_index("ix_xbrl_cik_concept_end", "xbrl_facts", ["cik", "concept", "end_date"])

    op.create_table(
        "query_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.String(36), nullable=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("answer_status", sa.String(20), nullable=True),
        sa.Column("route", sa.String(20), nullable=True),
        sa.Column("pipeline", sa.String(20), nullable=True),
        sa.Column("cost_usd", sa.Float, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("retries", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_query_log_thread_id", "query_log", ["thread_id"])


def downgrade() -> None:
    op.drop_table("query_log")
    op.drop_table("xbrl_facts")
    op.drop_table("chunks")
    op.drop_table("pages")
    op.drop_table("filings")
