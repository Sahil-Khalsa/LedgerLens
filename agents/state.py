"""
Single typed State schema for the LangGraph agent graph.

Rules:
- All shared state lives here — never in node globals.
- `pages` uses an additive reducer so reflection loops accumulate context.
- Every field has an explicit type; mypy --strict must pass.
"""

from operator import add
from typing import Annotated, Literal, TypedDict


class Fact(TypedDict):
    text: str  # the extracted claim, e.g. "Q3 revenue was $47.5B"
    value: float | None  # parsed numeric value, normalized to raw XBRL units
    concept: str | None  # best-guess XBRL concept name (e.g. "Revenues")
    page_ref: str  # "accn:page_idx" or "accn:xbrl" for fast-path facts
    verified: Literal["match", "mismatch", "unverifiable", "pending"]


class Critique(TypedDict):
    grounded: bool
    ungrounded_claims: list[str]
    missing_evidence: list[str]  # used as refined queries on the next retrieval pass


class PageResult(TypedDict):
    accn: str
    page_idx: int
    png_path: str
    score: float
    # Optional continuation pointers (for multi-page table handling)
    continued_on_page: int | None


class State(TypedDict):
    # Input
    question: str
    ticker: str | None
    filing_accn: str | None  # optional: scope retrieval to one filing
    thread_id: str | None  # for multi-turn checkpointing

    # Planner output
    plan: list[str]  # sub-questions / retrieval queries
    route: Literal["fast", "document"]

    # Retrieval (additive across reflection loops)
    pages: Annotated[list[PageResult], add]

    # Extraction + verification
    facts: list[Fact]

    # Synthesis
    draft: str | None
    answer: str | None
    answer_status: Literal["answered", "low_confidence", "insufficient_data", "error"] | None

    # Critic
    critique: Critique | None
    retries: int

    # Cost tracking
    cost_usd: float
    latency_ms: int
