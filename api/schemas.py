"""
Request and response schemas for all API endpoints.
"""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    ticker: str | None = None
    filing_accn: str | None = None
    pipeline: Literal["visual", "baseline", "auto"] = "auto"
    thread_id: str | None = None


class FactResult(BaseModel):
    text: str
    value: float | None
    concept: str | None
    page_ref: str
    verified: Literal["match", "mismatch", "unverifiable", "pending"]
    bbox: list[float] | None = None  # [x1%, y1%, x2%, y2%] in 0-1 range


class QueryResponse(BaseModel):
    query_id: str
    answer: str | None
    answer_status: Literal["answered", "low_confidence", "insufficient_data", "error"]
    facts: list[FactResult]
    route: Literal["fast", "document"]
    retries: int
    cost_usd: float
    latency_ms: int
    pipeline: Literal["visual", "baseline"]
    thread_id: str | None


class FilingMeta(BaseModel):
    accn: str
    ticker: str
    form: str
    filing_date: str
    report_date: str
    page_count: int
    is_indexed_visual: bool
    is_indexed_text: bool
    is_amendment: bool


class FilingsResponse(BaseModel):
    items: list[FilingMeta]
    total: int
    page: int
    page_size: int


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    tickers: list[str] = Field(min_length=2, max_length=4)
    period: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db: bool
    details: dict[str, object]


class ErrorResponse(BaseModel):
    error: str
    message: str
    retry_after: int | None = None
