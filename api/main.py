"""
FastAPI application — all routes, middleware, and SSE streaming.

Routes:
  POST /query          — submit a question, returns SSE stream
  GET  /query/{id}     — fetch a completed query result
  GET  /filings        — list indexed filings
  GET  /filings/{accn} — single filing metadata
  GET  /pages/{accn}/{page_idx} — serve page PNG for citation display
  GET  /health         — db + model connectivity check
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import verify_api_key
from api.schemas import (
    FilingMeta,
    FilingsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from api.validation import sanitize_question
from config import settings
from index.store import Filing, SessionLocal

logger = logging.getLogger(__name__)

# ── App + middleware ──────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="LedgerLens",
    description="Multimodal financial-filing intelligence engine",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_baseline(req: QueryRequest, query_id: str) -> AsyncIterator[str]:
    """Run the text baseline pipeline and stream status + result events."""
    from index.text_baseline import retrieve
    from openai import AsyncOpenAI

    yield _sse("status", {"stage": "retriever", "pipeline": "baseline"})

    chunks = await asyncio.to_thread(retrieve, req.question, req.filing_accn, 5)
    context = "\n\n".join(c["content"] for c in chunks)

    yield _sse("status", {"stage": "synthesizer", "chunks_found": len(chunks)})

    if not context.strip():
        result = QueryResponse(
            query_id=query_id,
            answer=None,
            answer_status="insufficient_data",
            facts=[],
            route="document",
            retries=0,
            cost_usd=0.0,
            latency_ms=0,
            pipeline="baseline",
            thread_id=req.thread_id,
        )
        yield _sse("result", result.model_dump())
        yield _sse("done", {})
        return

    client = AsyncOpenAI()
    prompt = (
        "You are a financial analyst. Answer using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"<user_question>{req.question}</user_question>\n\n"
        "State the exact figure if present. If not found, say 'not found'."
    )

    answer_tokens: list[str] = []
    async with client.chat.completions.stream(
        model=settings.planner_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                answer_tokens.append(delta)
                yield _sse("token", {"text": delta})

    answer = "".join(answer_tokens)
    status = "insufficient_data" if "not found" in answer.lower() else "answered"

    result = QueryResponse(
        query_id=query_id,
        answer=answer,
        answer_status=status,
        facts=[],
        route="document",
        retries=0,
        cost_usd=0.0,
        latency_ms=0,
        pipeline="baseline",
        thread_id=req.thread_id,
    )
    yield _sse("result", result.model_dump())
    yield _sse("done", {})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/query")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query(
    request: QueryRequest,
    _key: str = Depends(verify_api_key),
) -> StreamingResponse:
    try:
        request.question = sanitize_question(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    query_id = str(uuid.uuid4())
    pipeline = request.pipeline if request.pipeline != "auto" else "baseline"

    async def event_stream() -> AsyncIterator[str]:
        if pipeline == "baseline":
            async for event in _stream_baseline(request, query_id):
                yield event
        else:
            yield _sse("result", {
                "query_id": query_id,
                "answer": None,
                "answer_status": "insufficient_data",
                "facts": [],
                "route": "document",
                "retries": 0,
                "cost_usd": 0.0,
                "latency_ms": 0,
                "pipeline": pipeline,
                "thread_id": request.thread_id,
            })
            yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/filings")
async def list_filings(
    ticker: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _key: str = Depends(verify_api_key),
) -> FilingsResponse:
    db = SessionLocal()
    try:
        q = db.query(Filing)
        if ticker:
            q = q.filter(Filing.ticker == ticker.upper())
        total = q.count()
        filings = q.offset((page - 1) * page_size).limit(page_size).all()
        return FilingsResponse(
            items=[
                FilingMeta(
                    accn=f.accn,
                    ticker=f.ticker,
                    form=f.form,
                    filing_date=f.filing_date or "",
                    report_date=f.report_date or "",
                    page_count=f.page_count or 0,
                    is_indexed_visual=bool(f.is_indexed_visual),
                    is_indexed_text=bool(f.is_indexed_text),
                    is_amendment=bool(f.is_amendment),
                )
                for f in filings
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    finally:
        db.close()


@app.get("/filings/{accn}")
async def get_filing(
    accn: str,
    _key: str = Depends(verify_api_key),
) -> FilingMeta:
    db = SessionLocal()
    try:
        filing = db.query(Filing).filter(Filing.accn == accn).first()
        if not filing:
            raise HTTPException(status_code=404, detail="Filing not found")
        return FilingMeta(
            accn=filing.accn,
            ticker=filing.ticker,
            form=filing.form,
            filing_date=filing.filing_date or "",
            report_date=filing.report_date or "",
            page_count=filing.page_count or 0,
            is_indexed_visual=bool(filing.is_indexed_visual),
            is_indexed_text=bool(filing.is_indexed_text),
            is_amendment=bool(filing.is_amendment),
        )
    finally:
        db.close()


@app.get("/pages/{accn}/{page_idx}")
async def get_page(
    accn: str,
    page_idx: int,
    _key: str = Depends(verify_api_key),
) -> FileResponse:
    accn_nodashes = accn.replace("-", "")
    png_path = Path(settings.pages_dir) / accn_nodashes / f"page_{page_idx:04d}.png"
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(str(png_path), media_type="image/png")


@app.get("/health")
async def health() -> HealthResponse:
    import sqlalchemy

    db_ok = False
    try:
        db = SessionLocal()
        db.execute(sqlalchemy.text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        details={"version": "0.1.0"},
    )
