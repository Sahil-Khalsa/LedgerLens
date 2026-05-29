"""
LangGraph node implementations.

Phase 1: planner and verifier stubs sufficient for the text baseline pipeline.
Phase 2: visual_retriever, extractor wired to ColPali + VLM.
Phase 3: full synthesizer and critic with reflection loop.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from agents.state import Critique, Fact, PageResult, State
from api.validation import wrap_user_content
from config import settings

logger = logging.getLogger(__name__)


# ── Langfuse tracing (no-op when not configured) ──────────────────────────────


def _get_langfuse() -> Any:
    """Return a Langfuse client if configured, else None."""
    if not settings.langfuse_public_key:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None


@contextmanager
def _span(name: str, input_data: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager that creates a Langfuse span when configured, else is a no-op."""
    lf = _get_langfuse()
    start = time.monotonic()
    span = None
    try:
        if lf is not None:
            trace = lf.trace(name=name, input=input_data or {})
            span = trace.span(name=name)
        yield span
    finally:
        elapsed = int((time.monotonic() - start) * 1000)
        if span is not None:
            try:
                span.end(metadata={"latency_ms": elapsed})
                lf.flush()  # type: ignore[union-attr]
            except Exception:
                pass


# ── Planner ───────────────────────────────────────────────────────────────────


def planner(state: State) -> dict[str, Any]:
    """
    Decompose the question and decide routing:
    - fast: single XBRL fact lookup, skip retrieval and VLM entirely
    - document: needs page retrieval and extraction
    """
    with _span("planner", {"question": state["question"]}):
        return _planner(state)


def _planner(state: State) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    prompt = f"""You are a financial query planner.

Given the question below, decide:
1. route: "fast" if it asks for a single reported figure directly answerable
   from structured XBRL data (e.g. "What was revenue in FY2024?").
   "document" if it requires reading filing text, tables, or multi-step reasoning.
2. queries: list of 1-3 retrieval sub-queries (for document route).
3. xbrl_concept: best-guess XBRL concept name (for fast route, e.g. "Revenues").
4. period_end: fiscal period end date YYYY-MM-DD (for fast route).
5. period_start: fiscal period start date YYYY-MM-DD (for flow concepts, null for stock).

{wrap_user_content(state["question"])}

Respond with JSON only:
{{"route": "fast"|"document", "queries": [...], "xbrl_concept": null|"...",
  "period_end": null|"YYYY-MM-DD", "period_start": null|"YYYY-MM-DD"}}"""

    resp = client.chat.completions.create(
        model=settings.planner_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=256,
    )

    import json

    parsed = json.loads(resp.choices[0].message.content or "{}")
    route = parsed.get("route", "document")

    # Attempt fast path XBRL lookup
    if route == "fast" and parsed.get("xbrl_concept") and parsed.get("period_end"):
        fact = _lookup_xbrl_fact(
            ticker=state.get("ticker") or "",
            concept=parsed["xbrl_concept"],
            period_end=parsed["period_end"],
            period_start=parsed.get("period_start"),
        )
        if fact:
            return {
                "route": "fast",
                "plan": [],
                "facts": [fact],
            }

    # Fall through to document route
    return {
        "route": "document",
        "plan": parsed.get("queries", [state["question"]]),
    }


def _lookup_xbrl_fact(
    ticker: str,
    concept: str,
    period_end: str,
    period_start: str | None,
) -> Fact | None:
    """Query the xbrl_facts table for a single fact."""
    from index.store import SessionLocal, XbrlFact
    from ingest.xbrl import match_period

    db = SessionLocal()
    try:
        rows = (
            db.query(XbrlFact)
            .filter(XbrlFact.concept == concept, XbrlFact.end_date == period_end)
            .all()
        )
        for row in rows:
            f: dict[str, object] = {
                "end": row.end_date,
                "start": row.start_date,
                "is_flow": row.is_flow,
            }
            if match_period(f, period_end, period_start):
                return Fact(
                    text=f"{concept} = {row.value} {row.unit}",
                    value=row.value,
                    concept=concept,
                    page_ref=f"{row.accn}:xbrl",
                    verified="match",
                )
    finally:
        db.close()
    return None


# ── Numerical verifier ────────────────────────────────────────────────────────


def numerical_verifier(state: State) -> dict[str, Any]:
    with _span("numerical_verifier", {"facts_count": len(state["facts"])}):
        return _numerical_verifier(state)


def _numerical_verifier(state: State) -> dict[str, Any]:
    """
    For each extracted Fact with a numeric value, compare against XBRL ground truth.
    Sets verified = match | mismatch | unverifiable.
    Fast-path facts arrive pre-verified as "match".

    Amendment preference: when multiple rows exist for the same concept, rows from
    amendment filings (10-K/A, 10-Q/A) are checked first — they supersede the original.
    """
    from index.store import Filing, SessionLocal, XbrlFact
    from ingest.xbrl import compare_with_tolerance

    db = SessionLocal()
    verified_facts: list[Fact] = []

    try:
        for fact in state["facts"]:
            if fact["verified"] == "match":
                verified_facts.append(fact)
                continue
            if fact["value"] is None or fact["concept"] is None:
                verified_facts.append({**fact, "verified": "unverifiable"})
                continue

            # Prefer amendment facts: join to filings and sort amendments first
            rows = (
                db.query(XbrlFact)
                .join(Filing, Filing.accn == XbrlFact.accn)
                .filter(XbrlFact.concept == fact["concept"])
                .order_by(Filing.is_amendment.desc())
                .all()
            )

            matched = False
            for row in rows:
                if compare_with_tolerance(fact["value"], row.value):
                    verified_facts.append({**fact, "verified": "match"})
                    matched = True
                    break

            if not matched:
                if rows:
                    verified_facts.append({**fact, "verified": "mismatch"})
                    logger.warning("Mismatch: %s extracted %.2f", fact["concept"], fact["value"])
                else:
                    verified_facts.append({**fact, "verified": "unverifiable"})
    finally:
        db.close()

    return {"facts": verified_facts}


# ── Synthesizer ───────────────────────────────────────────────────────────────


def synthesizer(state: State) -> dict[str, Any]:
    with _span(
        "synthesizer",
        {"verified_facts": sum(1 for f in state["facts"] if f["verified"] == "match")},
    ):
        return _synthesizer(state)


def _synthesizer(state: State) -> dict[str, Any]:
    from openai import OpenAI

    verified = [f for f in state["facts"] if f["verified"] == "match"]
    unverifiable = [f for f in state["facts"] if f["verified"] == "unverifiable"]

    if not verified and not unverifiable:
        return {
            "answer": None,
            "answer_status": "insufficient_data",
            "draft": None,
        }

    facts_text = "\n".join(
        f"- {f['text']} [cite: {f['page_ref']}] [status: {f['verified']}]"
        for f in verified + unverifiable
    )

    prompt = f"""You are a financial analyst writing a precise, cited answer.

Use ONLY the verified facts below. Do not invent or estimate figures.
For unverifiable facts, flag them explicitly as unconfirmed.
Every numeric claim must include its citation in brackets.

Verified facts:
{facts_text}

{wrap_user_content(state["question"])}

Answer:"""

    resp = OpenAI().chat.completions.create(
        model=settings.synthesizer_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    draft = resp.choices[0].message.content or ""
    status = "answered" if verified else "low_confidence"

    return {"draft": draft, "answer": draft, "answer_status": status}


# ── Critic ────────────────────────────────────────────────────────────────────


def critic(state: State) -> dict[str, Any]:
    with _span("critic", {"retries": state.get("retries", 0)}):
        return _critic(state)


def _critic(state: State) -> dict[str, Any]:
    import json

    from openai import OpenAI

    if not state.get("draft"):
        return {
            "critique": Critique(
                grounded=False,
                ungrounded_claims=[],
                missing_evidence=[state["question"]],
            ),
            "retries": state.get("retries", 0) + 1,
        }

    pages_summary = "; ".join(f"{p['accn']}:p{p['page_idx']}" for p in state.get("pages", [])[:10])

    prompt = f"""You are a faithfulness critic for a financial Q&A system.

Draft answer:
{state["draft"]}

Retrieved pages: {pages_summary or "none"}

For each numeric or factual claim in the draft, check if it is supported by the
retrieved pages or a verified XBRL fact (page_ref ending in :xbrl).

Return JSON only:
{{"grounded": true|false,
  "ungrounded_claims": ["claim text..."],
  "missing_evidence": ["what to search for to ground each ungrounded claim"]}}"""

    resp = OpenAI().chat.completions.create(
        model=settings.planner_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    grounded: bool = bool(parsed.get("grounded", False))
    retries = state.get("retries", 0)

    critique = Critique(
        grounded=grounded,
        ungrounded_claims=parsed.get("ungrounded_claims", []),
        missing_evidence=parsed.get("missing_evidence", []),
    )

    new_retries = retries if grounded else retries + 1
    new_plan = parsed.get("missing_evidence", []) if not grounded else state.get("plan", [])

    return {
        "critique": critique,
        "retries": new_retries,
        "plan": new_plan,
    }


# ── Visual retriever ─────────────────────────────────────────────────────────


def visual_retriever(state: State) -> dict[str, Any]:
    with _span("visual_retriever", {"queries": len(state.get("plan") or [state["question"]])}):
        return _visual_retriever(state)


def _visual_retriever(state: State) -> dict[str, Any]:
    from index.visual import retrieve as visual_retrieve

    queries = state.get("plan") or [state["question"]]
    accn = state.get("filing_accn")

    raw: list[PageResult] = []
    for q in queries:
        pages = visual_retrieve(q, accn=accn, top_k=settings.retrieval_top_k)
        for p in pages:
            raw.append(
                PageResult(
                    accn=str(p["accn"]),
                    page_idx=int(p["page_idx"]),  # type: ignore[arg-type]
                    png_path=str(p["png_path"]),
                    score=float(p["score"]),  # type: ignore[arg-type]
                    continued_on_page=None,
                )
            )

    # Dedup by (accn, page_idx), keep highest score
    best: dict[tuple[str, int], PageResult] = {}
    for p in raw:
        key = (p["accn"], p["page_idx"])
        if key not in best or p["score"] > best[key]["score"]:
            best[key] = p

    return {"pages": list(best.values())}


# ── Extractor ─────────────────────────────────────────────────────────────────


def extractor(state: State) -> dict[str, Any]:
    with _span("extractor", {"pages": len(state.get("pages", []))}):
        return _extractor(state)


def _extractor(state: State) -> dict[str, Any]:
    import base64
    import json
    from pathlib import Path

    from openai import OpenAI

    pages = state.get("pages", [])
    if not pages:
        return {"facts": []}

    top_pages = sorted(pages, key=lambda p: p["score"], reverse=True)[: settings.retrieval_top_k]

    client = OpenAI()
    facts: list[Fact] = []

    for page in top_pages:
        png_path = page["png_path"]
        if not Path(png_path).exists():
            logger.warning("PNG not found: %s", png_path)
            continue

        img_bytes = Path(png_path).read_bytes()
        b64 = base64.standard_b64encode(img_bytes).decode()

        prompt = (
            "Extract every numeric financial fact visible on this filing page.\n"
            "Return a JSON object with key 'facts', value an array where each item has:\n"
            '  "text": the full claim as a sentence,\n'
            '  "value": the numeric value as a raw number (null if non-numeric),\n'
            '  "concept": best-guess XBRL concept name (e.g. "Revenues"), null if unknown.\n\n'
            f"{wrap_user_content(state['question'])}"
        )

        resp = client.chat.completions.create(
            model=settings.synthesizer_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
        )

        try:
            parsed = json.loads(resp.choices[0].message.content or "{}")
            items: list[dict[str, object]] = []
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                raw = parsed.get("facts", [])
                items = raw if isinstance(raw, list) else []

            for item in items:
                raw_val = item.get("value")
                facts.append(
                    Fact(
                        text=str(item.get("text", "")),
                        value=float(raw_val) if raw_val is not None else None,  # type: ignore[arg-type]
                        concept=str(item["concept"]) if item.get("concept") else None,
                        page_ref=f"{page['accn']}:{page['page_idx']}",
                        verified="pending",
                    )
                )
        except Exception as exc:
            logger.warning(
                "Extractor parse error for %s page %d: %s",
                page["accn"],
                page["page_idx"],
                exc,
            )

    logger.info("Extracted %d facts from %d pages", len(facts), len(top_pages))
    return {"facts": facts}
