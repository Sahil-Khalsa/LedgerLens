"""
LangGraph node implementations.

Phase 1: planner and verifier stubs sufficient for the text baseline pipeline.
Phase 2: visual_retriever, extractor wired to ColPali + VLM.
Phase 3: full synthesizer and critic with reflection loop.
"""

import logging
from typing import Any

from agents.state import Critique, Fact, State
from api.validation import wrap_user_content
from config import settings

logger = logging.getLogger(__name__)


# ── Planner ───────────────────────────────────────────────────────────────────


def planner(state: State) -> dict[str, Any]:
    """
    Decompose the question and decide routing:
    - fast: single XBRL fact lookup, skip retrieval and VLM entirely
    - document: needs page retrieval and extraction
    """
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
    """
    For each extracted Fact with a numeric value, compare against XBRL ground truth.
    Sets verified = match | mismatch | unverifiable.
    Fast-path facts arrive pre-verified as "match".
    """
    from index.store import SessionLocal, XbrlFact
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

            rows = db.query(XbrlFact).filter(XbrlFact.concept == fact["concept"]).all()
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
    """
    Compose the final answer from verified facts only.
    Never emits a mismatch figure. Degrades gracefully when no verified facts exist.
    """
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
    """
    Faithfulness check: is every claim in the draft grounded in a retrieved page?
    If not and retries remain, returns missing_evidence as refined retrieval queries.
    """
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


# ── Visual retriever (Phase 2 stub) ──────────────────────────────────────────


def visual_retriever(state: State) -> dict[str, Any]:
    """
    Two-stage ColPali retrieval: pgvector cosine filter → MaxSim rerank.
    Stub in Phase 1 — wired to index/visual.py in Phase 2.
    """
    logger.warning("visual_retriever called but not yet implemented (Phase 2)")
    return {"pages": []}


# ── Extractor (Phase 2 stub) ──────────────────────────────────────────────────


def extractor(state: State) -> dict[str, Any]:
    """
    Send top-k reranked page images to the VLM and extract Fact objects.
    Stub in Phase 1 — wired in Phase 2.
    """
    logger.warning("extractor called but not yet implemented (Phase 2)")
    return {"facts": []}
