"""
Eval metrics — all deterministic except reasoning_correctness which uses an LLM judge.

Metrics:
  numeric_exact_match  — headline metric; gated in CI
  retrieval_recall_at_k
  hallucination_rate
  citation_faithfulness
  reasoning_correctness  (LLM-as-judge; NOT used in CI gate)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.gold import GoldItem

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Output of one pipeline run on one gold item."""
    question: str
    predicted_answer: str | None
    answer_status: str                        # "answered" | "low_confidence" | "insufficient_data"
    predicted_value: float | None             # parsed from the answer
    facts: list[dict[str, object]] = field(default_factory=list)
    retrieved_page_refs: list[str] = field(default_factory=list)   # ["accn:page_idx", ...]
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class MetricResult:
    numeric_em: float | None           # exact-match rate on numeric items
    recall_at_k: float | None          # retrieval recall on numeric items
    hallucination_rate: float          # fraction of answers with unsupported figures
    citation_faithfulness: float       # fraction of cited facts that are verifiably correct
    negative_accuracy: float | None    # fraction of negative/adversarial items correctly declined
    total_questions: int = 0
    numeric_questions: int = 0
    correct_numeric: int = 0
    avg_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    failures: list[dict[str, object]] = field(default_factory=list)


# ── Numeric exact-match ───────────────────────────────────────────────────────

def _parse_value_from_answer(answer: str) -> float | None:
    """Extract the first numeric value from a free-text answer string."""
    # Match numbers like 47,532 or 47.5 or $47.5B (with scale suffix)
    patterns = [
        r"\$?([\d,]+\.?\d*)\s*[Bb]illion",
        r"\$?([\d,]+\.?\d*)\s*[Mm]illion",
        r"\$?([\d,]+\.?\d*)\s*[Tt]rillion",
        r"\$?([\d,]+\.?\d*)",
    ]
    scale_map = {"billion": 1e9, "million": 1e6, "trillion": 1e12}

    for pattern in patterns:
        m = re.search(pattern, answer, re.IGNORECASE)
        if m:
            raw = float(m.group(1).replace(",", ""))
            for scale_word, multiplier in scale_map.items():
                if scale_word in pattern.lower() and scale_word.lower() in answer.lower():
                    return raw * multiplier
            return raw
    return None


def numeric_exact_match(
    results: list[tuple["GoldItem", QueryResult]],
    tol: float = 0.01,
) -> tuple[float, list[dict[str, object]]]:
    """
    Compute numeric exact-match rate.
    Returns (rate, failures) where failures lists items that did not match.
    """
    correct = 0
    total = 0
    failures: list[dict[str, object]] = []

    for gold, result in results:
        if gold.kind != "numeric":
            continue
        total += 1

        if result.answer_status == "insufficient_data":
            failures.append({"q": gold.q, "reason": "system declined", "gold": gold.answer_value})
            continue

        pred = result.predicted_value or _parse_value_from_answer(result.predicted_answer or "")
        if pred is None:
            failures.append({"q": gold.q, "reason": "no value extracted", "gold": gold.answer_value})
            continue

        assert gold.answer_value is not None
        gold_val = gold.answer_value
        match = abs(pred) < 1 if gold_val == 0 else abs(pred - gold_val) / abs(gold_val) <= tol

        if match:
            correct += 1
        else:
            failures.append({
                "q": gold.q,
                "predicted": pred,
                "gold": gold_val,
                "rel_error": abs(pred - gold_val) / max(abs(gold_val), 1),
            })

    rate = correct / total if total > 0 else 0.0
    return rate, failures


# ── Retrieval recall@k ────────────────────────────────────────────────────────

def retrieval_recall_at_k(
    results: list[tuple["GoldItem", QueryResult]],
) -> float:
    """
    Was the ground-truth filing's page in the retrieved set?
    Uses gold item's accn to determine the relevant filing.
    """
    hits = 0
    total = 0
    for gold, result in results:
        if gold.kind != "numeric" or not gold.accn:
            continue
        total += 1
        for ref in result.retrieved_page_refs:
            if ref.startswith(gold.accn):
                hits += 1
                break

    return hits / total if total > 0 else 0.0


# ── Hallucination rate ────────────────────────────────────────────────────────

def hallucination_rate(
    results: list[tuple["GoldItem", QueryResult]],
) -> float:
    """
    Fraction of answers that contain a numeric figure with no verified XBRL support
    and no cited page reference. Targets zero.
    """
    hallucinated = 0
    total = 0
    for _gold, result in results:
        if result.answer_status == "insufficient_data":
            continue
        total += 1
        for fact in result.facts:
            if (
                fact.get("verified") == "mismatch"
                or (fact.get("verified") == "unverifiable" and not fact.get("page_ref"))
            ):
                hallucinated += 1
                break

    return hallucinated / total if total > 0 else 0.0


# ── Negative / adversarial accuracy ──────────────────────────────────────────

def negative_accuracy(
    results: list[tuple["GoldItem", QueryResult]],
) -> float | None:
    """
    Fraction of negative/adversarial items correctly answered with insufficient_data.
    Returns None if no such items in the set.
    """
    correct = 0
    total = 0
    for gold, result in results:
        if gold.kind not in ("negative", "adversarial"):
            continue
        total += 1
        if result.answer_status == "insufficient_data":
            correct += 1

    return correct / total if total > 0 else None


# ── Aggregate ─────────────────────────────────────────────────────────────────

def compute_all(
    results: list[tuple["GoldItem", QueryResult]],
) -> MetricResult:
    em, failures = numeric_exact_match(results)
    recall = retrieval_recall_at_k(results)
    hall = hallucination_rate(results)
    neg_acc = negative_accuracy(results)

    costs = [r.cost_usd for _, r in results]
    latencies = [r.latency_ms for _, r in results]

    return MetricResult(
        numeric_em=em,
        recall_at_k=recall,
        hallucination_rate=hall,
        citation_faithfulness=1.0 - hall,
        negative_accuracy=neg_acc,
        total_questions=len(results),
        avg_cost_usd=sum(costs) / len(costs) if costs else 0.0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        failures=failures,
    )
