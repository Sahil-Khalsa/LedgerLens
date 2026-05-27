"""
Eval runner — runs the pipeline against the gold set and writes results.

Usage:
  uv run python -m eval.run --pipeline baseline --subset ci --report results.json
  uv run python -m eval.run --pipeline visual  --report results_visual.json

Always run BOTH pipelines and compare the delta. The delta is the headline number.
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from eval.gold import GoldItem, load_gold_set
from eval.metrics import MetricResult, QueryResult, compute_all

logger = logging.getLogger(__name__)

Pipeline = Literal["baseline", "visual", "auto"]


def run_baseline_query(gold: GoldItem) -> QueryResult:
    """
    Run one question through the text-only baseline pipeline.
    Retrieves from the chunks table, generates an answer with an LLM.
    """
    from openai import OpenAI

    from index.text_baseline import retrieve

    start = time.monotonic()
    client = OpenAI()

    chunks = retrieve(gold.q, accn=gold.accn, top_k=5)
    context = "\n\n".join(c["content"] for c in chunks)
    refs = [f"{c['accn']}:text" for c in chunks]

    if not context.strip():
        return QueryResult(
            question=gold.q,
            predicted_answer=None,
            answer_status="insufficient_data",
            predicted_value=None,
            retrieved_page_refs=refs,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    prompt = (
        "You are a financial analyst. Answer the question based only on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"<user_question>{gold.q}</user_question>\n\n"
        "Answer with the exact figure if it is present. If not present, say 'not found'."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    answer = resp.choices[0].message.content or ""
    usage = resp.usage
    cost = 0.0
    if usage:
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

    latency = int((time.monotonic() - start) * 1000)
    status = "insufficient_data" if "not found" in answer.lower() else "answered"

    return QueryResult(
        question=gold.q,
        predicted_answer=answer,
        answer_status=status,
        predicted_value=None,
        retrieved_page_refs=refs,
        cost_usd=cost,
        latency_ms=latency,
    )


def run_visual_query(gold: GoldItem) -> QueryResult:
    """
    Placeholder — wired up in Phase 3 when the visual index and agent graph are built.
    """
    return QueryResult(
        question=gold.q,
        predicted_answer=None,
        answer_status="insufficient_data",
        predicted_value=None,
    )


def run_eval(
    pipeline: Pipeline,
    subset: str | None = None,
    report_path: str = "results.json",
) -> MetricResult:
    """
    Run the full eval loop and write a JSON + markdown report.
    """
    gold_items = load_gold_set(subset=subset)
    if not gold_items:
        logger.warning("No gold items loaded — check gold_set.jsonl and subset filter")

    runner = run_baseline_query if pipeline == "baseline" else run_visual_query

    pairs: list[tuple[GoldItem, QueryResult]] = []
    for i, gold in enumerate(gold_items):
        logger.info("[%d/%d] %s", i + 1, len(gold_items), gold.q[:80])
        try:
            result = runner(gold)
        except Exception as exc:
            logger.error("Query failed: %s", exc)
            result = QueryResult(
                question=gold.q,
                predicted_answer=None,
                answer_status="error",
                predicted_value=None,
            )
        pairs.append((gold, result))

    metrics = compute_all(pairs)

    # Write JSON report
    report: dict[str, object] = {
        "pipeline": pipeline,
        "subset": subset,
        "metrics": asdict(metrics),
        "results": [
            {
                "q": g.q,
                "kind": g.kind,
                "gold_value": g.answer_value,
                "predicted": r.predicted_answer,
                "status": r.answer_status,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
            }
            for g, r in pairs
        ],
    }
    Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report written to %s", report_path)

    # Write markdown summary
    _write_markdown_report(metrics, pipeline, subset, pairs, report_path)

    return metrics


def _write_markdown_report(
    m: MetricResult,
    pipeline: str,
    subset: str | None,
    pairs: list[tuple[GoldItem, QueryResult]],
    report_path: str,
) -> None:
    md_path = report_path.replace(".json", ".md")
    lines = [
        f"# Eval Report — {pipeline} pipeline",
        f"Subset: `{subset or 'all'}`  |  Questions: {m.total_questions}",
        "",
        "## Metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Numeric exact-match | {m.numeric_em:.1%}" if m.numeric_em is not None else "| Numeric exact-match | N/A |",
        f"| Retrieval recall@k | {m.recall_at_k:.1%}" if m.recall_at_k is not None else "| Retrieval recall@k | N/A |",
        f"| Hallucination rate | {m.hallucination_rate:.1%} |",
        f"| Negative accuracy | {m.negative_accuracy:.1%}" if m.negative_accuracy is not None else "| Negative accuracy | N/A |",
        f"| Avg cost / query | ${m.avg_cost_usd:.4f} |",
        f"| Avg latency | {m.avg_latency_ms:.0f} ms |",
        "",
        "## Worst failures",
    ]
    for f in m.failures[:10]:
        lines.append(f"- **{f.get('q', '')}**")
        lines.append(f"  Gold: `{f.get('gold')}` | Predicted: `{f.get('predicted', 'N/A')}` | Reason: {f.get('reason', '')}")
        lines.append("")

    Path(md_path).write_text("\n".join(lines))
    logger.info("Markdown report written to %s", md_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the eval harness")
    parser.add_argument("--pipeline", choices=["baseline", "visual"], default="baseline")
    parser.add_argument("--subset", default=None, help="Filter gold set by subset tag")
    parser.add_argument("--report", default="results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    metrics = run_eval(args.pipeline, subset=args.subset, report_path=args.report)

    print(f"\n{'='*50}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Numeric exact-match: {metrics.numeric_em:.1%}" if metrics.numeric_em is not None else "Numeric EM: N/A")
    print(f"Hallucination rate:  {metrics.hallucination_rate:.1%}")
    if metrics.negative_accuracy is not None:
        print(f"Negative accuracy:   {metrics.negative_accuracy:.1%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    _cli()
