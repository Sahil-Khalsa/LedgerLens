"""
CI eval gate — exits 1 if metrics fall below thresholds.

Usage:
  uv run python -m eval.gate results.json
  uv run python -m eval.gate results.json --min-numeric-em 0.85 --max-hallucination 0.02

Do NOT lower thresholds to go green — fix the root cause.
"""

import json
import logging
import sys
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def check_gate(
    report_path: str,
    min_numeric_em: float | None = None,
    max_hallucination: float | None = None,
) -> bool:
    em_threshold = min_numeric_em if min_numeric_em is not None else settings.eval_min_numeric_em
    hall_threshold = (
        max_hallucination if max_hallucination is not None else settings.eval_max_hallucination
    )

    data = json.loads(Path(report_path).read_text())
    metrics = data.get("metrics", {})

    numeric_em: float | None = metrics.get("numeric_em")
    hallucination: float = float(metrics.get("hallucination_rate", 1.0))

    passed = True
    failures: list[str] = []

    if numeric_em is not None and numeric_em < em_threshold:
        failures.append(f"Numeric EM {numeric_em:.1%} < threshold {em_threshold:.1%}")
        passed = False

    if hallucination > hall_threshold:
        failures.append(f"Hallucination {hallucination:.1%} > threshold {hall_threshold:.1%}")
        passed = False

    if passed:
        print("✓ Eval gate passed")
        if numeric_em is not None:
            print(f"  Numeric EM:    {numeric_em:.1%}")
        print(f"  Hallucination: {hallucination:.1%}")
    else:
        print("✗ Eval gate FAILED")
        for msg in failures:
            print(f"  → {msg}")
        print("\nFix the root cause. Do NOT lower the thresholds to go green.")

    return passed


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--min-numeric-em", type=float, default=None)
    parser.add_argument("--max-hallucination", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if check_gate(args.report, args.min_numeric_em, args.max_hallucination) else 1)


if __name__ == "__main__":
    _cli()
