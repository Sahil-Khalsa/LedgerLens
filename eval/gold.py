"""
Gold set management: schema validation, auto-generation from XBRL, and loading.

gold_set.jsonl is versioned and committed. Never edit it to make a failing run pass.
Each entry is validated against GoldItem on load — a malformed entry raises immediately.
"""

import logging
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

GOLD_SET_PATH = Path(__file__).parent / "gold_set.jsonl"


class GoldItem(BaseModel):
    q: str
    kind: Literal["numeric", "reasoning", "negative", "adversarial"]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    subset: list[str] = []
    expected_status: Literal["answered", "insufficient_data"] = "answered"

    # Numeric items
    answer_value: float | None = None
    unit: str | None = None
    concept: str | None = None
    accn: str | None = None

    # Reasoning items
    rubric: str | None = None

    @field_validator("q")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "GoldItem":
        if self.kind == "numeric" and self.answer_value is None:
            raise ValueError("numeric items require answer_value")
        if self.kind == "reasoning" and not self.rubric:
            raise ValueError("reasoning items require rubric")
        if self.kind in ("negative", "adversarial") and self.expected_status != "insufficient_data":
            raise ValueError(
                "negative/adversarial items must have expected_status=insufficient_data"
            )
        return self


def load_gold_set(
    path: Path = GOLD_SET_PATH,
    subset: str | None = None,
) -> list[GoldItem]:
    """
    Load and validate the gold set. Raises ValueError on the first malformed entry.
    Optionally filter by subset tag (e.g. "ci").
    """
    if not path.exists():
        logger.warning("gold_set.jsonl not found at %s — returning empty set", path)
        return []

    items: list[GoldItem] = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = GoldItem.model_validate_json(line)
            except Exception as e:
                raise ValueError(f"gold_set.jsonl line {i} is invalid: {e}") from e
            if subset is None or subset in item.subset:
                items.append(item)

    logger.info("Loaded %d gold items (subset=%s)", len(items), subset or "all")
    return items


def auto_generate(
    facts: list[dict[str, object]],
    ticker: str,
    max_per_concept: int = 3,
) -> Iterator[GoldItem]:
    """
    Generate numeric GoldItems from normalized XBRL facts.
    Deduplicates to at most max_per_concept items per concept (newest periods first).
    """
    by_concept: dict[str, list[dict[str, object]]] = defaultdict(list)
    for f in facts:
        if f.get("value") is not None and f.get("fp") and f.get("accn"):
            by_concept[str(f["concept"])].append(f)

    for concept, group in by_concept.items():
        # Sort newest first by end date
        group.sort(key=lambda x: str(x.get("end", "")), reverse=True)
        for f in group[:max_per_concept]:
            period = f"{f.get('fp', '')} {f.get('fy', '')}".strip()
            q = f"What was {ticker}'s {_humanize(concept)} for {period}?"
            yield GoldItem(
                q=q,
                kind="numeric",
                difficulty="easy",
                answer_value=float(f["value"]),  # type: ignore[arg-type]
                unit=str(f.get("unit", "USD")),
                concept=concept,
                accn=str(f["accn"]),
            )


def _humanize(concept_name: str) -> str:
    """Convert CamelCase XBRL concept name to a human-readable label."""
    import re
    # Insert spaces before capital letters
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", concept_name)
    return spaced.lower()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gold set management")
    parser.add_argument("--ticker", help="Generate auto questions for a ticker")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate gold_set.jsonl")
    parser.add_argument("--subset", help="Filter by subset tag")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.validate:
        items = load_gold_set(subset=args.subset)
        print(f"✓ {len(items)} valid gold items")
        return

    if args.auto and args.ticker:
        from ingest.edgar import resolve_cik
        from ingest.xbrl import stream_normalize_facts
        cik10 = resolve_cik(args.ticker)
        facts = list(stream_normalize_facts(cik10))
        for item in auto_generate(facts, args.ticker):
            print(item.model_dump_json())


if __name__ == "__main__":
    _cli()
