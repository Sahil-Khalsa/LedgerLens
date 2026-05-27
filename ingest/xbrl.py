"""
XBRL companyfacts ingestion and normalization.

Key design decisions (all from SPEC.md §1):
- Stream-parse large companyfacts JSON with ijson to avoid OOM.
- Skip val=null (restatement markers) and facts with no accn.
- Tag each fact as flow (has start date) vs stock (point-in-time).
- Detect and store the display scale from the filing HTML header.
- normalize_display_value() converts displayed figures to raw XBRL units for comparison.
"""

import io
import logging
import re
from collections.abc import Iterator

import ijson  # type: ignore[import-untyped]

from ingest.edgar import get_stream, resolve_cik

logger = logging.getLogger(__name__)

# ── Scale detection ───────────────────────────────────────────────────────────

_SCALE_PATTERNS: list[tuple[str, int]] = [
    (r"in\s+billions", 1_000_000_000),
    (r"in\s+millions", 1_000_000),
    (r"in\s+thousands", 1_000),
]


def detect_filing_scale(html_text: str) -> int:
    """
    Read the scale declaration from the filing header text.
    Returns the multiplier needed to convert displayed values to raw XBRL units.
    Returns 1 if no scale declaration is found (raw dollars).
    """
    lower = html_text.lower()
    for pattern, scale in _SCALE_PATTERNS:
        if re.search(pattern, lower):
            return scale
    return 1


def normalize_display_value(text_value: str, filing_scale: int) -> float | None:
    """
    Convert a value as displayed in the filing (e.g. "47,532" in millions)
    to raw units matching XBRL storage (e.g. 47532000000).

    Handles:
    - Comma separators
    - Dollar signs
    - Parentheses for negatives (accounting convention)
    - Decimal points
    """
    is_negative = "(" in text_value and ")" in text_value
    cleaned = re.sub(r"[,$\s()]", "", text_value)
    if not cleaned:
        return None
    try:
        v = float(cleaned)
    except ValueError:
        return None
    if is_negative:
        v = -v
    return v * filing_scale


def compare_with_tolerance(extracted: float, xbrl: float, tol: float = 0.01) -> bool:
    """
    Return True if extracted value matches XBRL ground truth within relative tolerance.
    Default tol=0.01 allows 1% rounding difference (e.g. displayed in millions vs raw).
    """
    if xbrl == 0:
        return abs(extracted) < 1  # treat near-zero as match
    return abs(extracted - xbrl) / abs(xbrl) <= tol


# ── XBRL fact normalization ───────────────────────────────────────────────────

def normalize_facts(companyfacts: dict[str, object]) -> list[dict[str, object]]:
    """
    Convert companyfacts JSON (already parsed into a dict) into a flat list of fact rows.
    Use stream_normalize_facts() for large companies to avoid loading the full JSON.
    """
    rows: list[dict[str, object]] = []
    facts_block = companyfacts.get("facts", {})
    assert isinstance(facts_block, dict)

    for taxonomy in ("us-gaap", "dei"):
        concepts = facts_block.get(taxonomy, {})
        assert isinstance(concepts, dict)
        for concept, body in concepts.items():
            assert isinstance(body, dict)
            for unit, fact_list in body.get("units", {}).items():
                assert isinstance(fact_list, list)
                for f in fact_list:
                    row = _make_row(concept, unit, f)
                    if row is not None:
                        rows.append(row)
    return rows


def stream_normalize_facts(cik10: str) -> Iterator[dict[str, object]]:
    """
    Stream-parse companyfacts for a company without loading the full JSON into memory.
    Safe for large companies (AAPL, MSFT) where the JSON can exceed 200 MB.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    raw_bytes = b"".join(get_stream(url))
    stream = io.BytesIO(raw_bytes)

    for taxonomy in ("us-gaap", "dei"):
        prefix = f"facts.{taxonomy}"
        try:
            for concept, body in ijson.kvitems(stream, prefix):
                stream.seek(0)  # ijson needs a fresh read per prefix call
                if not isinstance(body, dict):
                    continue
                for unit, fact_list in body.get("units", {}).items():
                    if not isinstance(fact_list, list):
                        continue
                    for f in fact_list:
                        row = _make_row(str(concept), str(unit), f)
                        if row is not None:
                            yield row
        except Exception:
            logger.warning("ijson parse error for taxonomy=%s cik=%s", taxonomy, cik10)
            stream.seek(0)


def _make_row(
    concept: str, unit: str, f: dict[str, object]
) -> dict[str, object] | None:
    """Build a normalized fact row, returning None for restatement/incomplete entries."""
    val = f.get("val")
    accn = f.get("accn")

    if val is None:  # restatement marker — skip
        return None
    if accn is None:  # unattributed fact — skip
        return None

    return {
        "concept": concept,
        "unit": unit,
        "value": float(val),  # type: ignore[arg-type]
        "start": f.get("start"),
        "end": f.get("end"),
        "fy": f.get("fy"),
        "fp": f.get("fp"),
        "form": f.get("form"),
        "accn": str(accn),
        "is_flow": "start" in f,  # flow=income stmt/cashflow; stock=balance sheet
    }


# ── Period matching ───────────────────────────────────────────────────────────

def match_period(
    fact: dict[str, object],
    query_end: str,
    query_start: str | None = None,
) -> bool:
    """
    Match a fact to the period implied by a query.
    Flow facts (is_flow=True) require both start and end to match.
    Stock facts require only end to match.
    """
    if fact.get("is_flow"):
        return fact.get("start") == query_start and fact.get("end") == query_end
    return fact.get("end") == query_end


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Show XBRL facts for a filing")
    parser.add_argument("--accn", required=True, help="Accession number")
    parser.add_argument("--ticker", help="Ticker (used to resolve CIK if accn lookup needed)")
    parser.add_argument("--concept", default=None, help="Filter by concept name")
    args = parser.parse_args()

    if args.ticker:
        cik10 = resolve_cik(args.ticker)
        facts = list(stream_normalize_facts(cik10))
    else:
        print("Provide --ticker to look up XBRL facts")
        return

    if args.concept:
        facts = [f for f in facts if args.concept.lower() in str(f["concept"]).lower()]

    for f in facts[:50]:
        print(json.dumps(f, default=str))
    print(f"\n({len(facts)} total facts)")


if __name__ == "__main__":
    _cli()
