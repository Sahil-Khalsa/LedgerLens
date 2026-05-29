"""
End-to-end ingest pipeline for one ticker.

Chains: EDGAR filing discovery → HTML download → page render → XBRL normalization
        → DB storage → text baseline indexing.

Designed for Phase 1 scope (NVDA + MSFT, 3-4 filings each).
Each step is idempotent — safe to re-run after a partial failure.

Usage:
  uv run python -m ingest.pipeline --ticker NVDA --forms 10-K 10-Q --limit 4
  uv run python -m ingest.pipeline --ticker NVDA --skip-render   # index only
"""

import logging
from collections import defaultdict

from index.store import upsert_filing, upsert_pages, upsert_xbrl_facts
from index.text_baseline import index_filing
from ingest.edgar import resolve_cik
from ingest.filings import ingest_filings
from ingest.render import render_filing
from ingest.xbrl import stream_normalize_facts

logger = logging.getLogger(__name__)


def run_pipeline(
    ticker: str,
    forms: list[str] | None = None,
    limit: int = 4,
    skip_render: bool = False,
    skip_xbrl: bool = False,
    skip_index: bool = False,
) -> list[str]:
    """
    Full ingest pipeline for one ticker. Returns list of accession numbers processed.

    Steps:
    1. Discover + download primary HTML for each filing.
    2. Upsert filing metadata to DB.
    3. Render HTML → per-page PNGs, upsert pages.
    4. Stream XBRL companyfacts; store facts for the downloaded accns.
    5. Build text baseline chunk index.
    """
    # ── Step 1 & 2: Download + store filing metadata ──────────────────────────
    logger.info("Ingesting filings for %s (forms=%s, limit=%d)", ticker, forms, limit)
    filings = ingest_filings(ticker, forms=forms, limit=limit)

    if not filings:
        logger.warning("No filings downloaded for %s — aborting", ticker)
        return []

    for meta in filings:
        upsert_filing(meta)
        logger.info("Upserted filing %s (%s)", meta["accn"], meta["form"])

    accns = [str(f["accn"]) for f in filings]

    # ── Step 3: Render pages ──────────────────────────────────────────────────
    if skip_render:
        logger.info("Skipping page render (--skip-render)")
    else:
        for meta in filings:
            accn = str(meta["accn"])
            html_path = str(meta["html_path"])
            logger.info("Rendering pages for %s", accn)
            pages = render_filing(html_path, accn)
            if pages:
                upsert_pages(accn, pages)
                logger.info("Stored %d pages for %s", len(pages), accn)
            else:
                logger.warning("No pages rendered for %s", accn)

    # ── Step 4: XBRL facts ────────────────────────────────────────────────────
    if skip_xbrl:
        logger.info("Skipping XBRL fetch (--skip-xbrl)")
    else:
        cik10 = resolve_cik(ticker)
        target_accns = set(accns)

        # Stream the full companyfacts JSON once; filter to the accns we care about.
        # Facts are grouped by accn so upsert_xbrl_facts can do a clean replace.
        facts_by_accn: dict[str, list[dict[str, object]]] = defaultdict(list)
        logger.info("Streaming XBRL facts for CIK %s ...", cik10)

        for fact in stream_normalize_facts(cik10):
            fact_accn = str(fact.get("accn", ""))
            if fact_accn in target_accns:
                facts_by_accn[fact_accn].append(fact)

        for accn in accns:
            facts = facts_by_accn.get(accn, [])
            if facts:
                n = upsert_xbrl_facts(cik10, accn, facts)
                logger.info("Stored %d XBRL facts for %s", n, accn)
            else:
                logger.warning("No XBRL facts found for accn=%s", accn)

    # ── Step 5: Text baseline index ───────────────────────────────────────────
    if skip_index:
        logger.info("Skipping text index (--skip-index)")
    else:
        for meta in filings:
            accn = str(meta["accn"])
            html_path = str(meta["html_path"])
            logger.info("Building text index for %s", accn)
            n = index_filing(accn, html_path)
            logger.info("Indexed %d chunks for %s", n, accn)

    logger.info("Pipeline complete for %s: %d filings", ticker, len(filings))
    return accns


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full ingest pipeline for a ticker")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. NVDA)")
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--skip-render", action="store_true", help="Skip HTML→PNG rendering")
    parser.add_argument("--skip-xbrl", action="store_true", help="Skip XBRL fact ingestion")
    parser.add_argument("--skip-index", action="store_true", help="Skip text baseline indexing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    accns = run_pipeline(
        ticker=args.ticker,
        forms=args.forms,
        limit=args.limit,
        skip_render=args.skip_render,
        skip_xbrl=args.skip_xbrl,
        skip_index=args.skip_index,
    )

    print(f"\nProcessed {len(accns)} filings:")
    for a in accns:
        print(f"  {a}")


if __name__ == "__main__":
    _cli()
