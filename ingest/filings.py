"""
Filing discovery, download, and metadata storage.

Resolves ticker → CIK → filing list → downloads primary HTML document.
All SEC HTTP traffic routes through ingest.edgar — never requests.get() directly.
"""

import logging
from pathlib import Path

from config import settings
from ingest.edgar import (
    build_doc_url,
    get,
    get_all_filing_rows,
    resolve_cik,
)
from ingest.xbrl import detect_filing_scale

logger = logging.getLogger(__name__)

SUPPORTED_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})


def is_amendment(form: str) -> bool:
    return form.endswith("/A")


def list_filings(
    ticker: str,
    forms: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """
    Return filing metadata rows for a ticker, filtered by form type.
    Rows are sorted newest-first by filingDate.

    Each row: {accn, form, filing_date, report_date, primary_doc, cik, ticker,
               is_amendment, amends_accn}
    """
    cik10 = resolve_cik(ticker)
    all_rows = get_all_filing_rows(cik10)

    allowed = {f.upper() for f in forms} if forms else {f.upper() for f in SUPPORTED_FORMS}
    filtered = [
        r for r in all_rows if str(r.get("form", "")).upper() in allowed
    ]
    filtered.sort(key=lambda r: str(r.get("filingDate", "")), reverse=True)

    if limit:
        filtered = filtered[:limit]

    result = []
    for r in filtered:
        accn = str(r["accessionNumber"])
        form = str(r["form"])
        result.append({
            "accn": accn,
            "cik": cik10,
            "ticker": ticker.upper(),
            "form": form,
            "filing_date": str(r.get("filingDate", "")),
            "report_date": str(r.get("reportDate", "")),
            "primary_doc": str(r.get("primaryDocument", "")),
            "is_amendment": is_amendment(form),
            "amends_accn": None,  # populated later if needed
        })
    return result


def download_filing_html(
    cik: str,
    accn: str,
    primary_doc: str,
    filing_date: str,
) -> tuple[str, int]:
    """
    Download the primary HTML document for a filing.
    Returns (local_path, filing_scale).

    Stores to: data/html/{accn_nodashes}/primary.html
    The filing_date is included for cache-key purposes (amendment invalidation).
    """
    accn_nodashes = accn.replace("-", "")
    out_dir = Path(settings.html_dir) / accn_nodashes
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "primary.html"

    url = build_doc_url(cik, accn, primary_doc)
    logger.info("Downloading filing HTML: %s", url)

    # Cache key includes filing_date so amendments at the same URL get a fresh fetch.
    # requests_cache keys on the URL; we append a query param to bust the cache on amendments.
    cache_busted_url = f"{url}?_filed={filing_date}"
    response = get(cache_busted_url)

    # requests_cache may have followed the busted URL but the actual content URL is clean.
    # Write the response body using the real URL's content.
    html_text = response.text
    out_path.write_text(html_text, encoding="utf-8")

    filing_scale = detect_filing_scale(html_text)
    logger.info("Filing scale detected: %d (accn=%s)", filing_scale, accn)
    return str(out_path), filing_scale


def ingest_filings(
    ticker: str,
    forms: list[str] | None = None,
    limit: int = 4,
) -> list[dict[str, object]]:
    """
    Full ingest pipeline for one ticker:
    1. Resolve ticker → CIK
    2. List recent filings filtered by form type
    3. Download primary HTML for each
    4. Return enriched filing metadata (with html_path, filing_scale)

    Does NOT render pages or build indexes — call render.py and index/ separately.
    """
    filings = list_filings(ticker, forms=forms, limit=limit)
    logger.info("Found %d filings for %s", len(filings), ticker)

    enriched = []
    for f in filings:
        try:
            html_path, scale = download_filing_html(
                cik=str(f["cik"]),
                accn=str(f["accn"]),
                primary_doc=str(f["primary_doc"]),
                filing_date=str(f["filing_date"]),
            )
            enriched.append({**f, "html_path": html_path, "filing_scale": scale})
        except Exception as exc:
            logger.error("Failed to download %s: %s", f["accn"], exc)

    return enriched


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ingest SEC filings for a ticker")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--list", action="store_true", help="List filings without downloading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        rows = list_filings(args.ticker, forms=args.forms, limit=args.limit)
        for r in rows:
            print(json.dumps(r, default=str))
        return

    results = ingest_filings(args.ticker, forms=args.forms, limit=args.limit)
    for r in results:
        print(json.dumps(r, default=str))


if __name__ == "__main__":
    _cli()
