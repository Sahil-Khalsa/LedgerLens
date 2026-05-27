"""
Single entry point for all SEC EDGAR HTTP access.
All other modules must call get() / get_stream() from here — never requests.get() directly.

Implements:
- Descriptive User-Agent (SEC requirement)
- Thread-safe rate throttle (~6-7 req/s, well under the ~10 req/s cap)
- requests_cache with 24-hour TTL
- Streaming response for large payloads (companyfacts can be 200+ MB)
"""

import logging
import threading
import time
from collections.abc import Iterator

import requests
import requests_cache

from config import settings

logger = logging.getLogger(__name__)

# One cached session shared across the process.
_session = requests_cache.CachedSession(
    cache_name="edgar_cache",
    expire_after=86400,  # 24 hours
    allowable_codes=[200],
)

# Thread-safe throttle state.
_throttle_lock = threading.Lock()
_last_request_time: float = 0.0
_MIN_INTERVAL = 0.15  # seconds between requests → ~6-7 req/s


def _throttle() -> None:
    global _last_request_time
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _headers() -> dict[str, str]:
    return {"User-Agent": settings.edgar_user_agent}


def get(url: str) -> requests.Response:
    """Throttled, cached GET. Raises on non-2xx."""
    _throttle()
    logger.debug("GET %s", url)
    r = _session.get(url, headers=_headers())
    r.raise_for_status()
    return r


def get_stream(url: str) -> Iterator[bytes]:
    """
    Throttled streaming GET for large responses (e.g. companyfacts JSON).
    Bypasses the cache — callers are responsible for their own persistence.
    Yields raw byte chunks.
    """
    _throttle()
    logger.debug("GET (stream) %s", url)
    with requests.get(url, headers=_headers(), stream=True, timeout=120) as r:
        r.raise_for_status()
        yield from r.iter_content(chunk_size=65536)


def get_ticker_to_cik() -> dict[str, str]:
    """
    Returns mapping of uppercase ticker → zero-padded 10-digit CIK string.
    Result is cached by requests_cache for 24 hours.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    data: dict[str, dict[str, object]] = get(url).json()
    return {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def resolve_cik(ticker: str) -> str:
    """Resolve ticker (case-insensitive) to zero-padded 10-digit CIK. Raises KeyError if unknown."""
    mapping = get_ticker_to_cik()
    key = ticker.upper()
    if key not in mapping:
        raise KeyError(f"Ticker {ticker!r} not found in SEC company_tickers.json")
    return mapping[key]


def get_submissions(cik10: str) -> dict[str, object]:
    """Fetch the submissions JSON for a company, following pagination."""
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    data: dict[str, object] = get(url).json()
    return data


def get_all_filing_rows(cik10: str) -> list[dict[str, object]]:
    """
    Return all filing rows (form, accn, filingDate, primaryDocument, reportDate)
    for a company, following the submissions pagination.
    """
    data = get_submissions(cik10)
    filings_block = data.get("filings", {})
    assert isinstance(filings_block, dict)

    rows = _extract_rows(filings_block.get("recent", {}))

    # Pagination: additional filing pages listed under filings.files[]
    for extra in filings_block.get("files", []):
        assert isinstance(extra, dict)
        name = str(extra["name"])
        extra_data: dict[str, object] = get(f"https://data.sec.gov/submissions/{name}").json()
        rows.extend(_extract_rows(extra_data))

    return rows


def _extract_rows(block: dict[str, object]) -> list[dict[str, object]]:
    """Convert the parallel-array structure in a submissions block to a list of dicts."""
    if not block:
        return []
    keys = ["accessionNumber", "form", "filingDate", "primaryDocument", "reportDate"]
    arrays = {k: block.get(k, []) for k in keys}
    length = len(arrays["accessionNumber"])  # type: ignore[arg-type]
    rows = []
    for i in range(length):
        rows.append({k: arrays[k][i] for k in keys})  # type: ignore[index]
    return rows


def build_doc_url(cik: str, accn: str, filename: str) -> str:
    """Build the EDGAR archives URL for a filing document."""
    accn_nodashes = accn.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodashes}/{filename}"
