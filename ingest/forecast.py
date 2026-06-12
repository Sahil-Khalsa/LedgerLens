"""
XBRL time-series forecasting via linear regression.

IMPORTANT: All projected values are explicitly labeled as model outputs.
They are never presented as reported facts and must not enter the verifier.

Usage:
  uv run python -m ingest.forecast --ticker NVDA --concept Revenues --periods 2
"""

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)

_PROJECTION_DISCLAIMER = "projected — model output, not a reported fact"


class ForecastPoint(TypedDict):
    period_end: str   # YYYY-MM-DD
    value: float
    unit: str
    is_projection: bool
    label: str        # human-readable, e.g. "FY2026 (projected — model output, not a reported fact)"


def forecast_concept(
    ticker: str,
    concept: str,
    periods_ahead: int = 2,
) -> list[ForecastPoint]:
    """
    Extrapolate an annual XBRL concept trend using OLS linear regression.

    Pulls all 10-K rows for (ticker, concept) from the DB, fits a line through
    the year-indexed values, and projects forward by `periods_ahead` years.

    Returns a list of ForecastPoint — historical rows (is_projection=False) first,
    then projected rows (is_projection=True). Requires at least 2 historical rows.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for forecasting — install it with: uv add numpy") from e

    from index.store import Filing, SessionLocal, XbrlFact

    db = SessionLocal()
    try:
        rows = (
            db.query(XbrlFact)
            .join(Filing, Filing.accn == XbrlFact.accn)
            .filter(
                Filing.ticker == ticker.upper(),
                XbrlFact.concept == concept,
                XbrlFact.form.in_(["10-K", "10-K/A"]),
                XbrlFact.value.isnot(None),
            )
            .order_by(XbrlFact.end_date)
            .all()
        )
    finally:
        db.close()

    if len(rows) < 2:
        logger.warning(
            "Need >= 2 annual rows for %s %s forecast; found %d", ticker, concept, len(rows)
        )
        return []

    unit = rows[-1].unit or "USD"

    # Convert end-dates to decimal years for regression
    def _to_decimal_year(date_str: str) -> float:
        year = int(date_str[:4])
        month = int(date_str[5:7]) if len(date_str) >= 7 else 6
        return year + (month - 1) / 12.0

    years = np.array([_to_decimal_year(r.end_date) for r in rows], dtype=float)
    values = np.array([float(r.value) for r in rows], dtype=float)

    # OLS: y = slope * x + intercept
    design = np.vstack([years, np.ones_like(years)]).T
    result = np.linalg.lstsq(design, values, rcond=None)
    slope, intercept = float(result[0][0]), float(result[0][1])

    out: list[ForecastPoint] = []

    # Historical points
    for r in rows:
        out.append(
            ForecastPoint(
                period_end=r.end_date,
                value=float(r.value),
                unit=unit,
                is_projection=False,
                label=f"FY{r.end_date[:4]}",
            )
        )

    # Projected points
    last_year = int(rows[-1].end_date[:4])
    last_month_day = rows[-1].end_date[4:]  # e.g. "-01-28"
    for i in range(1, periods_ahead + 1):
        proj_year = last_year + i
        proj_decimal = _to_decimal_year(f"{proj_year}{last_month_day}")
        proj_value = max(0.0, slope * proj_decimal + intercept)
        out.append(
            ForecastPoint(
                period_end=f"{proj_year}{last_month_day}",
                value=proj_value,
                unit=unit,
                is_projection=True,
                label=f"FY{proj_year} ({_PROJECTION_DISCLAIMER})",
            )
        )

    return out


def format_forecast_table(points: list[ForecastPoint], scale: float = 1e9, unit_label: str = "B") -> str:
    """Return a plain-text table suitable for inclusion in an LLM prompt or report."""
    lines = ["Period       | Value          | Type"]
    lines.append("-" * 44)
    for p in points:
        kind = "PROJECTION*" if p["is_projection"] else "reported"
        scaled = p["value"] / scale
        lines.append(f"{p['period_end']} | {scaled:12.2f} {unit_label} | {kind}")
    if any(p["is_projection"] for p in points):
        lines.append("")
        lines.append(f"* {_PROJECTION_DISCLAIMER.capitalize()}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Forecast an XBRL concept trend")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--concept", required=True, help="XBRL concept (e.g. Revenues)")
    parser.add_argument("--periods", type=int, default=2, help="Periods ahead to project")
    parser.add_argument("--scale", type=float, default=1e9, help="Display scale divisor (default 1e9 = billions)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    points = forecast_concept(args.ticker, args.concept, args.periods)
    if not points:
        print("No data available for forecast.")
        return

    print(f"\n{args.ticker} — {args.concept} trend + {args.periods}-period projection\n")
    print(format_forecast_table(points, scale=args.scale))


if __name__ == "__main__":
    _cli()
