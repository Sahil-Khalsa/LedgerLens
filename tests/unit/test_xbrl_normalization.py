"""
Unit tests for XBRL normalization logic.
Pure functions — no DB, no network, no LLM.
"""

import pytest

from ingest.xbrl import (
    _make_row,
    compare_with_tolerance,
    detect_filing_scale,
    normalize_display_value,
    normalize_facts,
)


class TestDetectFilingScale:
    def test_detects_millions(self) -> None:
        html = "All amounts in millions of dollars"
        assert detect_filing_scale(html) == 1_000_000

    def test_detects_thousands(self) -> None:
        html = "Amounts stated in thousands"
        assert detect_filing_scale(html) == 1_000

    def test_detects_billions(self) -> None:
        html = "Figures presented in billions"
        assert detect_filing_scale(html) == 1_000_000_000

    def test_defaults_to_raw(self) -> None:
        html = "No scale declaration here"
        assert detect_filing_scale(html) == 1

    def test_case_insensitive(self) -> None:
        assert detect_filing_scale("IN MILLIONS") == 1_000_000


class TestNormalizeDisplayValue:
    def test_basic_millions(self) -> None:
        assert normalize_display_value("47,532", 1_000_000) == 47_532_000_000

    def test_with_dollar_sign(self) -> None:
        assert normalize_display_value("$47.5", 1_000_000_000) == 47_500_000_000

    def test_negative_parentheses(self) -> None:
        result = normalize_display_value("(1,234)", 1_000_000)
        assert result == -1_234_000_000

    def test_decimal_value(self) -> None:
        result = normalize_display_value("11.93", 1)
        assert result == pytest.approx(11.93)

    def test_raw_scale(self) -> None:
        assert normalize_display_value("47532000000", 1) == 47_532_000_000

    def test_invalid_returns_none(self) -> None:
        assert normalize_display_value("N/A", 1_000_000) is None

    def test_empty_returns_none(self) -> None:
        assert normalize_display_value("", 1_000_000) is None


class TestCompareWithTolerance:
    def test_exact_match(self) -> None:
        assert compare_with_tolerance(47_532_000_000, 47_532_000_000)

    def test_within_1_percent(self) -> None:
        # 0.07% difference — should match
        assert compare_with_tolerance(47.5e9, 47_532e6)

    def test_outside_1_percent(self) -> None:
        # 1.2% difference — should not match
        assert not compare_with_tolerance(48.1e9, 47_532e6)

    def test_zero_xbrl_value(self) -> None:
        assert compare_with_tolerance(0.0, 0.0)
        assert not compare_with_tolerance(100.0, 0.0)

    def test_custom_tolerance(self) -> None:
        assert compare_with_tolerance(48.1e9, 47_532e6, tol=0.02)


class TestNormalizeFacts:
    def test_skips_null_values(self) -> None:
        companyfacts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"val": None, "accn": "0001-01", "end": "2024-01-28"},
                                {"val": 60922000000, "accn": "0001-01", "end": "2024-01-28"},
                            ]
                        }
                    }
                }
            }
        }
        rows = normalize_facts(companyfacts)
        assert len(rows) == 1
        assert rows[0]["value"] == 60922000000

    def test_skips_missing_accn(self) -> None:
        companyfacts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"val": 100, "end": "2024-01-28"},  # no accn
                            ]
                        }
                    }
                }
            }
        }
        rows = normalize_facts(companyfacts)
        assert len(rows) == 0

    def test_flow_vs_stock_tagging(self) -> None:
        companyfacts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                # Flow fact — has start date
                                {
                                    "val": 100,
                                    "accn": "001",
                                    "start": "2023-02-01",
                                    "end": "2024-01-28",
                                },
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                # Stock fact — no start date
                                {"val": 200, "accn": "001", "end": "2024-01-28"},
                            ]
                        }
                    },
                }
            }
        }
        rows = normalize_facts(companyfacts)
        revenue_row = next(r for r in rows if r["concept"] == "Revenues")
        assets_row = next(r for r in rows if r["concept"] == "Assets")
        assert revenue_row["is_flow"] is True
        assert assets_row["is_flow"] is False

    def test_make_row_returns_none_for_null(self) -> None:
        assert _make_row("Revenues", "USD", {"val": None, "accn": "001"}) is None

    def test_make_row_returns_none_for_missing_accn(self) -> None:
        assert _make_row("Revenues", "USD", {"val": 100}) is None
