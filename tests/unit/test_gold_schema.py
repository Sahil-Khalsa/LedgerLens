"""
Unit tests for gold set schema validation.
Ensures malformed entries are caught before they silently break the eval runner.
"""

import pytest
from pydantic import ValidationError

from eval.gold import GoldItem, load_gold_set


class TestGoldItem:
    def test_valid_numeric_item(self) -> None:
        item = GoldItem(
            q="What was NVDA revenue in FY2024?",
            kind="numeric",
            answer_value=60922000000,
            unit="USD",
            concept="Revenues",
            accn="0001045810-24-022989",
        )
        assert item.kind == "numeric"

    def test_numeric_requires_answer_value(self) -> None:
        with pytest.raises(ValidationError):
            GoldItem(q="What was revenue?", kind="numeric")

    def test_reasoning_requires_rubric(self) -> None:
        with pytest.raises(ValidationError):
            GoldItem(q="How did margins trend?", kind="reasoning")

    def test_valid_reasoning_item(self) -> None:
        item = GoldItem(
            q="How did margins trend?",
            kind="reasoning",
            rubric="Should identify margin expansion and cite management commentary.",
        )
        assert item.kind == "reasoning"

    def test_negative_requires_insufficient_data_status(self) -> None:
        with pytest.raises(ValidationError):
            GoldItem(
                q="What was revenue in 2010?",
                kind="negative",
                expected_status="answered",
            )

    def test_valid_negative_item(self) -> None:
        item = GoldItem(
            q="What was revenue in 2010?",
            kind="negative",
            expected_status="insufficient_data",
        )
        assert item.expected_status == "insufficient_data"

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            GoldItem(q="   ", kind="numeric", answer_value=100.0)


class TestLoadGoldSet:
    def test_loads_committed_gold_set(self) -> None:
        items = load_gold_set()
        assert len(items) > 0

    def test_ci_subset_filter(self) -> None:
        all_items = load_gold_set()
        ci_items = load_gold_set(subset="ci")
        assert len(ci_items) <= len(all_items)
        assert all("ci" in item.subset for item in ci_items)

    def test_all_ci_items_have_valid_kind(self) -> None:
        ci_items = load_gold_set(subset="ci")
        valid_kinds = {"numeric", "reasoning", "negative", "adversarial"}
        for item in ci_items:
            assert item.kind in valid_kinds
