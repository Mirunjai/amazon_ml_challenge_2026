from __future__ import annotations

import pytest

from src.evaluation import evaluate


def test_correct_singleton_scores_one():
    result = evaluate(
        {"S1-1": set()},
        {"S1-1": set()},
    )

    metrics = result["per_s1"]["S1-1"]

    assert metrics["TP"] == 0
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F0.5"] == 1.0
    assert result["macro_f0.5"] == 1.0


def test_false_positive_on_singleton_scores_zero():
    result = evaluate(
        {"S1-1": set()},
        {"S1-1": {"S2-1"}},
    )

    metrics = result["per_s1"]["S1-1"]

    assert metrics["TP"] == 0
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0
    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0
    assert metrics["F0.5"] == 0.0


def test_missing_prediction_for_non_singleton_scores_zero():
    result = evaluate(
        {"S1-1": {"S2-1"}},
        {},
    )

    metrics = result["per_s1"]["S1-1"]

    assert metrics["TP"] == 0
    assert metrics["FP"] == 0
    assert metrics["FN"] == 1
    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0
    assert metrics["F0.5"] == 0.0


def test_perfect_single_match_scores_one():
    result = evaluate(
        {"S1-1": {"S2-1"}},
        {"S1-1": {"S2-1"}},
    )

    metrics = result["per_s1"]["S1-1"]

    assert metrics["TP"] == 1
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F0.5"] == 1.0


def test_multiple_matches_are_handled_correctly():
    result = evaluate(
        {"S1-1": {"S2-1", "S2-2", "S3-1"}},
        {"S1-1": {"S2-1", "S2-3"}},
    )

    metrics = result["per_s1"]["S1-1"]

    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 2


def test_missing_prediction_side_is_treated_as_empty():
    result = evaluate(
        {
            "S1-1": {"S2-1"},
            "S1-2": set(),
        },
        {
            "S1-1": {"S2-1"},
        },
    )

    assert result["per_s1"]["S1-1"]["F0.5"] == 1.0
    assert result["per_s1"]["S1-2"]["F0.5"] == 1.0


def test_extra_prediction_s1_is_not_added_to_macro_population():
    result = evaluate(
        {"S1-1": {"S2-1"}},
        {
            "S1-1": {"S2-1"},
            "S1-999": {"S2-999"},
        },
    )

    assert set(result["per_s1"].keys()) == {"S1-1"}
    assert result["macro_f0.5"] == 1.0


def test_macro_f05_averages_per_s1():
    result = evaluate(
        {
            "S1-1": {"S2-1"},
            "S1-2": set(),
        },
        {
            "S1-1": {"S2-2"},
            "S1-2": set(),
        },
    )

    # S1-1 -> F0.5 = 0
    # S1-2 -> F0.5 = 1
    # Macro = (0 + 1) / 2
    assert result["per_s1"]["S1-1"]["F0.5"] == 0.0
    assert result["per_s1"]["S1-2"]["F0.5"] == 1.0
    assert result["macro_f0.5"] == pytest.approx(0.5)


def test_ground_truth_duplicate_s1_is_rejected(tmp_path):
    from src.evaluation import parse_ground_truth

    path = tmp_path / "ground_truth.tsv"

    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-1\tS2-2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate S1 ID"):
        parse_ground_truth(path)