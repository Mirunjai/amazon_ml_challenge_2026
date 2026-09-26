import pandas as pd

from src.evaluation import macro_f05, precision_recall_f05, parse_ground_truth
from src.normalization import normalize_name


def test_normalize_name():
    assert normalize_name("  ABC, Pvt. Ltd.  ") == "abc pvt. ltd."


def test_exact_set_score():
    p, r, f = precision_recall_f05({"A", "B"}, {"A", "B"})
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0


def test_ground_truth_parser():
    df = pd.DataFrame({
        "source1_entity_id": ["S1-1"],
        "matched_entity_ids": ["S2-1,S3-2"],
    })
    parsed = parse_ground_truth(df)
    assert parsed == {"S1-1": {"S2-1", "S3-2"}}


def test_macro_f05_empty_singleton():
    actual = {"S1-1": set(), "S1-2": {"S2-1"}}
    pred = {"S1-1": set(), "S1-2": {"S2-1"}}
    assert macro_f05(pred, actual) == 1.0
