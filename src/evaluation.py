"""Evaluation utilities for entity matching."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Mapping

MatchSets = Mapping[str, Iterable[str]]


def parse_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """Parse train_ground_truth.tsv into S1 -> set(true matched IDs)."""
    path = Path(path)
    ground_truth: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "Ground-truth TSV must contain columns "
                "'source1_entity_id' and 'matched_entity_ids'."
            )
        for row in reader:
            s1 = (row["source1_entity_id"] or "").strip()
            if not s1:
                raise ValueError("Encountered a ground-truth row with an empty S1 ID.")
            raw = (row["matched_entity_ids"] or "").strip()
            ground_truth[s1] = {x.strip() for x in raw.split(",") if x.strip()}
    return ground_truth


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be greater than 0.")
    denominator = beta**2 * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denominator


def evaluate(
    ground_truth: MatchSets,
    predictions: MatchSets,
    *,
    beta: float = 0.5,
) -> dict[str, object]:
    """Return per-S1 TP/FP/FN/P/R/F0.5 plus macro metrics.

    S1 IDs appearing on either side are evaluated; a missing side is treated
    as an empty set. Undefined precision/recall cases use 0.0.
    """
    if beta <= 0:
        raise ValueError("beta must be greater than 0.")

    truth = {s1: set(matches) for s1, matches in ground_truth.items()}
    predicted = {s1: set(matches) for s1, matches in predictions.items()}
    all_s1 = set(truth) | set(predicted)
    per_s1: dict[str, dict[str, float | int]] = {}

    for s1 in sorted(all_s1):
        true_matches = truth.get(s1, set())
        predicted_matches = predicted.get(s1, set())
        tp = len(true_matches & predicted_matches)
        fp = len(predicted_matches - true_matches)
        fn = len(true_matches - predicted_matches)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        per_s1[s1] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F0.5": _f_beta(precision, recall, beta),
        }

    if per_s1:
        macro_precision = sum(x["Precision"] for x in per_s1.values()) / len(per_s1)
        macro_recall = sum(x["Recall"] for x in per_s1.values()) / len(per_s1)
        macro_f = sum(x["F0.5"] for x in per_s1.values()) / len(per_s1)
    else:
        macro_precision = macro_recall = macro_f = 0.0

    return {
        "per_s1": per_s1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f0.5": macro_f,
    }


def evaluate_from_file(path: str | Path, predictions: MatchSets) -> dict[str, object]:
    """Parse ground truth from a TSV and evaluate predictions."""
    return evaluate(parse_ground_truth(path), predictions)
