"""Evaluation utilities for entity matching."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Mapping

MatchSets = Mapping[str, Iterable[str]]
BETA = 0.5
BETA_SQ = BETA ** 2


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
            if s1 in ground_truth:
                raise ValueError(f"Duplicate S1 ID in ground truth: {s1}")
            raw = (row["matched_entity_ids"] or "").strip()
            ground_truth[s1] = {x.strip() for x in raw.split(",") if x.strip()}
    return ground_truth


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f05_score(precision: float, recall: float) -> float:
    denominator = BETA_SQ * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1 + BETA_SQ) * precision * recall / denominator


def evaluate(
    ground_truth: MatchSets,
    predictions: MatchSets,
) -> dict[str, object]:
    """Return per-S1 TP/FP/FN/P/R/F0.5 plus macro metrics over ground-truth S1 IDs.

    Only S1 IDs present in ground_truth are evaluated; a missing prediction is
    treated as an empty set. Correct singletons (both sides empty) score 1.0;
    other undefined precision/recall cases use 0.0.
    """
    truth = {s1: set(matches) for s1, matches in ground_truth.items()}
    predicted = {s1: set(matches) for s1, matches in predictions.items()}
    per_s1: dict[str, dict[str, float | int]] = {}

    for s1 in sorted(truth):
        true_matches = truth[s1]
        predicted_matches = predicted.get(s1, set())
        tp = len(true_matches & predicted_matches)
        fp = len(predicted_matches - true_matches)
        fn = len(true_matches - predicted_matches)

        if not true_matches and not predicted_matches:
            precision = 1.0
            recall = 1.0
            f_score = 1.0
        else:
            precision = _safe_divide(tp, tp + fp)
            recall = _safe_divide(tp, tp + fn)
            f_score = _f05_score(precision, recall)

        per_s1[s1] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F0.5": f_score,
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