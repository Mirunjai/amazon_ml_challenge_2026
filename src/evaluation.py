"""Evaluation utilities for entity matching."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

MatchSets = Mapping[str, Iterable[str]]

BETA = 0.5
BETA_SQ = BETA**2


# ============================================================
# Ground-truth parsing
# ============================================================

def parse_ground_truth(
    source: str | Path | pd.DataFrame,
) -> dict[str, set[str]]:
    """
    Parse ground truth into:
        S1 entity ID -> set of true matched S2/S3 IDs

    Accepts either a TSV path or a pandas DataFrame.
    Duplicate or empty S1 IDs are rejected.
    """
    if isinstance(source, pd.DataFrame):
        df = source
        required = {"source1_entity_id", "matched_entity_ids"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Ground-truth DataFrame must contain columns "
                "'source1_entity_id' and 'matched_entity_ids'."
            )

        rows = zip(
            df["source1_entity_id"].fillna("").astype(str),
            df["matched_entity_ids"].fillna("").astype(str),
            strict=False,
        )

        ground_truth: dict[str, set[str]] = {}
        for s1_raw, matches_raw in rows:
            s1 = s1_raw.strip()
            if not s1 or s1.lower() == "nan":
                raise ValueError(
                    "Encountered a ground-truth row with an empty S1 ID."
                )
            if s1 in ground_truth:
                raise ValueError(f"Duplicate S1 ID in ground truth: {s1}")

            ground_truth[s1] = {
                x.strip() for x in matches_raw.split(",") if x.strip()
            }

        return ground_truth

    path = Path(source)
    ground_truth = {}

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
                raise ValueError(
                    "Encountered a ground-truth row with an empty S1 ID."
                )
            if s1 in ground_truth:
                raise ValueError(f"Duplicate S1 ID in ground truth: {s1}")

            raw = (row["matched_entity_ids"] or "").strip()
            ground_truth[s1] = {
                x.strip() for x in raw.split(",") if x.strip()
            }

    return ground_truth


# ============================================================
# Core pair-level metric
# ============================================================

def precision_recall_f05(
    actual: Iterable[str],
    predicted: Iterable[str],
    beta: float = BETA,
) -> tuple[float, float, float]:
    """
    Calculate Precision, Recall and F_beta for one S1 entity.

    Correct singleton (actual == set() and predicted == set())
    receives Precision = 1.0, Recall = 1.0, F0.5 = 1.0.
    """
    if beta <= 0:
        raise ValueError("beta must be greater than 0.")

    actual_set = set(actual)
    pred_set = set(predicted)

    # Official singleton handling.
    if not actual_set and not pred_set:
        return 1.0, 1.0, 1.0

    tp = len(pred_set & actual_set)
    fp = len(pred_set - actual_set)
    fn = len(actual_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    f_beta = (
        (1 + beta_sq) * precision * recall / denominator
        if denominator
        else 0.0
    )

    return precision, recall, f_beta


# ============================================================
# Macro F0.5
# ============================================================

def macro_f05(
    actual_by_s1: MatchSets,
    predicted_by_s1: MatchSets,
    beta: float = BETA,
) -> float:
    """
    Calculate macro F0.5 across the S1 entities defined by actual_by_s1.
    """
    if beta <= 0:
        raise ValueError("beta must be greater than 0.")

    if not actual_by_s1:
        return 0.0

    scores = [
        precision_recall_f05(
            actual_matches,
            predicted_by_s1.get(s1_id, set()),
            beta,
        )[2]
        for s1_id, actual_matches in actual_by_s1.items()
    ]

    return float(np.mean(scores))


# ============================================================
# Candidate recall
# ============================================================

def candidate_recall(
    actual_by_s1: MatchSets,
    candidate_by_s1: MatchSets,
) -> float:
    """
    Recall of the candidate-generation stage over all ground-truth matches.
    Answers: Of all true matches, what fraction survived blocking?
    """
    total_true_matches = 0
    retained_true_matches = 0

    for s1_id, actual_matches in actual_by_s1.items():
        actual = set(actual_matches)
        candidates = set(candidate_by_s1.get(s1_id, set()))

        total_true_matches += len(actual)
        retained_true_matches += len(candidates & actual)

    return (
        retained_true_matches / total_true_matches
        if total_true_matches
        else 1.0
    )


# ============================================================
# Prediction generation from model scores
# ============================================================

def predictions_from_scores(
    score_df: pd.DataFrame,
    threshold: float,
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    score_col: str = "match_probability",
) -> dict[str, set[str]]:
    """
    Convert per-candidate match probabilities into multi-match predictions per S1.
    """
    required = {s1_col, candidate_col, score_col}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(f"Score DataFrame missing columns: {sorted(missing)}")

    grouped: dict[str, set[str]] = {
        str(s1): set() for s1 in score_df[s1_col].unique()
    }

    passed = score_df.loc[
        score_df[score_col] >= threshold, [s1_col, candidate_col]
    ]

    for s1_id, cand_id in zip(
        passed[s1_col].astype(str),
        passed[candidate_col].astype(str),
        strict=False,
    ):
        grouped[s1_id].add(cand_id)

    return grouped


# ============================================================
# Threshold sweep
# ============================================================

def threshold_sweep(
    score_df: pd.DataFrame,
    actual_by_s1: MatchSets,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate several probability thresholds and return a summary DataFrame.
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.10, 0.951, 0.01)]

    actual_sets = {s1: set(matches) for s1, matches in actual_by_s1.items()}
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        predictions = predictions_from_scores(score_df, threshold)
        f_score = macro_f05(actual_sets, predictions, BETA)

        tp = fp = fn = 0
        for s1_id, actual in actual_sets.items():
            predicted = predictions.get(s1_id, set())
            tp += len(predicted & actual)
            fp += len(predicted - actual)
            fn += len(actual - predicted)

        global_precision = tp / (tp + fp) if (tp + fp) else 1.0
        global_recall = tp / (tp + fn) if (tp + fn) else 1.0

        rows.append(
            {
                "threshold": float(threshold),
                "macro_f05": float(f_score),
                "global_precision": float(global_precision),
                "global_recall": float(global_recall),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("threshold")
        .reset_index(drop=True)
    )


# ============================================================
# Rich evaluator
# ============================================================

def evaluate(
    ground_truth: MatchSets,
    predictions: MatchSets,
) -> dict[str, object]:
    """
    Rich evaluation report over the S1 population present in ground_truth.
    Missing predictions are treated as empty sets.
    Correct singletons receive Precision = 1.0, Recall = 1.0, F0.5 = 1.0.
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

        precision, recall, f_score = precision_recall_f05(
            true_matches,
            predicted_matches,
            BETA,
        )

        per_s1[s1] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F0.5": f_score,
        }

    if per_s1:
        n = len(per_s1)
        macro_precision = sum(float(x["Precision"]) for x in per_s1.values()) / n
        macro_recall = sum(float(x["Recall"]) for x in per_s1.values()) / n
        macro_f = sum(float(x["F0.5"]) for x in per_s1.values()) / n
    else:
        macro_precision = macro_recall = macro_f = 0.0

    return {
        "per_s1": per_s1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f0.5": macro_f,
    }


def evaluate_from_file(
    path: str | Path,
    predictions: MatchSets,
) -> dict[str, object]:
    """Parse ground truth from a TSV and evaluate predictions."""
    return evaluate(parse_ground_truth(path), predictions)