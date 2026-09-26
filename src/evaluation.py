from __future__ import annotations

import numpy as np
import pandas as pd


def precision_recall_f05(
    predicted: set[str],
    actual: set[str],
    beta: float = 0.5,
) -> tuple[float, float, float]:
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not actual else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    beta2 = beta * beta
    denom = beta2 * precision + recall
    f_beta = ((1 + beta2) * precision * recall / denom) if denom else 0.0
    return precision, recall, f_beta


def macro_f05(
    predicted_by_s1: dict[str, set[str]],
    actual_by_s1: dict[str, set[str]],
    beta: float = 0.5,
) -> float:
    scores = []
    for s1_id, actual in actual_by_s1.items():
        predicted = predicted_by_s1.get(s1_id, set())
        _, _, f = precision_recall_f05(predicted, actual, beta)
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def candidate_recall(
    candidate_by_s1: dict[str, set[str]],
    actual_by_s1: dict[str, set[str]],
) -> float:
    total = 0
    found = 0
    for s1_id, actual in actual_by_s1.items():
        total += len(actual)
        found += len(candidate_by_s1.get(s1_id, set()) & actual)
    return found / total if total else 1.0


def parse_ground_truth(df: pd.DataFrame) -> dict[str, set[str]]:
    """Convert the official comma-separated target column into sets."""
    out: dict[str, set[str]] = {}
    for s1_id, value in zip(
        df["source1_entity_id"].astype(str),
        df["matched_entity_ids"].fillna("").astype(str),
        strict=False,
    ):
        ids = {x.strip() for x in value.split(",") if x.strip()}
        out[s1_id] = ids
    return out


def predictions_from_scores(
    score_df: pd.DataFrame,
    threshold: float,
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    score_col: str = "match_probability",
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for s1_id, group in score_df.groupby(s1_col, sort=False):
        grouped[str(s1_id)] = set(
            group.loc[group[score_col] >= threshold, candidate_col].astype(str)
        )
    return grouped


def threshold_sweep(
    score_df: pd.DataFrame,
    actual_by_s1: dict[str, set[str]],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.10, 0.951, 0.01)]

    rows = []
    for threshold in thresholds:
        pred = predictions_from_scores(score_df, threshold)
        f = macro_f05(pred, actual_by_s1)
        # Global pair precision/recall are supplementary diagnostics.
        tp = fp = fn = 0
        for s1_id, actual in actual_by_s1.items():
            p = pred.get(s1_id, set())
            tp += len(p & actual)
            fp += len(p - actual)
            fn += len(actual - p)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        rows.append(
            {
                "threshold": threshold,
                "macro_f05": f,
                "global_precision": precision,
                "global_recall": recall,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
