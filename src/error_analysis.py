"""Error-analysis scaffolding for inspecting model predictions.

Builds one wide table per (S1, candidate) pair with everything needed to
eyeball hard cases later: probability, predicted/true label, outcome
(TP/FP/FN/TN), the raw business fields from both sides, and (optionally) the
pairwise feature values.

This is infrastructure only -- it does not decide what "hard" means or which
new features to add. That comes after a real model exists.
"""

from __future__ import annotations

import pandas as pd

_OUTCOMES = {
    (1, 1): "TP",
    (0, 1): "FP",
    (1, 0): "FN",
    (0, 0): "TN",
}


def categorize_predictions(
    df: pd.DataFrame,
    threshold: float,
    *,
    label_col: str = "label",
    score_col: str = "match_probability",
    predicted_col: str = "predicted_label",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """Add predicted_label (0/1 at `threshold`) and outcome (TP/FP/FN/TN) columns."""
    out = df.copy()
    out[predicted_col] = (out[score_col] >= threshold).astype(int)
    out[outcome_col] = [
        _OUTCOMES[(int(true), int(pred))]
        for true, pred in zip(out[label_col], out[predicted_col], strict=False)
    ]
    return out


def _rename_entity_lookup(lookup: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    return lookup.rename(
        columns={
            "entity_id": id_col,
            "business_name": f"{prefix}_business_name",
            "business_address": f"{prefix}_business_address",
            "country": f"{prefix}_country",
        }
    )[[id_col, f"{prefix}_business_name", f"{prefix}_business_address", f"{prefix}_country"]]


def build_error_table(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    threshold: float,
    *,
    feature_df: pd.DataFrame | None = None,
    source1_lookup: pd.DataFrame | None = None,
    candidate_lookup: pd.DataFrame | None = None,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    score_col: str = "match_probability",
    label_col: str = "label",
) -> pd.DataFrame:
    """Assemble the full inspection table for one threshold.

    Args:
        score_df: Output of `src.model.predict_match_scores`
            (s1_col, candidate_col, score_col).
        label_df: True labels for the same pairs, e.g. output of
            `src.labeling.label_candidate_pairs` (s1_col, candidate_col,
            label_col).
        threshold: Probability cutoff to turn scores into predicted labels.
        feature_df: Optional pairwise feature values (e.g. output of
            `src.features.build_pair_feature_frame`) to merge in for context.
        source1_lookup / candidate_lookup: Optional DataFrames with columns
            entity_id, business_name, business_address, country -- e.g.
            train_source1.tsv and a concatenation of train_source2.tsv /
            train_source3.tsv. When supplied, the returned table includes the
            raw business fields for both sides of the pair.

    Returns:
        One row per pair with probability, true/predicted label, outcome,
        and whatever context columns were supplied.
    """
    merged = score_df.merge(
        label_df[[s1_col, candidate_col, label_col]],
        on=[s1_col, candidate_col],
        how="inner",
    )
    merged = categorize_predictions(
        merged, threshold, label_col=label_col, score_col=score_col
    )

    if feature_df is not None:
        extra_cols = [c for c in feature_df.columns if c not in (s1_col, candidate_col)]
        merged = merged.merge(
            feature_df[[s1_col, candidate_col, *extra_cols]],
            on=[s1_col, candidate_col],
            how="left",
        )

    if source1_lookup is not None:
        merged = merged.merge(
            _rename_entity_lookup(source1_lookup, s1_col, "source1"),
            on=s1_col,
            how="left",
        )

    if candidate_lookup is not None:
        merged = merged.merge(
            _rename_entity_lookup(candidate_lookup, candidate_col, "candidate"),
            on=candidate_col,
            how="left",
        )

    return merged


def filter_outcome(
    error_table: pd.DataFrame,
    outcome: str,
    *,
    outcome_col: str = "outcome",
    sort_by: str = "match_probability",
    ascending: bool = False,
    n: int | None = None,
) -> pd.DataFrame:
    """Slice the error table down to one outcome (TP/FP/FN/TN), sorted for review.

    For FPs, sorting by probability descending surfaces the most confidently
    wrong predictions first. For FNs, ascending surfaces the true matches the
    model was most confidently wrong about missing.
    """
    if outcome not in _OUTCOMES.values():
        raise ValueError(f"outcome must be one of {sorted(set(_OUTCOMES.values()))}, got {outcome!r}")
    subset = error_table.loc[error_table[outcome_col] == outcome].sort_values(
        sort_by, ascending=ascending
    )
    return subset.head(n) if n is not None else subset.reset_index(drop=True)
