"""Model-independent training/prediction interface for the pairwise matcher.

candidate pairs -> labels -> pairwise features -> training data -> classifier
-> prediction probabilities

This module does NOT decide the final model. Any object exposing a
scikit-learn-compatible `.fit(X, y)` / `.predict_proba(X)` interface can be
passed in (ExtraTreesClassifier, RandomForestClassifier,
HistGradientBoostingClassifier, etc. all satisfy this). The default
classifier below exists only so this module is independently testable
tonight -- it is a placeholder, not a claimed final model choice.

`predict_match_scores` outputs exactly the (source1_entity_id,
candidate_entity_id, match_probability) shape that
`src.evaluation.predictions_from_scores` and `src.evaluation.threshold_sweep`
already expect, so Narendra's evaluator can consume it without changes.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

# Baseline features currently implemented in src/features.py.
# Passed explicitly wherever training/prediction happens so a future feature
# version can be swapped in without touching this module.
BASELINE_FEATURE_COLUMNS: list[str] = [
    "name_ratio",
    "name_token_set_ratio",
    "address_ratio",
    "address_token_set_ratio",
    "country_match",
    "s1_address_missing",
    "candidate_address_missing",
    "both_address_missing",
]


class PairwiseClassifier(Protocol):
    """Minimal interface a matching model must satisfy to be swappable here."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def default_classifier(*, random_state: int = 42) -> PairwiseClassifier:
    """A reasonable placeholder classifier for exercising this interface.

    NOT a claim of the final model choice -- that is an open experiment
    (ExtraTrees / RandomForest / HistGradientBoosting are all candidates per
    the project brief). Swap by passing `model=...` to `train_classifier`.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
    )


def assemble_training_data(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    label_col: str = "label",
) -> pd.DataFrame:
    """Join pairwise features (Namitha/features.py output) with labels
    (labeling.label_candidate_pairs output) into one training frame.

    An inner join is used deliberately: a pair without both a computed
    feature row and a label is not usable for training, and this makes any
    mismatch between the two immediately visible (via row-count) rather than
    silently producing NaNs.
    """
    merged = feature_df.merge(
        label_df[[s1_col, candidate_col, label_col]],
        on=[s1_col, candidate_col],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(feature_df):
        raise ValueError(
            "assemble_training_data: not every feature row had a matching label "
            f"({len(merged)} matched out of {len(feature_df)} feature rows). "
            "Check that `feature_df` and `label_df` were built from the same pairs."
        )
    return merged


def train_classifier(
    training_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    label_col: str = "label",
    model: PairwiseClassifier | None = None,
) -> PairwiseClassifier:
    """Fit a pairwise classifier on an assembled training frame.

    Args:
        training_df: Output of `assemble_training_data` (or anything with the
            same feature + label columns).
        feature_columns: Which columns to use as model input. Defaults to
            `BASELINE_FEATURE_COLUMNS`; pass an explicit list once new
            features are added so callers stay in control of feature version.
        label_col: Name of the 0/1 label column.
        model: Any unfitted scikit-learn-compatible classifier. Defaults to
            `default_classifier()` (a placeholder, see its docstring).

    Returns:
        The fitted model (same object passed in, or a new default one).
    """
    feature_columns = feature_columns or BASELINE_FEATURE_COLUMNS
    missing = [c for c in feature_columns if c not in training_df.columns]
    if missing:
        raise ValueError(f"train_classifier: training_df is missing feature columns: {missing}")

    if model is None:
        model = default_classifier()
    X = training_df[feature_columns].to_numpy(dtype=float)
    y = training_df[label_col].to_numpy()
    model.fit(X, y)
    return model


def predict_match_scores(
    model: PairwiseClassifier,
    feature_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    score_col: str = "match_probability",
) -> pd.DataFrame:
    """Score candidate pairs with a fitted model.

    Output columns (s1_col, candidate_col, score_col) match the defaults
    expected by `src.evaluation.predictions_from_scores` and
    `src.evaluation.threshold_sweep`, so downstream evaluation code does not
    need to know anything about how the scores were produced.
    """
    feature_columns = feature_columns or BASELINE_FEATURE_COLUMNS
    missing = [c for c in feature_columns if c not in feature_df.columns]
    if missing:
        raise ValueError(f"predict_match_scores: feature_df is missing feature columns: {missing}")

    X = feature_df[feature_columns].to_numpy(dtype=float)
    proba = model.predict_proba(X)

    classes = list(getattr(model, "classes_", [0, 1]))
    positive_idx = classes.index(1) if 1 in classes else len(classes) - 1
    scores = proba[:, positive_idx]

    return pd.DataFrame(
        {
            s1_col: feature_df[s1_col].astype(str).to_numpy(),
            candidate_col: feature_df[candidate_col].astype(str).to_numpy(),
            score_col: scores,
        }
    )
