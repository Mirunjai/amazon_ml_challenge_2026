"""Ground-truth -> candidate-pair label construction.

This module turns whatever candidate pairs Sudarshan's blocking stage produces
into labeled training rows, using Narendra's ground-truth parser
(`src.evaluation.parse_ground_truth`) rather than re-implementing GT parsing.

It intentionally does NOT generate candidate pairs itself (no blocking, no
S1 x S2/S3 cartesian product). It only labels pairs that are handed to it.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from .evaluation import parse_ground_truth
from .io_utils import validate_columns

GroundTruthLike = "pd.DataFrame | Mapping[str, set[str]]"


def _resolve_ground_truth(ground_truth) -> Mapping[str, set[str]]:
    """Accept either the raw ground-truth DataFrame or an already-parsed dict."""
    if isinstance(ground_truth, pd.DataFrame):
        return parse_ground_truth(ground_truth)
    return ground_truth


def label_candidate_pairs(
    pairs: "pd.DataFrame | Iterable[tuple[str, str]]",
    ground_truth: "pd.DataFrame | Mapping[str, set[str]]",
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    label_col: str = "label",
) -> pd.DataFrame:
    """Label supplied candidate pairs against the training ground truth.

    Only pairs that are actually supplied in `pairs` are labeled. This never
    materializes the full S1 x (S2 union S3) space -- if a true match was not
    proposed as a candidate, it simply does not appear here (that is a
    candidate-recall problem for the blocking stage, not something this
    function should paper over).

    Args:
        pairs: Either a DataFrame with `s1_col`/`candidate_col` columns (one
            row per candidate pair, e.g. exploded from Sudarshan's
            candidate_pairs.tsv), or any iterable of (s1_id, candidate_id)
            tuples.
        ground_truth: Either the raw train_ground_truth.tsv DataFrame
            (source1_entity_id, matched_entity_ids) or an already-parsed
            dict[str, set[str]] (e.g. the output of
            `src.evaluation.parse_ground_truth`, or a cached result of it).
        s1_col, candidate_col: Column names to use/expect.
        label_col: Name of the output label column.

    Returns:
        DataFrame with columns [s1_col, candidate_col, label_col], one row
        per input pair, label_col is 1 if the pair is a true match else 0.
    """
    gt_map = _resolve_ground_truth(ground_truth)

    if isinstance(pairs, pd.DataFrame):
        validate_columns(pairs.columns, [s1_col, candidate_col], context="candidate pairs")
        s1_ids = pairs[s1_col].astype(str).tolist()
        candidate_ids = pairs[candidate_col].astype(str).tolist()
    else:
        pairs_list = list(pairs)
        s1_ids = [str(s1_id) for s1_id, _ in pairs_list]
        candidate_ids = [str(candidate_id) for _, candidate_id in pairs_list]

    labels = [
        int(candidate_id in gt_map.get(s1_id, set()))
        for s1_id, candidate_id in zip(s1_ids, candidate_ids, strict=False)
    ]

    return pd.DataFrame(
        {
            s1_col: s1_ids,
            candidate_col: candidate_ids,
            label_col: labels,
        }
    )


def explode_candidate_pairs(
    candidate_df: pd.DataFrame,
    *,
    s1_col: str = "source1_entity_id",
    candidates_col: str = "candidate_entity_ids",
    output_candidate_col: str = "candidate_entity_id",
) -> pd.DataFrame:
    """Convert the official wide candidate_pairs.tsv shape into long pairs.

    Input: one row per S1 entity, with a comma-separated `candidate_entity_ids`
    column (this is the format described in the problem statement and in
    `src.config.CANDIDATE_COLUMNS`).

    Output: one row per (source1_entity_id, candidate_entity_id) pair, ready
    to feed into `src.features.build_pair_feature_frame` and
    `label_candidate_pairs`. Rows where the candidate list is empty
    (singleton S1 entities) simply contribute no rows.

    This is a pure reshape utility -- it does not decide which candidates
    belong in the list; that remains Sudarshan's blocking stage.
    """
    validate_columns(candidate_df.columns, [s1_col, candidates_col], context="candidate_pairs")

    rows: list[tuple[str, str]] = []
    for s1_id, raw in zip(
        candidate_df[s1_col].astype(str),
        candidate_df[candidates_col].fillna("").astype(str),
        strict=False,
    ):
        for candidate_id in (token.strip() for token in raw.split(",")):
            if candidate_id:
                rows.append((s1_id, candidate_id))

    return pd.DataFrame(rows, columns=[s1_col, output_candidate_col])


def group_pairs_by_s1(
    pairs: pd.DataFrame,
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
) -> dict[str, set[str]]:
    """Reshape long-format pairs into {s1_id: {candidate_id, ...}}.

    Useful for feeding Sudarshan's candidate set into
    `src.evaluation.candidate_recall` without duplicating that function.
    """
    grouped: dict[str, set[str]] = {}
    for s1_id, group in pairs.groupby(pairs[s1_col].astype(str), sort=False):
        grouped[str(s1_id)] = set(group[candidate_col].astype(str))
    return grouped
