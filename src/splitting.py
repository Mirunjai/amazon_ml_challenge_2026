"""Entity-level (Source 1) train/validation splitting.

Candidate pairs must NOT be split independently: every candidate pair for a
given Source 1 entity has to land entirely in train or entirely in
validation, otherwise the validation score leaks information from training.

This formalizes the "Reproducible split scaffold" cell already sketched in
the team notebook (RANDOM_SEED=42, VALIDATION_FRAC=0.15) into a reusable,
tested function.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_RANDOM_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.15


def entity_level_train_val_split(
    entity_ids: Iterable[str],
    *,
    val_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[set[str], set[str]]:
    """Split a population of Source 1 entity IDs into train/validation sets.

    Args:
        entity_ids: Source 1 entity IDs (duplicates are fine, they are
            deduplicated before splitting).
        val_fraction: Fraction of distinct entities to place in validation.
        seed: Random seed, for reproducibility across runs.

    Returns:
        (train_ids, val_ids) -- disjoint sets covering every distinct input
        entity ID exactly once.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    # Sort first so the split is reproducible regardless of input ordering
    # (a Python set's iteration order is not a stable contract to rely on).
    unique_ids = np.array(sorted({str(x) for x in entity_ids}))

    rng = np.random.default_rng(seed)
    val_size = int(round(len(unique_ids) * val_fraction))
    val_ids = set(rng.choice(unique_ids, size=val_size, replace=False)) if val_size else set()
    train_ids = set(unique_ids) - val_ids

    return train_ids, val_ids


def split_pairs_by_entity(
    pairs_df: pd.DataFrame,
    train_ids: set[str],
    val_ids: set[str],
    *,
    s1_col: str = "source1_entity_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a candidate-pair (or feature/label) DataFrame by S1 entity.

    Any row whose S1 id is in neither `train_ids` nor `val_ids` is silently
    dropped (e.g. rows for entities outside the population that was split).
    """
    s1 = pairs_df[s1_col].astype(str)
    train_df = pairs_df.loc[s1.isin(train_ids)].reset_index(drop=True)
    val_df = pairs_df.loc[s1.isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def assert_no_entity_leakage(train_ids: Iterable[str], val_ids: Iterable[str]) -> None:
    """Raise if any Source 1 entity appears in both splits."""
    overlap = set(map(str, train_ids)) & set(map(str, val_ids))
    if overlap:
        preview = sorted(overlap)[:10]
        suffix = "" if len(overlap) <= 10 else f" (+{len(overlap) - 10} more)"
        raise ValueError(f"Entity leakage between train/val split: {preview}{suffix}")
