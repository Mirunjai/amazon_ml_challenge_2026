from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .normalization import normalize_address, normalize_name


@dataclass(frozen=True)
class CandidateBlock:
    """One blocking rule. Higher recall is preferred, then we control volume."""
    name: str
    key_fn: object


def build_index(
    df: pd.DataFrame,
    key_column: str,
    *,
    id_column: str = "entity_id",
) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for key, entity_id in zip(df[key_column].fillna(""), df[id_column], strict=False):
        if key:
            index[str(key)].append(str(entity_id))
    return dict(index)


def add_normalized_keys(
    df: pd.DataFrame,
    *,
    name_col: str = "business_name",
    address_col: str = "business_address",
    country_col: str = "country",
) -> pd.DataFrame:
    out = df.copy()
    out["_name_norm"] = out[name_col].map(normalize_name)
    out["_name_nosuffix"] = out[name_col].map(
        lambda x: normalize_name(x, strip_legal_suffixes=True)
    )
    out["_address_norm"] = out[address_col].map(normalize_address)
    out["_country_norm"] = out[country_col].fillna("").map(lambda x: str(x).casefold().strip())
    out["_name_prefix"] = out["_name_nosuffix"].str.replace(" ", "", regex=False).str[:8]
    return out


def candidate_ids_from_exact_keys(
    source1_row: pd.Series,
    source2_index: Mapping[str, list[str]],
    source3_index: Mapping[str, list[str]],
    *,
    keys: Iterable[str],
) -> set[str]:
    """Union candidates across selected precomputed key columns.

    This is intentionally a framework, not the final challenge blocking strategy.
    The main notebook should measure recall/volume for each strategy before locking it.
    """
    candidates: set[str] = set()
    for key in keys:
        value = source1_row.get(key, "")
        if value:
            candidates.update(source2_index.get(value, []))
            candidates.update(source3_index.get(value, []))
    return candidates


def merge_candidate_maps(*maps: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for mapping in maps:
        for s1_id, ids in mapping.items():
            merged[s1_id].update(ids)
    return dict(merged)
