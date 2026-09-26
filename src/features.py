from __future__ import annotations

from math import isfinite

import pandas as pd
from rapidfuzz import fuzz

from .normalization import normalize_address, normalize_name, token_set


def _safe_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    value = fuzz.ratio(a, b) / 100.0
    return value if isfinite(value) else 0.0


def pair_features(
    s1: pd.Series,
    candidate: pd.Series,
) -> dict[str, float | str]:
    name1 = normalize_name(s1.get("business_name", ""))
    name2 = normalize_name(candidate.get("business_name", ""))
    name1_ns = normalize_name(s1.get("business_name", ""), strip_legal_suffixes=True)
    name2_ns = normalize_name(candidate.get("business_name", ""), strip_legal_suffixes=True)

    addr1 = normalize_address(s1.get("business_address", ""))
    addr2 = normalize_address(candidate.get("business_address", ""))

    name_tokens_1 = token_set(name1)
    name_tokens_2 = token_set(name2)
    addr_tokens_1 = token_set(addr1)
    addr_tokens_2 = token_set(addr2)

    name_union = len(name_tokens_1 | name_tokens_2)
    addr_union = len(addr_tokens_1 | addr_tokens_2)

    return {
        "name_ratio": _safe_ratio(name1, name2),
        "name_ratio_nosuffix": _safe_ratio(name1_ns, name2_ns),
        "name_token_jaccard": (
            len(name_tokens_1 & name_tokens_2) / name_union if name_union else 1.0
        ),
        "name_token_set_ratio": fuzz.token_set_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0,
        "address_ratio": _safe_ratio(addr1, addr2),
        "address_token_jaccard": (
            len(addr_tokens_1 & addr_tokens_2) / addr_union if addr_union else 1.0
        ),
        "address_token_set_ratio": fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0,
        "country_match": float(
            str(s1.get("country", "")).strip().casefold()
            == str(candidate.get("country", "")).strip().casefold()
        ),
        "name_len_abs_diff": float(abs(len(name1) - len(name2))),
        "address_len_abs_diff": float(abs(len(addr1) - len(addr2))),
        "source_is_s2": float(str(candidate.get("entity_id", "")).startswith("S2-")),
        "source_is_s3": float(str(candidate.get("entity_id", "")).startswith("S3-")),
    }


def build_pair_feature_frame(pairs: list[tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    rows = []
    for left, right in pairs:
        row = {
            "source1_entity_id": left["entity_id"],
            "candidate_entity_id": right["entity_id"],
        }
        row.update(pair_features(left, right))
        rows.append(row)
    return pd.DataFrame(rows)
