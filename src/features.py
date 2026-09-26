from __future__ import annotations

from math import isfinite

import pandas as pd
from rapidfuzz import fuzz

from .normalization import normalize_address, normalize_name


def _is_empty(text: object) -> bool:
    """Check if text is None, pandas NaN, empty string, or whitespace-only."""
    if text is None:
        return True
    # Check for pandas NaN (which is a float nan)
    if pd.isna(text):
        return True
    text_str = str(text).strip()
    return len(text_str) == 0


def _safe_ratio(a: str, b: str) -> float:
    """
    Compute string similarity ratio between a and b.
    Returns 0.0 if either string is empty or missing.
    Returns 0.0 if both are empty (not 1.0).
    """
    if _is_empty(a) or _is_empty(b):
        return 0.0
    value = fuzz.ratio(a, b) / 100.0
    return value if isfinite(value) else 0.0


def _safe_token_set_ratio(a: str, b: str) -> float:
    """
    Compute token-set similarity ratio between a and b.
    Returns 0.0 if either string is empty or missing.
    Returns 0.0 if both are empty (not 1.0).
    """
    if _is_empty(a) or _is_empty(b):
        return 0.0
    value = fuzz.token_set_ratio(a, b) / 100.0
    return value if isfinite(value) else 0.0


def _normalize_country(country: object) -> str:
    """Normalize country value: strip, casefold, handle None/empty."""
    if country is None:
        return ""
    return str(country).strip().casefold()


def pair_features(
    s1: pd.Series,
    candidate: pd.Series,
) -> dict[str, float | int]:
    """
    Compute baseline pairwise features (B1-B6) for entity resolution matching.
    
    Args:
        s1: Series with entity_id, business_name, business_address, country from Source 1
        candidate: Series with entity_id, business_name, business_address, country from S2/S3
    
    Returns:
        Dictionary with baseline features:
        - name_ratio (B1)
        - name_token_set_ratio (B2)
        - address_ratio (B3)
        - address_token_set_ratio (B4)
        - country_match (B5)
        - s1_address_missing (B6a)
        - candidate_address_missing (B6b)
        - both_address_missing (B6c)
    """
    # Extract raw values and check for missingness BEFORE normalization
    raw_addr1 = s1.get("business_address")
    raw_addr2 = candidate.get("business_address")
    s1_address_empty = _is_empty(raw_addr1)
    candidate_address_empty = _is_empty(raw_addr2)
    
    # Helper to convert None/NaN to "" (note: "or" doesn't work with NaN because NaN is truthy)
    def _to_string(val):
        if val is None or pd.isna(val):
            return ""
        return val
    
    # Convert None/NaN to empty string for normalization
    raw_name1 = _to_string(s1.get("business_name"))
    raw_name2 = _to_string(candidate.get("business_name"))
    raw_addr1 = _to_string(raw_addr1)
    raw_addr2 = _to_string(raw_addr2)
    raw_country1 = _to_string(s1.get("country"))
    raw_country2 = _to_string(candidate.get("country"))
    
    # Normalize input text
    name1 = normalize_name(raw_name1)
    name2 = normalize_name(raw_name2)
    
    addr1 = normalize_address(raw_addr1)
    addr2 = normalize_address(raw_addr2)
    
    country1 = _normalize_country(raw_country1)
    country2 = _normalize_country(raw_country2)
    
    # B1: name_ratio — character-level string similarity
    name_ratio = _safe_ratio(name1, name2)
    
    # B2: name_token_set_ratio — token-based similarity (handles reordering)
    name_token_set_ratio = _safe_token_set_ratio(name1, name2)
    
    # B3: address_ratio — character-level string similarity
    address_ratio = _safe_ratio(addr1, addr2)
    
    # B4: address_token_set_ratio — token-based similarity (handles reordering)
    address_token_set_ratio = _safe_token_set_ratio(addr1, addr2)
    
    # B5: country_match — binary exact match
    country_match = float(country1 == country2)
    
    # B6a, B6b, B6c: address missingness indicators (checked BEFORE normalization)
    s1_address_missing = int(s1_address_empty)
    candidate_address_missing = int(candidate_address_empty)
    both_address_missing = int(s1_address_empty and candidate_address_empty)
    
    return {
        "name_ratio": name_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "address_ratio": address_ratio,
        "address_token_set_ratio": address_token_set_ratio,
        "country_match": country_match,
        "s1_address_missing": s1_address_missing,
        "candidate_address_missing": candidate_address_missing,
        "both_address_missing": both_address_missing,
    }


def build_pair_feature_frame(pairs: list[tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """
    Build a feature DataFrame from a list of (S1, candidate) pairs.
    
    Args:
        pairs: List of (s1_series, candidate_series) tuples
    
    Returns:
        DataFrame with columns:
        - source1_entity_id
        - candidate_entity_id
        - name_ratio
        - name_token_set_ratio
        - address_ratio
        - address_token_set_ratio
        - country_match
        - s1_address_missing
        - candidate_address_missing
        - both_address_missing
    """
    rows = []
    for left, right in pairs:
        row = {
            "source1_entity_id": left["entity_id"],
            "candidate_entity_id": right["entity_id"],
        }
        row.update(pair_features(left, right))
        rows.append(row)
    return pd.DataFrame(rows)