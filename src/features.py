from __future__ import annotations

from math import isfinite
from typing import Iterator

import pandas as pd
from rapidfuzz import fuzz

from .io_utils import write_tsv
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


def pair_features_normalized(
    s1: pd.Series,
    candidate: pd.Series,
) -> dict[str, float | int]:
    """
    Compute the same baseline pairwise features (B1-B6) as `pair_features`,
    but from PRE-NORMALIZED columns instead of re-normalizing raw text.

    This is the production-scale counterpart to `pair_features`. It exists
    because `pair_features` re-runs `normalize_name`/`normalize_address`
    (character-by-character Unicode normalization) on every single call,
    which is wasteful once a normalized data root already exists with
    `name_norm`/`name_nosuffix`/`address_norm`/`country_norm` precomputed
    for every row (see `src.normalization.add_normalized_columns`) -- at
    millions of candidate pairs, repeating that work per-pair is a real
    performance problem, not a style preference.

    `pair_features` is left completely unchanged (raw fields in, internal
    normalization) so its existing behavior and tests keep working; this
    function is an additive, clearly-named alternative entry point for the
    normalized-data production path, not a replacement.

    Args:
        s1: Series for a Source-1 row. Must carry `name_norm`, `address_norm`,
            `country_norm` (as produced by `add_normalized_columns`) AND the
            original raw `business_address` column (also preserved by
            `add_normalized_columns`) -- the raw address is used only to
            decide the B6 missingness flags (see "Known differences" below),
            never for similarity scoring.
        candidate: Same shape as `s1`, for the S2/S3 candidate row.

    Returns:
        The exact same dict shape as `pair_features`: name_ratio,
        name_token_set_ratio, address_ratio, address_token_set_ratio,
        country_match, s1_address_missing, candidate_address_missing,
        both_address_missing.

    Equivalence to `pair_features` (raw path):
        - name_ratio / name_token_set_ratio / address_ratio /
          address_token_set_ratio are IDENTICAL to `pair_features`, because
          `name_norm` == `normalize_name(business_name)` and `address_norm`
          == `normalize_address(business_address)` by construction (the same
          functions build both). `name_nosuffix` is intentionally NOT used
          here: the current B1-B6 baseline does not strip legal suffixes
          (see `pair_features`), and using `name_nosuffix` instead would
          silently change feature semantics as well as add an experimental
          feature (out of scope here).
        - s1_address_missing / candidate_address_missing /
          both_address_missing are computed from the RAW `business_address`
          field (present alongside the norm columns on a normalized
          dataframe), exactly like `pair_features`. Deriving "missing" from
          `address_norm == ""` instead would NOT be identical in a rare edge
          case: a raw address that is entirely punctuation/symbols (e.g.
          "!!!") normalizes to "" even though it was not actually missing.
        - country_match: KNOWN, DOCUMENTED DIVERGENCE. `pair_features` uses
          `_normalize_country` (`str.strip().casefold()`, no alias table).
          `country_norm` is built by `src.normalization.normalize_country`,
          which additionally resolves `COUNTRY_ALIASES` (e.g. "usa"/"us"/
          "united states" -> "US", "india"/"in" -> "India"). The two paths
          agree for the overwhelming majority of real values (anything not
          hitting an alias), but can disagree when one side of a pair uses
          an aliased spelling not matched verbatim by the other side (e.g.
          s1 country="USA", candidate country="US": the raw path scores no
          match, this function scores a match, because country_norm already
          collapsed both to "US"). This is a pre-existing inconsistency
          between `src/features.py` and `src/normalization.py` (there is
          even a THIRD, also-different country normalizer in
          `src/blocking.py::add_normalized_keys`), not something introduced
          here. Per the "do not blindly rewrite `pair_features`" instruction
          it is left as-is and documented rather than silently patched; see
          `tests/test_normalized_features.py` for a test that exercises this
          exact boundary explicitly instead of hiding it.
    """
    raw_addr1 = s1.get("business_address")
    raw_addr2 = candidate.get("business_address")
    s1_address_empty = _is_empty(raw_addr1)
    candidate_address_empty = _is_empty(raw_addr2)

    def _to_string(val):
        if val is None or pd.isna(val):
            return ""
        return val

    name1 = _to_string(s1.get("name_norm"))
    name2 = _to_string(candidate.get("name_norm"))

    addr1 = _to_string(s1.get("address_norm"))
    addr2 = _to_string(candidate.get("address_norm"))

    country1 = _to_string(s1.get("country_norm"))
    country2 = _to_string(candidate.get("country_norm"))
    # Defensive casefold only -- country_norm values are a mix of
    # alias-resolved display case ("US", "India") and plain-casefolded
    # values ("france"); casefolding again here does not change already
    # lowercase values and makes the comparison case-insensitive without
    # re-deriving the alias table. See the docstring above for why this can
    # still diverge from `pair_features`'s country_match on aliased inputs.
    country1 = str(country1).strip().casefold()
    country2 = str(country2).strip().casefold()

    name_ratio = _safe_ratio(name1, name2)
    name_token_set_ratio = _safe_token_set_ratio(name1, name2)
    address_ratio = _safe_ratio(addr1, addr2)
    address_token_set_ratio = _safe_token_set_ratio(addr1, addr2)
    country_match = float(country1 == country2)

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


def build_pair_feature_frame_normalized(pairs: list[tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """
    Same as `build_pair_feature_frame`, but using `pair_features_normalized`
    (pre-normalized columns) instead of `pair_features` (raw columns).

    Kept as a small, in-memory, list-of-tuples function -- like
    `build_pair_feature_frame` -- for API symmetry and unit testing at small
    scale. For real (millions-of-pairs) production use, prefer
    `iter_pair_feature_frames_normalized` / `build_pair_feature_frame_normalized_chunked`
    below, which never materialize a Python list of Series for the entire
    candidate set at once.
    """
    rows = []
    for left, right in pairs:
        row = {
            "source1_entity_id": left["entity_id"],
            "candidate_entity_id": right["entity_id"],
        }
        row.update(pair_features_normalized(left, right))
        rows.append(row)
    return pd.DataFrame(rows)


def iter_pair_feature_frames_normalized(
    pairs_df: pd.DataFrame,
    s1_lookup: pd.DataFrame,
    candidate_lookup: pd.DataFrame,
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """
    Yield pre-normalized pairwise features one bounded chunk at a time.

    This is the scale-safe replacement for the pattern of building one giant
    Python list of `(s1_row, candidate_row)` Series tuples for the ENTIRE
    candidate-pair set before computing features (that pattern -- millions of
    `.loc[...]` row lookups held in memory simultaneously -- is exactly the
    "giant Python list of pandas Series" scale hazard this module's tests and
    the project's performance requirements call out). Peak memory here is
    bounded by `chunksize`, not by the total number of candidate pairs.

    Args:
        pairs_df: Long-format candidate pairs (`s1_col`, `candidate_col`),
            e.g. the output of `src.labeling.explode_candidate_pairs`. Can
            safely have millions of rows -- only `chunksize` of them are
            materialized as row-lookups at a time.
        s1_lookup: Source-1 rows indexed by `entity_id` (as a column), already
            carrying `name_norm`/`address_norm`/`country_norm`/raw
            `business_address` -- e.g. Source-1 loaded from the normalized
            data root. Source-1 is bounded in size, so holding it (and an
            `entity_id`-indexed view of it) fully in memory is fine.
        candidate_lookup: Candidate (S2/S3) rows, same column shape as
            `s1_lookup`. Expected to already be restricted to the (much
            smaller) set of entity IDs actually referenced as candidates --
            never the full multi-million-row Source-2/Source-3 tables.
        chunksize: Number of candidate PAIRS processed per yielded chunk.

    Yields:
        One compact, all-numeric feature DataFrame per chunk (plus the two
        ID columns). Rows whose S1 or candidate ID is not found in the
        supplied lookups are skipped (mirrors the `if s1_id in ... and
        cand_id in ...` filtering already used in the notebook), rather than
        raising, since a candidate lookup that was itself built by streaming
        Source-2/3 could legitimately miss an ID if the data is incomplete.
    """
    if s1_lookup.index.name != "entity_id" or not s1_lookup.index.is_unique:
        s1_lookup = s1_lookup.set_index("entity_id", drop=False)
    if candidate_lookup.index.name != "entity_id" or not candidate_lookup.index.is_unique:
        candidate_lookup = candidate_lookup.set_index("entity_id", drop=False)

    n = len(pairs_df)
    for start in range(0, n, chunksize):
        chunk = pairs_df.iloc[start:start + chunksize]
        pairs_for_chunk = [
            (s1_lookup.loc[s1_id], candidate_lookup.loc[cand_id])
            for s1_id, cand_id in zip(chunk[s1_col], chunk[candidate_col], strict=False)
            if s1_id in s1_lookup.index and cand_id in candidate_lookup.index
        ]
        yield build_pair_feature_frame_normalized(pairs_for_chunk)


def build_pair_feature_frame_normalized_chunked(
    pairs_df: pd.DataFrame,
    s1_lookup: pd.DataFrame,
    candidate_lookup: pd.DataFrame,
    *,
    s1_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    chunksize: int = 200_000,
    persist_path: str | None = None,
) -> pd.DataFrame | str:
    """
    Drive `iter_pair_feature_frames_normalized` to completion.

    Two modes:
      - `persist_path=None` (default): accumulate each (compact, all-float)
        chunk and concatenate once at the end, returning a DataFrame. This is
        safe even at millions of pairs because each chunk is already reduced
        to numeric feature columns (a few floats/ints per pair) rather than
        Series objects or dicts -- the danger this module guards against is
        holding ROW OBJECTS for every pair at once, not holding the final
        small numeric feature table.
      - `persist_path=<path>`: never hold more than one chunk in memory at
        once. Each chunk is appended to `persist_path` (TSV) as soon as it is
        computed and then dropped. Returns the path written. Use this if a
        candidate set is large enough that even the final numeric feature
        table would not comfortably fit in RAM.

    Returns:
        A DataFrame (in-memory mode) or the persisted file path (string).
    """
    if persist_path is None:
        chunks = list(
            iter_pair_feature_frames_normalized(
                pairs_df,
                s1_lookup,
                candidate_lookup,
                s1_col=s1_col,
                candidate_col=candidate_col,
                chunksize=chunksize,
            )
        )
        non_empty = [c for c in chunks if len(c)]
        if not non_empty:
            return pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                    "name_ratio",
                    "name_token_set_ratio",
                    "address_ratio",
                    "address_token_set_ratio",
                    "country_match",
                    "s1_address_missing",
                    "candidate_address_missing",
                    "both_address_missing",
                ]
            )
        return pd.concat(non_empty, ignore_index=True)

    from pathlib import Path

    out_path = Path(persist_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    wrote_header = False
    for chunk in iter_pair_feature_frames_normalized(
        pairs_df,
        s1_lookup,
        candidate_lookup,
        s1_col=s1_col,
        candidate_col=candidate_col,
        chunksize=chunksize,
    ):
        if chunk.empty:
            continue
        chunk.to_csv(
            out_path,
            sep="\t",
            index=False,
            encoding="utf-8",
            mode="a",
            header=not wrote_header,
        )
        wrote_header = True

    if not wrote_header:
        # Nothing was ever written (no pairs resolved) -- still produce an
        # empty-but-correctly-headed file rather than leaving no file at all.
        write_tsv(
            pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                    "name_ratio",
                    "name_token_set_ratio",
                    "address_ratio",
                    "address_token_set_ratio",
                    "country_match",
                    "s1_address_missing",
                    "candidate_address_missing",
                    "both_address_missing",
                ]
            ),
            out_path,
        )

    return str(out_path)