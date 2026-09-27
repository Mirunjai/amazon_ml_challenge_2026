"""
Unit tests for the pre-normalized pairwise feature path
(`pair_features_normalized`, `build_pair_feature_frame_normalized`, and the
chunked production helpers) in `src/features.py`.

The central claim under test: the pre-normalized path produces IDENTICAL
feature values to the existing raw path (`pair_features`) on representative
(s1, candidate) pairs, because `name_norm`/`address_norm` are constructed by
the exact same `normalize_name`/`normalize_address` functions the raw path
calls internally. The one documented exception is `country_match` on inputs
that trigger `COUNTRY_ALIASES` resolution -- that boundary is tested
explicitly, not hidden.
"""

import pandas as pd
import pytest

from src.features import (
    build_pair_feature_frame,
    build_pair_feature_frame_normalized,
    build_pair_feature_frame_normalized_chunked,
    iter_pair_feature_frames_normalized,
    pair_features,
    pair_features_normalized,
)
from src.normalization import add_normalized_columns

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _raw_row(entity_id, name, address, country):
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    }


# Representative (s1, candidate) pairs: realistic business-name/address/
# country variation, deliberately AVOIDING country values that would trigger
# COUNTRY_ALIASES resolution (that boundary gets its own dedicated test).
REPRESENTATIVE_PAIRS_RAW = [
    (
        _raw_row("S1-1", "Acme Corp Pvt. Ltd.", "12 Market Road, Springfield", "India"),
        _raw_row("S2-1", "ACME CORPORATION", "12 Market Rd, Springfield", "india"),
    ),
    (
        # Missing S1 address.
        _raw_row("S1-2", "Global Traders, LLC", None, "France"),
        _raw_row("S2-2", "Global Traders LLC", "99 Unknown Ave", "France"),
    ),
    (
        # Unicode business name + address, candidate address abbreviation.
        _raw_row("S1-3", "Café Münster GmbH", "5 Bahnhof Straße", "Germany"),
        _raw_row("S2-3", "Cafe Munster GmbH", "5 Bahnhof St", "Germany"),
    ),
    (
        # Both addresses missing.
        _raw_row("S1-4", "Nile Traders", None, "Egypt"),
        _raw_row("S2-4", "Totally Unrelated Co", None, "Egypt"),
    ),
    (
        # Completely different business, different (non-aliased) country.
        _raw_row("S1-5", "Blue Sky Aviation", "1 Runway Way", "France"),
        _raw_row("S2-5", "Red Ocean Fisheries", "2 Harbor Rd", "Japan"),
    ),
    (
        # Reordered name tokens; punctuation-only-ish address fragment.
        _raw_row("S1-6", "Traders Global United", "Suite 4, Tower B", "Japan"),
        _raw_row("S2-6", "United Global Traders", "Tower B, Suite 4", "Japan"),
    ),
    (
        # Country values already share the exact same casing/spelling.
        _raw_row("S1-7", "Sunrise Logistics", "88 Harbor Blvd", "France"),
        _raw_row("S2-7", "Sunrise Logistic", "88 Harbour Blvd", "france"),
    ),
]


def _to_frames(pairs_raw):
    s1_df = pd.DataFrame([p[0] for p in pairs_raw])[RAW_COLUMNS]
    cand_df = pd.DataFrame([p[1] for p in pairs_raw])[RAW_COLUMNS]
    return s1_df, cand_df


class TestIdenticalToRawPathOnRepresentativePairs:
    """The core equivalence guarantee the task requires."""

    def test_each_pair_identical_dict(self):
        s1_df, cand_df = _to_frames(REPRESENTATIVE_PAIRS_RAW)
        s1_norm = add_normalized_columns(s1_df)
        cand_norm = add_normalized_columns(cand_df)

        for i in range(len(REPRESENTATIVE_PAIRS_RAW)):
            raw_result = pair_features(s1_df.iloc[i], cand_df.iloc[i])
            norm_result = pair_features_normalized(s1_norm.iloc[i], cand_norm.iloc[i])
            assert raw_result == norm_result, (
                f"Mismatch on representative pair {i}: "
                f"raw={raw_result} normalized={norm_result}"
            )

    def test_build_pair_feature_frame_identical(self):
        """Same equivalence, but through the DataFrame-building entry points."""
        s1_df, cand_df = _to_frames(REPRESENTATIVE_PAIRS_RAW)
        s1_norm = add_normalized_columns(s1_df)
        cand_norm = add_normalized_columns(cand_df)

        raw_pairs = [(s1_df.iloc[i], cand_df.iloc[i]) for i in range(len(s1_df))]
        norm_pairs = [(s1_norm.iloc[i], cand_norm.iloc[i]) for i in range(len(s1_norm))]

        raw_frame = build_pair_feature_frame(raw_pairs)
        norm_frame = build_pair_feature_frame_normalized(norm_pairs)

        pd.testing.assert_frame_equal(
            raw_frame.reset_index(drop=True), norm_frame.reset_index(drop=True)
        )

    def test_name_ratio_matches_because_name_norm_equals_internal_normalization(self):
        """name_norm is exactly normalize_name(business_name); spot check."""
        s1_df, cand_df = _to_frames(REPRESENTATIVE_PAIRS_RAW[:1])
        s1_norm = add_normalized_columns(s1_df)
        cand_norm = add_normalized_columns(cand_df)

        raw_result = pair_features(s1_df.iloc[0], cand_df.iloc[0])
        norm_result = pair_features_normalized(s1_norm.iloc[0], cand_norm.iloc[0])
        assert raw_result["name_ratio"] == norm_result["name_ratio"]
        assert raw_result["name_token_set_ratio"] == norm_result["name_token_set_ratio"]


class TestAddressMissingnessUsesRawNotNormalizedEmptiness:
    """B6 flags must come from raw missingness, not from address_norm == ''."""

    def test_punctuation_only_address_is_not_missing(self):
        """A raw address of only punctuation normalizes to '' but is NOT
        'missing' -- both paths must agree it's present (not missing)."""
        s1_raw_df = pd.DataFrame([_raw_row("S1-1", "Acme", "!!!", "France")])[RAW_COLUMNS]
        cand_raw_df = pd.DataFrame([_raw_row("S2-1", "Acme", "10 Main St", "France")])[
            RAW_COLUMNS
        ]
        s1_norm = add_normalized_columns(s1_raw_df)
        cand_norm = add_normalized_columns(cand_raw_df)

        # Confirm the premise: normalize_address("!!!") really is "".
        assert s1_norm.iloc[0]["address_norm"] == ""

        raw_result = pair_features(s1_raw_df.iloc[0], cand_raw_df.iloc[0])
        norm_result = pair_features_normalized(s1_norm.iloc[0], cand_norm.iloc[0])

        assert raw_result["s1_address_missing"] == 0
        assert norm_result["s1_address_missing"] == 0
        assert raw_result == norm_result

    def test_none_address_is_missing_in_both_paths(self):
        s1_raw_df = pd.DataFrame([_raw_row("S1-1", "Acme", None, "France")])[RAW_COLUMNS]
        cand_raw_df = pd.DataFrame([_raw_row("S2-1", "Acme", None, "France")])[RAW_COLUMNS]
        s1_norm = add_normalized_columns(s1_raw_df)
        cand_norm = add_normalized_columns(cand_raw_df)

        raw_result = pair_features(s1_raw_df.iloc[0], cand_raw_df.iloc[0])
        norm_result = pair_features_normalized(s1_norm.iloc[0], cand_norm.iloc[0])

        assert raw_result["both_address_missing"] == 1
        assert norm_result["both_address_missing"] == 1
        assert raw_result == norm_result


class TestDocumentedCountryAliasDivergence:
    """
    KNOWN, DOCUMENTED divergence: pair_features' _normalize_country (plain
    casefold, no alias table) vs country_norm (which resolves
    COUNTRY_ALIASES). This test makes the boundary explicit and verifiable
    rather than letting it silently pass or silently break.
    """

    def test_aliased_country_spelling_diverges_from_raw_path(self):
        s1_raw_df = pd.DataFrame([_raw_row("S1-1", "X", "a", "USA")])[RAW_COLUMNS]
        cand_raw_df = pd.DataFrame([_raw_row("S2-1", "X", "a", "US")])[RAW_COLUMNS]
        s1_norm = add_normalized_columns(s1_raw_df)
        cand_norm = add_normalized_columns(cand_raw_df)

        raw_result = pair_features(s1_raw_df.iloc[0], cand_raw_df.iloc[0])
        norm_result = pair_features_normalized(s1_norm.iloc[0], cand_norm.iloc[0])

        # Raw path: "usa" != "us" -> no match.
        assert raw_result["country_match"] == 0.0
        # Pre-normalized path: country_norm alias-resolves both to "US" -> match.
        assert norm_result["country_match"] == 1.0
        # Everything else about the two dicts still agrees.
        assert {k: v for k, v in raw_result.items() if k != "country_match"} == {
            k: v for k, v in norm_result.items() if k != "country_match"
        }

    def test_non_aliased_country_values_agree(self):
        """Countries with no alias table entry are unaffected by the divergence."""
        s1_raw_df = pd.DataFrame([_raw_row("S1-1", "X", "a", "Germany")])[RAW_COLUMNS]
        cand_raw_df = pd.DataFrame([_raw_row("S2-1", "X", "a", "GERMANY")])[RAW_COLUMNS]
        s1_norm = add_normalized_columns(s1_raw_df)
        cand_norm = add_normalized_columns(cand_raw_df)

        raw_result = pair_features(s1_raw_df.iloc[0], cand_raw_df.iloc[0])
        norm_result = pair_features_normalized(s1_norm.iloc[0], cand_norm.iloc[0])
        assert raw_result == norm_result
        assert raw_result["country_match"] == 1.0


class TestChunkedProductionPath:
    """Chunked generator/driver used for real (millions-of-pairs) scale."""

    def _lookups(self):
        s1_df, cand_df = _to_frames(REPRESENTATIVE_PAIRS_RAW)
        s1_norm = add_normalized_columns(s1_df).set_index("entity_id", drop=False)
        cand_norm = add_normalized_columns(cand_df).set_index("entity_id", drop=False)
        pairs_df = pd.DataFrame(
            {
                "source1_entity_id": s1_norm["entity_id"].to_numpy(),
                "candidate_entity_id": cand_norm["entity_id"].to_numpy(),
            }
        )
        return pairs_df, s1_norm, cand_norm

    def test_chunked_matches_unchunked_for_various_chunksizes(self):
        pairs_df, s1_norm, cand_norm = self._lookups()

        unchunked_pairs = [
            (s1_norm.loc[s1_id], cand_norm.loc[cand_id])
            for s1_id, cand_id in zip(
                pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]
            )
        ]
        expected = build_pair_feature_frame_normalized(unchunked_pairs)

        for chunksize in (1, 2, 3, 1000):
            chunked = build_pair_feature_frame_normalized_chunked(
                pairs_df, s1_norm, cand_norm, chunksize=chunksize
            )
            pd.testing.assert_frame_equal(
                chunked.reset_index(drop=True), expected.reset_index(drop=True)
            )

    def test_iterator_yields_multiple_chunks_and_bounds_chunk_size(self):
        pairs_df, s1_norm, cand_norm = self._lookups()
        chunks = list(
            iter_pair_feature_frames_normalized(
                pairs_df, s1_norm, cand_norm, chunksize=2
            )
        )
        assert len(chunks) == (len(pairs_df) + 1) // 2
        for chunk in chunks[:-1]:
            assert len(chunk) == 2
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == len(pairs_df)

    def test_unresolvable_ids_are_skipped_not_raised(self):
        pairs_df, s1_norm, cand_norm = self._lookups()
        pairs_df = pd.concat(
            [
                pairs_df,
                pd.DataFrame(
                    [{"source1_entity_id": "NOPE", "candidate_entity_id": "ALSO-NOPE"}]
                ),
            ],
            ignore_index=True,
        )
        result = build_pair_feature_frame_normalized_chunked(
            pairs_df, s1_norm, cand_norm, chunksize=3
        )
        assert len(result) == len(REPRESENTATIVE_PAIRS_RAW)
        assert "NOPE" not in set(result["source1_entity_id"])

    def test_empty_pairs_returns_empty_correctly_shaped_frame(self):
        _, s1_norm, cand_norm = self._lookups()
        empty_pairs = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        )
        result = build_pair_feature_frame_normalized_chunked(
            empty_pairs, s1_norm, cand_norm, chunksize=100
        )
        assert len(result) == 0
        assert "name_ratio" in result.columns

    def test_persist_path_writes_file_and_matches_in_memory_result(self, tmp_path):
        pairs_df, s1_norm, cand_norm = self._lookups()

        in_memory = build_pair_feature_frame_normalized_chunked(
            pairs_df, s1_norm, cand_norm, chunksize=2
        )

        out_file = tmp_path / "features" / "train_pair_features.tsv"
        returned_path = build_pair_feature_frame_normalized_chunked(
            pairs_df, s1_norm, cand_norm, chunksize=2, persist_path=str(out_file)
        )
        assert returned_path == str(out_file)
        assert out_file.exists()

        from_disk = pd.read_csv(out_file, sep="\t")
        pd.testing.assert_frame_equal(
            from_disk.reset_index(drop=True),
            in_memory.reset_index(drop=True),
            check_dtype=False,
        )

    def test_persist_path_with_no_resolvable_pairs_still_writes_headered_file(
        self, tmp_path
    ):
        _, s1_norm, cand_norm = self._lookups()
        unresolvable = pd.DataFrame(
            [{"source1_entity_id": "NOPE", "candidate_entity_id": "ALSO-NOPE"}]
        )
        out_file = tmp_path / "features" / "empty.tsv"
        returned_path = build_pair_feature_frame_normalized_chunked(
            unresolvable, s1_norm, cand_norm, chunksize=10, persist_path=str(out_file)
        )
        assert returned_path == str(out_file)
        assert out_file.exists()
        from_disk = pd.read_csv(out_file, sep="\t")
        assert len(from_disk) == 0
        assert "name_ratio" in from_disk.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])