"""
Unit tests for baseline pairwise features (B1-B6).

Tests cover:
- Both addresses missing
- One address missing
- Both addresses present
- Identical names
- Reordered names
- Identical countries
- Different countries
- Edge cases (None, empty strings, whitespace)
"""

import pandas as pd
import pytest

from src.features import (
    _is_empty,
    _normalize_country,
    _safe_ratio,
    _safe_token_set_ratio,
    pair_features,
    build_pair_feature_frame,
)
from src.normalization import normalize_address, normalize_name


class TestIsEmpty:
    """Test the _is_empty helper function."""

    def test_none_is_empty(self):
        assert _is_empty(None) is True

    def test_empty_string_is_empty(self):
        assert _is_empty("") is True

    def test_whitespace_is_empty(self):
        assert _is_empty("   ") is True
        assert _is_empty("\t") is True
        assert _is_empty("\n") is True

    def test_non_empty_string(self):
        assert _is_empty("hello") is False
        assert _is_empty(" hello ") is False
        assert _is_empty("0") is False


class TestNormalizeCountry:
    """Test the _normalize_country helper function."""

    def test_none_becomes_empty(self):
        assert _normalize_country(None) == ""

    def test_empty_string(self):
        assert _normalize_country("") == ""

    def test_uppercase_converted(self):
        assert _normalize_country("US") == "us"

    def test_whitespace_stripped(self):
        assert _normalize_country("  India  ") == "india"

    def test_combined(self):
        assert _normalize_country("  FRANCE  ") == "france"


class TestSafeRatio:
    """Test the _safe_ratio string similarity function."""

    def test_both_empty_returns_zero(self):
        """Both addresses empty should return 0.0, not 1.0."""
        assert _safe_ratio("", "") == 0.0

    def test_one_empty_returns_zero(self):
        """One address empty should return 0.0."""
        assert _safe_ratio("hello", "") == 0.0
        assert _safe_ratio("", "hello") == 0.0

    def test_both_none_returns_zero(self):
        """Both None should return 0.0."""
        assert _safe_ratio(None, None) == 0.0

    def test_one_none_returns_zero(self):
        """One None should return 0.0."""
        assert _safe_ratio("hello", None) == 0.0
        assert _safe_ratio(None, "hello") == 0.0

    def test_identical_strings(self):
        """Identical strings should have ratio close to 1.0."""
        result = _safe_ratio("hello", "hello")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_very_similar_strings(self):
        """Very similar strings should have high ratio."""
        result = _safe_ratio("hello", "helo")
        assert 0.7 < result < 1.0

    def test_different_strings(self):
        """Very different strings should have low ratio."""
        result = _safe_ratio("hello", "world")
        assert 0.0 < result < 0.5

    def test_whitespace_only(self):
        """Whitespace-only strings should be treated as empty."""
        assert _safe_ratio("   ", "   ") == 0.0
        assert _safe_ratio("hello", "   ") == 0.0


class TestSafeTokenSetRatio:
    """Test the _safe_token_set_ratio function."""

    def test_both_empty_returns_zero(self):
        """Both addresses empty should return 0.0, not 1.0."""
        assert _safe_token_set_ratio("", "") == 0.0

    def test_one_empty_returns_zero(self):
        """One address empty should return 0.0."""
        assert _safe_token_set_ratio("hello world", "") == 0.0
        assert _safe_token_set_ratio("", "hello world") == 0.0

    def test_identical_strings(self):
        """Identical strings should have ratio = 1.0."""
        result = _safe_token_set_ratio("hello world", "hello world")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_reordered_tokens(self):
        """Reordered tokens should have high ratio (token_set handles order)."""
        result = _safe_token_set_ratio("hello world", "world hello")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_extra_token(self):
        """Extra token should still have high ratio (token_set asymmetric)."""
        result = _safe_token_set_ratio("hello world foo", "hello world")
        assert 0.8 < result <= 1.0

    def test_missing_token(self):
        """Missing token should still have high ratio."""
        result = _safe_token_set_ratio("hello world", "hello world foo")
        assert 0.8 < result <= 1.0

    def test_completely_different(self):
        """Completely different tokens should have low ratio."""
        result = _safe_token_set_ratio("hello world", "alpha bravo")
        assert 0.0 <= result < 0.5


class TestPairFeaturesAddressMissing:
    """Test address missingness scenarios (B6a, B6b, B6c)."""

    def test_both_addresses_missing(self):
        """When both addresses are missing/NaN, all address features should be 0.0 and both_address_missing=1."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": None,
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["address_ratio"] == 0.0
        assert features["address_token_set_ratio"] == 0.0
        assert features["s1_address_missing"] == 1
        assert features["candidate_address_missing"] == 1
        assert features["both_address_missing"] == 1

    def test_s1_address_missing_only(self):
        """When S1 address is missing but candidate has address."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": None,
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["address_ratio"] == 0.0
        assert features["address_token_set_ratio"] == 0.0
        assert features["s1_address_missing"] == 1
        assert features["candidate_address_missing"] == 0
        assert features["both_address_missing"] == 0

    def test_candidate_address_missing_only(self):
        """When candidate address is missing but S1 has address."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "   ",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["address_ratio"] == 0.0
        assert features["address_token_set_ratio"] == 0.0
        assert features["s1_address_missing"] == 0
        assert features["candidate_address_missing"] == 1
        assert features["both_address_missing"] == 0

    def test_both_addresses_present(self):
        """When both addresses are present, missingness flags should all be 0."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["s1_address_missing"] == 0
        assert features["candidate_address_missing"] == 0
        assert features["both_address_missing"] == 0
        # Address similarity should not be 0.0
        assert features["address_ratio"] > 0.0
        assert features["address_token_set_ratio"] > 0.0


class TestPairFeaturesNameSimilarity:
    """Test name similarity features (B1, B2)."""

    def test_identical_names(self):
        """Identical names should have high similarity scores."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corporation",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corporation",
            "business_address": "123 Main St",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["name_ratio"] > 0.9
        assert features["name_token_set_ratio"] > 0.9

    def test_reordered_names(self):
        """Reordered name tokens should have high token_set_ratio."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "Hendricks and Flowers Inc",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "Hendricks and Inc Flowers",  # reordered
            "business_address": "123 Main St",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        # token_set_ratio should handle reordering
        assert features["name_token_set_ratio"] > 0.8
        # character ratio might be lower
        assert features["name_ratio"] < features["name_token_set_ratio"]

    def test_name_with_extra_token(self):
        """Names differing by one token should have moderate-to-high token_set_ratio."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "Dahlia Power Reliable Scientific LLC",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "Dahlia Power Reliable",
            "business_address": "123 Main St",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        # token_set_ratio handles asymmetric overlap
        assert features["name_token_set_ratio"] > 0.7

    def test_different_names(self):
        """Completely different names should have low similarity."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corporation",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "XYZ Enterprises",
            "business_address": "456 Oak Ave",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["name_ratio"] < 0.6
        assert features["name_token_set_ratio"] < 0.6

    def test_name_with_typo(self):
        """Small typos should still have high name_ratio."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "Maure Williams Colombier",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "Maure Wilblims Colombier",  # typo in "Williams"
            "business_address": "123 Main St",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        # Should capture character-level similarity even with typo
        assert features["name_ratio"] > 0.75


class TestPairFeaturesAddressSimilarity:
    """Test address similarity features (B3, B4)."""

    def test_identical_addresses(self):
        """Identical addresses should have high similarity scores."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "630 45th Terrace, Kansas City, MO",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "630 45th Terrace, Kansas City, MO",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["address_ratio"] > 0.9
        assert features["address_token_set_ratio"] > 0.9

    def test_reordered_address_components(self):
        """Reordered address components should have high token_set_ratio."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "630 45th Terrace, Kansas City, MO",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "Kansas City, MO, 630 45th Terrace",  # reordered
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        # token_set_ratio should handle reordering
        assert features["address_token_set_ratio"] > 0.8

    def test_address_with_missing_component(self):
        """Missing address component should have moderate-to-high token_set_ratio."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "584 Summer Street, Unit 1441, Holyoke, MA",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "Summer Street, Holyoke, MA",  # missing street number and unit
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        # token_set_ratio handles partial overlap
        assert features["address_token_set_ratio"] > 0.6

    def test_different_addresses(self):
        """Completely different addresses should have low similarity."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St, Springfield, IL",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "456 Oak Ave, Denver, CO",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["address_ratio"] < 0.5
        assert features["address_token_set_ratio"] < 0.5

    def test_address_abbreviation_handling(self):
        """Address abbreviations (St vs Street, Rd vs Road) should normalize and match."""
        # normalization.py handles "Street" -> "st", "Road" -> "rd", etc.
        # After normalization, these should be identical
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "3315 Fremont Street, Peoria, IL",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "3315 Fremont St, Peoria, IL",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        # After normalization in normalize_address, should be very similar
        assert features["address_ratio"] > 0.9


class TestPairFeaturesCountry:
    """Test country_match feature (B5)."""

    def test_identical_countries(self):
        """Identical countries should have country_match = 1.0."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["country_match"] == 1.0

    def test_different_countries(self):
        """Different countries should have country_match = 0.0."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "India",
        })
        features = pair_features(s1, candidate)
        
        assert features["country_match"] == 0.0

    def test_unseen_country_france(self):
        """Unseen country (France, not in training) should handle gracefully."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "France",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "France",
        })
        features = pair_features(s1, candidate)
        
        assert features["country_match"] == 1.0

    def test_case_insensitive_country(self):
        """Country comparison should be case-insensitive."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "us",
        })
        features = pair_features(s1, candidate)
        
        assert features["country_match"] == 1.0

    def test_whitespace_country(self):
        """Whitespace in country should be stripped."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "  US  ",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        features = pair_features(s1, candidate)
        
        assert features["country_match"] == 1.0

    def test_none_country(self):
        """None country should be handled gracefully."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": None,
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": None,
        })
        features = pair_features(s1, candidate)
        
        # Both None should match (both normalize to "")
        assert features["country_match"] == 1.0


class TestBuildPairFeatureFrame:
    """Test the build_pair_feature_frame function."""

    def test_single_pair(self):
        """Test building features for a single pair."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        
        pairs = [(s1, candidate)]
        df = build_pair_feature_frame(pairs)
        
        assert len(df) == 1
        assert df.loc[0, "source1_entity_id"] == "S1-001"
        assert df.loc[0, "candidate_entity_id"] == "S2-001"
        assert "name_ratio" in df.columns
        assert "country_match" in df.columns

    def test_multiple_pairs(self):
        """Test building features for multiple pairs."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate1 = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate2 = pd.Series({
            "entity_id": "S2-002",
            "business_name": "ABC Services",
            "business_address": "456 Oak Ave",
            "country": "US",
        })
        
        pairs = [(s1, candidate1), (s1, candidate2)]
        df = build_pair_feature_frame(pairs)
        
        assert len(df) == 2
        assert list(df["source1_entity_id"]) == ["S1-001", "S1-001"]
        assert list(df["candidate_entity_id"]) == ["S2-001", "S2-002"]

    def test_feature_columns_present(self):
        """Test that all baseline features are in the output DataFrame."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        
        pairs = [(s1, candidate)]
        df = build_pair_feature_frame(pairs)
        
        expected_columns = [
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
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_feature_value_ranges(self):
        """Test that feature values are in expected ranges."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        
        pairs = [(s1, candidate)]
        df = build_pair_feature_frame(pairs)
        
        # Similarity features should be [0.0, 1.0]
        for col in ["name_ratio", "name_token_set_ratio", "address_ratio", "address_token_set_ratio"]:
            assert 0.0 <= df.loc[0, col] <= 1.0, f"{col} out of range"
        
        # country_match should be 0.0 or 1.0
        assert df.loc[0, "country_match"] in [0.0, 1.0]
        
        # Missingness indicators should be 0 or 1
        for col in ["s1_address_missing", "candidate_address_missing", "both_address_missing"]:
            assert df.loc[0, col] in [0, 1], f"{col} should be 0 or 1"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_missing_keys_in_series(self):
        """Test handling of missing keys in input Series."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": "ABC Corp",
            # business_address missing
            "country": "US",
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            # country missing
        })
        
        # Should not raise; uses .get() with default ""
        features = pair_features(s1, candidate)
        
        assert "s1_address_missing" in features
        assert features["s1_address_missing"] == 1  # missing from Series
        assert features["country_match"] == 0.0  # one "" == ""? No, both "" == ""? Actually, depends

    def test_all_fields_empty(self):
        """Test when all fields are empty/None."""
        s1 = pd.Series({
            "entity_id": "S1-001",
            "business_name": None,
            "business_address": None,
            "country": None,
        })
        candidate = pd.Series({
            "entity_id": "S2-001",
            "business_name": "",
            "business_address": "   ",
            "country": "",
        })
        
        features = pair_features(s1, candidate)
        
        assert features["name_ratio"] == 0.0
        assert features["name_token_set_ratio"] == 0.0
        assert features["address_ratio"] == 0.0
        assert features["address_token_set_ratio"] == 0.0
        assert features["country_match"] == 1.0  # both empty "" == ""
        assert features["s1_address_missing"] == 1
        assert features["candidate_address_missing"] == 1
        assert features["both_address_missing"] == 1