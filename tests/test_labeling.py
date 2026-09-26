import pandas as pd
import pytest

from src.labeling import (
    explode_candidate_pairs,
    group_pairs_by_s1,
    label_candidate_pairs,
)


class TestLabelCandidatePairsFromTuples:
    def test_multiple_matches_per_s1(self):
        """Matches the worked example from the project brief."""
        pairs = [
            ("S1_001", "S2_100"),
            ("S1_001", "S2_101"),
            ("S1_001", "S3_200"),
        ]
        ground_truth = {"S1_001": {"S2_100", "S3_200"}}

        labeled = label_candidate_pairs(pairs, ground_truth)

        expected = {
            ("S1_001", "S2_100"): 1,
            ("S1_001", "S2_101"): 0,
            ("S1_001", "S3_200"): 1,
        }
        actual = {
            (row.source1_entity_id, row.candidate_entity_id): row.label
            for row in labeled.itertuples()
        }
        assert actual == expected

    def test_zero_matches(self):
        pairs = [("S1_002", "S2_050"), ("S1_002", "S3_099")]
        ground_truth = {"S1_002": set()}

        labeled = label_candidate_pairs(pairs, ground_truth)

        assert set(labeled["label"]) == {0}
        assert len(labeled) == 2

    def test_s1_absent_from_ground_truth_defaults_to_no_match(self):
        """An S1 id with no ground-truth entry at all should label as 0, not error."""
        pairs = [("S1_999", "S2_001")]
        ground_truth: dict[str, set[str]] = {}

        labeled = label_candidate_pairs(pairs, ground_truth)

        assert labeled.loc[0, "label"] == 0

    def test_does_not_invent_pairs(self):
        """Only pairs that were supplied should appear in the output -- no
        cartesian product with the ground truth's other matches."""
        pairs = [("S1_001", "S2_100")]  # deliberately omit S3_200
        ground_truth = {"S1_001": {"S2_100", "S3_200"}}

        labeled = label_candidate_pairs(pairs, ground_truth)

        assert len(labeled) == 1
        assert set(labeled["candidate_entity_id"]) == {"S2_100"}

    def test_mixed_s2_and_s3_ids(self):
        pairs = [("S1_A", "S2_1"), ("S1_A", "S3_1"), ("S1_A", "S2_2")]
        ground_truth = {"S1_A": {"S3_1"}}

        labeled = label_candidate_pairs(pairs, ground_truth)
        by_id = dict(zip(labeled["candidate_entity_id"], labeled["label"], strict=False))

        assert by_id == {"S2_1": 0, "S3_1": 1, "S2_2": 0}


class TestLabelCandidatePairsFromDataFrame:
    def test_dataframe_input(self):
        pairs_df = pd.DataFrame(
            {
                "source1_entity_id": ["S1_001", "S1_001", "S1_002"],
                "candidate_entity_id": ["S2_100", "S2_101", "S3_050"],
            }
        )
        ground_truth = {"S1_001": {"S2_100"}, "S1_002": set()}

        labeled = label_candidate_pairs(pairs_df, ground_truth)

        assert list(labeled["label"]) == [1, 0, 0]

    def test_raw_ground_truth_dataframe_is_parsed(self):
        """Passing the raw GT DataFrame should reuse evaluation.parse_ground_truth."""
        pairs_df = pd.DataFrame(
            {
                "source1_entity_id": ["S1_1", "S1_1"],
                "candidate_entity_id": ["S2_1", "S3_2"],
            }
        )
        raw_gt = pd.DataFrame(
            {
                "source1_entity_id": ["S1_1"],
                "matched_entity_ids": ["S2_1,S3_2"],
            }
        )

        labeled = label_candidate_pairs(pairs_df, raw_gt)

        assert list(labeled["label"]) == [1, 1]

    def test_missing_required_column_raises(self):
        bad_df = pd.DataFrame({"source1_entity_id": ["S1_1"]})
        with pytest.raises(ValueError):
            label_candidate_pairs(bad_df, {})


class TestExplodeCandidatePairs:
    def test_basic_explode(self):
        wide = pd.DataFrame(
            {
                "source1_entity_id": ["S1-00001", "S1-00002", "S1-00003"],
                "candidate_entity_ids": [
                    "S2-00047,S2-00193,S3-00812",
                    "S3-00004",
                    "",
                ],
            }
        )

        long = explode_candidate_pairs(wide)

        assert list(long["source1_entity_id"]) == [
            "S1-00001",
            "S1-00001",
            "S1-00001",
            "S1-00002",
        ]
        assert list(long["candidate_entity_id"]) == [
            "S2-00047",
            "S2-00193",
            "S3-00812",
            "S3-00004",
        ]
        # The singleton (S1-00003, empty candidates) contributes no rows.
        assert "S1-00003" not in set(long["source1_entity_id"])

    def test_empty_and_nan_candidates_produce_no_rows(self):
        wide = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2"],
                "candidate_entity_ids": [None, "   "],
            }
        )
        long = explode_candidate_pairs(wide)
        assert len(long) == 0

    def test_missing_required_column_raises(self):
        bad = pd.DataFrame({"source1_entity_id": ["S1-1"]})
        with pytest.raises(ValueError):
            explode_candidate_pairs(bad)


class TestGroupPairsByS1:
    def test_groups_correctly(self):
        pairs = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
                "candidate_entity_id": ["S2-1", "S3-1", "S2-2"],
            }
        )
        grouped = group_pairs_by_s1(pairs)
        assert grouped == {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}}

    def test_empty_input(self):
        pairs = pd.DataFrame({"source1_entity_id": [], "candidate_entity_id": []})
        assert group_pairs_by_s1(pairs) == {}
