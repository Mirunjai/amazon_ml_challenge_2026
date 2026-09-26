import pandas as pd
import pytest

from src.error_analysis import build_error_table, categorize_predictions, filter_outcome


def _toy_scores_and_labels():
    score_df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S3-1", "S3-2"],
            "match_probability": [0.9, 0.3, 0.6, 0.1],
        }
    )
    label_df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S3-1", "S3-2"],
            "label": [1, 0, 0, 0],
        }
    )
    return score_df, label_df


class TestCategorizePredictions:
    def test_all_four_outcomes(self):
        df = pd.DataFrame(
            {
                "label": [1, 0, 1, 0],
                "match_probability": [0.9, 0.9, 0.1, 0.1],
            }
        )
        out = categorize_predictions(df, threshold=0.5)
        assert list(out["predicted_label"]) == [1, 1, 0, 0]
        assert list(out["outcome"]) == ["TP", "FP", "FN", "TN"]

    def test_threshold_boundary_is_inclusive(self):
        df = pd.DataFrame({"label": [1], "match_probability": [0.5]})
        out = categorize_predictions(df, threshold=0.5)
        assert out.loc[0, "predicted_label"] == 1


class TestBuildErrorTable:
    def test_basic_merge_and_outcomes(self):
        score_df, label_df = _toy_scores_and_labels()
        table = build_error_table(score_df, label_df, threshold=0.5)

        assert len(table) == 4
        outcomes = dict(zip(table["candidate_entity_id"], table["outcome"], strict=False))
        assert outcomes == {"S2-1": "TP", "S2-2": "TN", "S3-1": "FP", "S3-2": "TN"}

    def test_only_scored_and_labeled_pairs_appear(self):
        """Pairs with a score but no label (or vice versa) should be dropped,
        not silently filled in."""
        score_df, label_df = _toy_scores_and_labels()
        label_df = label_df.iloc[:-1]  # one fewer labeled pair than scored
        table = build_error_table(score_df, label_df, threshold=0.5)
        assert len(table) == 3

    def test_merges_feature_values(self):
        score_df, label_df = _toy_scores_and_labels()
        feature_df = score_df[["source1_entity_id", "candidate_entity_id"]].copy()
        feature_df["name_ratio"] = [0.9, 0.2, 0.5, 0.1]

        table = build_error_table(score_df, label_df, threshold=0.5, feature_df=feature_df)
        assert "name_ratio" in table.columns
        assert len(table) == 4

    def test_merges_entity_lookup_details(self):
        score_df, label_df = _toy_scores_and_labels()
        source1_lookup = pd.DataFrame(
            {
                "entity_id": ["S1-1", "S1-2"],
                "business_name": ["Acme Corp", "Beta LLC"],
                "business_address": ["1 Main St", "2 Oak Ave"],
                "country": ["US", "US"],
            }
        )
        candidate_lookup = pd.DataFrame(
            {
                "entity_id": ["S2-1", "S2-2", "S3-1", "S3-2"],
                "business_name": ["Acme Corporation", "Zeta Inc", "Gamma Ltd", "Delta Co"],
                "business_address": ["1 Main Street", "9 Pine Rd", "3 Elm St", "4 Cedar Blvd"],
                "country": ["US", "US", "India", "US"],
            }
        )

        table = build_error_table(
            score_df,
            label_df,
            threshold=0.5,
            source1_lookup=source1_lookup,
            candidate_lookup=candidate_lookup,
        )

        assert "source1_business_name" in table.columns
        assert "candidate_business_name" in table.columns
        row = table[table["candidate_entity_id"] == "S2-1"].iloc[0]
        assert row["source1_business_name"] == "Acme Corp"
        assert row["candidate_business_name"] == "Acme Corporation"


class TestFilterOutcome:
    def test_filters_and_sorts_false_positives(self):
        score_df, label_df = _toy_scores_and_labels()
        table = build_error_table(score_df, label_df, threshold=0.05)  # everything predicted positive
        fps = filter_outcome(table, "FP", sort_by="match_probability", ascending=False)
        assert set(fps["outcome"]) == {"FP"}
        assert list(fps["match_probability"]) == sorted(fps["match_probability"], reverse=True)

    def test_invalid_outcome_raises(self):
        score_df, label_df = _toy_scores_and_labels()
        table = build_error_table(score_df, label_df, threshold=0.5)
        with pytest.raises(ValueError):
            filter_outcome(table, "NOT_REAL")

    def test_n_limits_rows(self):
        score_df, label_df = _toy_scores_and_labels()
        table = build_error_table(score_df, label_df, threshold=0.05)
        fps = filter_outcome(table, "FP", n=1)
        assert len(fps) <= 1
