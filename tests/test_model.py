import numpy as np
import pandas as pd
import pytest

from src.model import (
    BASELINE_FEATURE_COLUMNS,
    assemble_training_data,
    predict_match_scores,
    train_classifier,
)


def _toy_feature_df(n_pos: int = 15, n_neg: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small synthetic feature+label frames that are trivially separable,
    so a fitted classifier's behavior is easy to assert on."""
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    for i in range(n_pos):
        rows.append(
            {
                "source1_entity_id": f"S1-{i}",
                "candidate_entity_id": f"S2-{i}",
                "name_ratio": 0.95 + rng.uniform(-0.02, 0.02),
                "name_token_set_ratio": 0.95 + rng.uniform(-0.02, 0.02),
                "address_ratio": 0.9 + rng.uniform(-0.02, 0.02),
                "address_token_set_ratio": 0.9 + rng.uniform(-0.02, 0.02),
                "country_match": 1.0,
                "s1_address_missing": 0,
                "candidate_address_missing": 0,
                "both_address_missing": 0,
            }
        )
        labels.append(1)
    for i in range(n_neg):
        rows.append(
            {
                "source1_entity_id": f"S1-{i}",
                "candidate_entity_id": f"S3-{i}",
                "name_ratio": 0.1 + rng.uniform(-0.02, 0.02),
                "name_token_set_ratio": 0.1 + rng.uniform(-0.02, 0.02),
                "address_ratio": 0.1 + rng.uniform(-0.02, 0.02),
                "address_token_set_ratio": 0.1 + rng.uniform(-0.02, 0.02),
                "country_match": 0.0,
                "s1_address_missing": 0,
                "candidate_address_missing": 0,
                "both_address_missing": 0,
            }
        )
        labels.append(0)

    feature_df = pd.DataFrame(rows)
    label_df = feature_df[["source1_entity_id", "candidate_entity_id"]].copy()
    label_df["label"] = labels
    return feature_df, label_df


class TestAssembleTrainingData:
    def test_joins_features_and_labels(self):
        feature_df, label_df = _toy_feature_df(3, 3)
        training_df = assemble_training_data(feature_df, label_df)
        assert len(training_df) == len(feature_df)
        assert "label" in training_df.columns
        for col in BASELINE_FEATURE_COLUMNS:
            assert col in training_df.columns

    def test_raises_when_label_missing_for_a_pair(self):
        feature_df, label_df = _toy_feature_df(3, 3)
        label_df = label_df.iloc[:-1]  # drop the label for one pair
        with pytest.raises(ValueError):
            assemble_training_data(feature_df, label_df)


class TestTrainClassifier:
    def test_trains_without_error_with_default_model(self):
        feature_df, label_df = _toy_feature_df(20, 20)
        training_df = assemble_training_data(feature_df, label_df)
        model = train_classifier(training_df)
        assert hasattr(model, "predict_proba")

    def test_missing_feature_column_raises(self):
        feature_df, label_df = _toy_feature_df(5, 5)
        training_df = assemble_training_data(feature_df, label_df)
        with pytest.raises(ValueError):
            train_classifier(training_df, feature_columns=["not_a_real_feature"])

    def test_model_can_be_swapped(self):
        """A different sklearn-compatible classifier should work with no
        other code changes -- this is the whole point of the interface."""
        from sklearn.ensemble import ExtraTreesClassifier

        feature_df, label_df = _toy_feature_df(20, 20)
        training_df = assemble_training_data(feature_df, label_df)
        model = train_classifier(training_df, model=ExtraTreesClassifier(n_estimators=50, random_state=0))
        assert hasattr(model, "predict_proba")


class TestPredictMatchScores:
    def test_output_shape_and_columns(self):
        feature_df, label_df = _toy_feature_df(20, 20)
        training_df = assemble_training_data(feature_df, label_df)
        model = train_classifier(training_df)

        scores = predict_match_scores(model, feature_df)

        assert list(scores.columns) == [
            "source1_entity_id",
            "candidate_entity_id",
            "match_probability",
        ]
        assert len(scores) == len(feature_df)
        assert scores["match_probability"].between(0.0, 1.0).all()

    def test_separates_obvious_positives_and_negatives(self):
        """Sanity check on a trivially separable synthetic set: the model
        should score the high-similarity pairs higher than the low-similarity
        ones on average."""
        feature_df, label_df = _toy_feature_df(30, 30)
        training_df = assemble_training_data(feature_df, label_df)
        model = train_classifier(training_df)
        scores = predict_match_scores(model, feature_df).merge(
            label_df, on=["source1_entity_id", "candidate_entity_id"]
        )
        pos_mean = scores.loc[scores["label"] == 1, "match_probability"].mean()
        neg_mean = scores.loc[scores["label"] == 0, "match_probability"].mean()
        assert pos_mean > neg_mean

    def test_output_feeds_directly_into_evaluation_threshold_sweep(self):
        """Integration check: predict_match_scores' output shape must be
        directly consumable by src.evaluation.threshold_sweep /
        predictions_from_scores without any adaptation, since that module is
        owned by the evaluation layer and should not need to change."""
        from src.evaluation import predictions_from_scores, threshold_sweep

        feature_df, label_df = _toy_feature_df(15, 15)
        training_df = assemble_training_data(feature_df, label_df)
        model = train_classifier(training_df)
        scores = predict_match_scores(model, feature_df)

        actual_by_s1 = {
            s1: set(group.loc[group["label"] == 1, "candidate_entity_id"])
            for s1, group in label_df.groupby("source1_entity_id")
        }

        preds_at_half = predictions_from_scores(scores, threshold=0.5)
        assert isinstance(preds_at_half, dict)

        sweep = threshold_sweep(scores, actual_by_s1, thresholds=[0.2, 0.5, 0.8])
        assert list(sweep["threshold"]) == [0.2, 0.5, 0.8]
        assert sweep["macro_f05"].between(0.0, 1.0).all()

    def test_missing_feature_column_raises(self):
        feature_df, label_df = _toy_feature_df(5, 5)
        training_df = assemble_training_data(feature_df, label_df)
        model = train_classifier(training_df)
        bad_features = feature_df.drop(columns=["name_ratio"])
        with pytest.raises(ValueError):
            predict_match_scores(model, bad_features)
