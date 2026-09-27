import unittest
import pandas as pd
from src.evaluation import (
    candidate_recall,
    evaluate,
    macro_f05,
    parse_ground_truth,
    precision_recall_f05,
    predictions_from_scores,
)


class TestEvaluation(unittest.TestCase):
    def test_correct_singleton_scores_one(self):
        """GT empty + prediction empty -> P=1.0, R=1.0, F0.5=1.0."""
        gt = {"s1_1": set()}
        pred = {"s1_1": set()}

        result = evaluate(gt, pred)
        s1_metrics = result["per_s1"]["s1_1"]

        self.assertEqual(s1_metrics["TP"], 0)
        self.assertEqual(s1_metrics["FP"], 0)
        self.assertEqual(s1_metrics["FN"], 0)
        self.assertEqual(s1_metrics["Precision"], 1.0)
        self.assertEqual(s1_metrics["Recall"], 1.0)
        self.assertEqual(s1_metrics["F0.5"], 1.0)
        self.assertEqual(result["macro_f0.5"], 1.0)

    def test_incorrect_singleton_false_positive(self):
        """GT empty + prediction non-empty -> F0.5=0.0."""
        gt = {"s1_1": set()}
        pred = {"s1_1": {"s2_99"}}

        result = evaluate(gt, pred)
        self.assertEqual(result["per_s1"]["s1_1"]["F0.5"], 0.0)
        self.assertEqual(result["macro_f0.5"], 0.0)

    def test_incorrect_singleton_false_negative(self):
        """GT non-empty + prediction empty -> F0.5=0.0."""
        gt = {"s1_1": {"s2_10"}}
        pred = {"s1_1": set()}

        result = evaluate(gt, pred)
        self.assertEqual(result["per_s1"]["s1_1"]["F0.5"], 0.0)
        self.assertEqual(result["macro_f0.5"], 0.0)

    def test_missing_prediction_s1_is_treated_as_empty(self):
        """S1 present in GT but absent from predictions behaves like empty prediction."""
        gt = {"s1_1": {"s2_10"}}
        pred = {}

        result = evaluate(gt, pred)

        self.assertEqual(result["per_s1"]["s1_1"]["TP"], 0)
        self.assertEqual(result["per_s1"]["s1_1"]["FP"], 0)
        self.assertEqual(result["per_s1"]["s1_1"]["FN"], 1)
        self.assertEqual(result["per_s1"]["s1_1"]["F0.5"], 0.0)

    def test_macro_is_average_of_per_s1_scores(self):
        """Macro F0.5 is the arithmetic mean across S1 entities."""
        gt = {
            "s1_1": set(),             # F0.5 = 1
            "s1_2": {"s2_1"},          # F0.5 = 0
        }
        pred = {
            "s1_1": set(),
            "s1_2": set(),
        }

        result = evaluate(gt, pred)

        self.assertAlmostEqual(result["macro_f0.5"], 0.5)

    def test_standardized_argument_order_f05(self):
        """Verify consistent (actual, predicted) argument ordering across functions."""
        # 1 TP, 1 FP, 0 FN:
        # P = 0.5, R = 1.0
        # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)
        #      = 5/9 ≈ 0.5556
        actual = {"m1"}
        predicted = {"m1", "m2"}

        p, r, f05 = precision_recall_f05(actual, predicted)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f05, 5.0 / 9.0)

        # Confirm macro_f05 and evaluate agree on the same (actual, predicted) order
        gt_map = {"s1_1": actual}
        pred_map = {"s1_1": predicted}
        self.assertAlmostEqual(macro_f05(gt_map, pred_map), 5.0 / 9.0)
        self.assertAlmostEqual(evaluate(gt_map, pred_map)["macro_f0.5"], 5.0 / 9.0)

    def test_extra_predicted_s1_ignored(self):
        """Predictions for S1 IDs not in GT must not enter the macro average."""
        gt = {"s1_1": set()}
        pred = {"s1_1": set(), "s1_extra": {"s2_99"}}

        self.assertEqual(macro_f05(gt, pred), 1.0)
        self.assertEqual(evaluate(gt, pred)["macro_f0.5"], 1.0)

    def test_duplicate_or_nan_s1_rejected_in_gt(self):
        """Duplicate or NaN S1 IDs in ground-truth DataFrame raise ValueError."""
        dup_df = pd.DataFrame({
            "source1_entity_id": ["s1_1", "s1_1"],
            "matched_entity_ids": ["s2_1", "s2_2"],
        })
        with self.assertRaises(ValueError):
            parse_ground_truth(dup_df)

        nan_df = pd.DataFrame({
            "source1_entity_id": [None],
            "matched_entity_ids": ["s2_1"],
        })
        with self.assertRaises(ValueError):
            parse_ground_truth(nan_df)

    def test_candidate_recall_and_multi_match_predictions(self):
        """Verify candidate recall and multi-match thresholding."""
        gt = {"s1_1": {"s2_1", "s3_1"}, "s1_2": set()}
        cands = {"s1_1": {"s2_1", "s2_99"}}
        self.assertAlmostEqual(candidate_recall(gt, cands), 0.5)

        score_df = pd.DataFrame({
            "source1_entity_id": ["s1_1", "s1_1", "s1_2"],
            "candidate_entity_id": ["s2_1", "s3_1", "s2_5"],
            "match_probability": [0.85, 0.75, 0.20],
        })
        preds = predictions_from_scores(score_df, threshold=0.70)
        self.assertEqual(preds["s1_1"], {"s2_1", "s3_1"})
        self.assertEqual(preds["s1_2"], set())


if __name__ == "__main__":
    unittest.main()