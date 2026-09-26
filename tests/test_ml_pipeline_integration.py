"""End-to-end integration test for the ML-side infrastructure.

Uses a tiny, fully synthetic candidate-pair set (NOT real blocking output --
Sudarshan's candidate pairs are not available yet) to prove that every piece
built tonight actually connects:

  synthetic candidate pairs
    -> label_candidate_pairs (ground truth)
    -> build_pair_feature_frame (existing, already-tested feature module)
    -> entity_level_train_val_split + split_pairs_by_entity
    -> assemble_training_data + train_classifier
    -> predict_match_scores
    -> src.evaluation.threshold_sweep (existing evaluation layer, untouched)
    -> build_error_table

This is not a claim of real model performance -- it only proves the wiring.
"""

from __future__ import annotations

import pandas as pd

from src.error_analysis import build_error_table
from src.evaluation import parse_ground_truth, threshold_sweep
from src.features import build_pair_feature_frame
from src.labeling import explode_candidate_pairs, group_pairs_by_s1, label_candidate_pairs
from src.model import assemble_training_data, predict_match_scores, train_classifier
from src.splitting import (
    assert_no_entity_leakage,
    entity_level_train_val_split,
    split_pairs_by_entity,
)


def _synthetic_dataset(n_entities: int = 40):
    """Build a tiny fully-synthetic S1/S2/S3 + ground truth + candidate set."""
    s1_rows, s2_rows, s3_rows, gt_rows, candidate_rows = [], [], [], [], []

    for i in range(n_entities):
        s1_id = f"S1-{i:03d}"
        name = f"Business {i} Pvt Ltd"
        address = f"{i} Market Street, City{i % 5}"
        country = "US" if i % 2 == 0 else "India"
        s1_rows.append(
            {
                "entity_id": s1_id,
                "business_name": name,
                "business_address": address,
                "country": country,
            }
        )

        # Every S1 gets one true S2 match (near-duplicate) and one true S3
        # match for even-indexed entities only (some S1s get 2 matches,
        # exercising multi-match; odd ones get 1; every 7th gets 0).
        candidates = []
        matches = []

        if i % 7 != 0:
            s2_id = f"S2-{i:03d}"
            s2_rows.append(
                {
                    "entity_id": s2_id,
                    "business_name": name.replace("Pvt Ltd", "Private Limited"),
                    "business_address": address,
                    "country": country,
                }
            )
            candidates.append(s2_id)
            matches.append(s2_id)

        if i % 2 == 0 and i % 7 != 0:
            s3_id = f"S3-{i:03d}"
            s3_rows.append(
                {
                    "entity_id": s3_id,
                    "business_name": name.upper(),
                    "business_address": address.upper(),
                    "country": country,
                }
            )
            candidates.append(s3_id)
            matches.append(s3_id)

        # A hard negative: a same-country, unrelated business as a decoy candidate.
        decoy_id = f"S2-decoy-{i:03d}"
        s2_rows.append(
            {
                "entity_id": decoy_id,
                "business_name": f"Unrelated Traders {i}",
                "business_address": f"{i + 500} Far Avenue, Town{i % 3}",
                "country": country,
            }
        )
        candidates.append(decoy_id)

        gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matches)})
        candidate_rows.append(
            {"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(candidates)}
        )

    s1 = pd.DataFrame(s1_rows)
    s2 = pd.DataFrame(s2_rows)
    s3 = pd.DataFrame(s3_rows)
    ground_truth = pd.DataFrame(gt_rows)
    candidate_pairs_wide = pd.DataFrame(candidate_rows)
    return s1, s2, s3, ground_truth, candidate_pairs_wide


def test_full_ml_side_wiring():
    s1, s2, s3, ground_truth, candidate_pairs_wide = _synthetic_dataset(n_entities=40)
    candidates_by_source = pd.concat([s2, s3], ignore_index=True).set_index(
        "entity_id", drop=False
    )
    s1_by_id = s1.set_index("entity_id", drop=False)

    # 1. Ground-truth -> candidate-pair labels, from Sudarshan-shaped wide input.
    long_pairs = explode_candidate_pairs(candidate_pairs_wide)
    assert set(long_pairs.columns) == {"source1_entity_id", "candidate_entity_id"}

    gt_map = parse_ground_truth(ground_truth)
    label_df = label_candidate_pairs(long_pairs, gt_map)
    assert set(label_df["label"]) <= {0, 1}
    assert label_df["label"].sum() > 0  # some true matches exist
    assert (label_df["label"] == 0).sum() > 0  # some negatives exist too

    # candidate recall sanity check via group_pairs_by_s1 + evaluation.candidate_recall
    from src.evaluation import candidate_recall

    candidate_by_s1 = group_pairs_by_s1(long_pairs)
    recall = candidate_recall(candidate_by_s1, gt_map)
    assert recall == 1.0  # by construction every true match was included as a candidate

    # 2. Pairwise features via the existing, already-tested feature module.
    pairs_for_features = [
        (s1_by_id.loc[s1_id], candidates_by_source.loc[cand_id])
        for s1_id, cand_id in zip(long_pairs["source1_entity_id"], long_pairs["candidate_entity_id"])
    ]
    feature_df = build_pair_feature_frame(pairs_for_features)

    # 3. Entity-level split -- no leakage, all pairs for one S1 stay together.
    train_ids, val_ids = entity_level_train_val_split(
        s1["entity_id"], val_fraction=0.25, seed=42
    )
    assert_no_entity_leakage(train_ids, val_ids)

    train_features, val_features = split_pairs_by_entity(feature_df, train_ids, val_ids)
    train_labels, val_labels = split_pairs_by_entity(label_df, train_ids, val_ids)
    assert set(train_features["source1_entity_id"]) <= train_ids
    assert set(val_features["source1_entity_id"]) <= val_ids

    # 4. Train + predict.
    training_df = assemble_training_data(train_features, train_labels)
    model = train_classifier(training_df)
    val_scores = predict_match_scores(model, val_features)
    assert len(val_scores) == len(val_features)

    # 5. Threshold sweep via the EXISTING evaluation module (not re-implemented here).
    val_actual_by_s1 = {
        s1_id: set(group.loc[group["label"] == 1, "candidate_entity_id"])
        for s1_id, group in val_labels.groupby("source1_entity_id")
    }
    sweep = threshold_sweep(val_scores, val_actual_by_s1, thresholds=[0.2, 0.4, 0.6, 0.8])
    assert list(sweep["threshold"]) == [0.2, 0.4, 0.6, 0.8]
    assert sweep["macro_f05"].between(0.0, 1.0).all()

    # 6. Error analysis scaffolding on the validation predictions.
    best_threshold = sweep.sort_values("macro_f05", ascending=False).iloc[0]["threshold"]
    error_table = build_error_table(
        val_scores,
        val_labels,
        threshold=best_threshold,
        feature_df=val_features,
        source1_lookup=s1,
        candidate_lookup=candidates_by_source.reset_index(drop=True),
    )
    assert {"TP", "FP", "FN", "TN"} >= set(error_table["outcome"].unique())
    assert "source1_business_name" in error_table.columns
    assert "candidate_business_name" in error_table.columns
