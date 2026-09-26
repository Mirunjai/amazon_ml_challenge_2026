from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_submission_frames(
    test_source1: pd.DataFrame,
    predicted_by_s1: dict[str, set[str]],
    candidate_by_s1: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the two exact output tables required by the challenge."""
    matching_rows = []
    candidate_rows = []

    for s1_id in test_source1["entity_id"].astype(str):
        matches = sorted(predicted_by_s1.get(s1_id, set()))
        candidates = sorted(candidate_by_s1.get(s1_id, set()))

        matching_rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matches),
            }
        )
        candidate_rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(candidates),
            }
        )

    matching = pd.DataFrame(
        matching_rows,
        columns=["source1_entity_id", "matched_entity_ids"],
    )
    candidates = pd.DataFrame(
        candidate_rows,
        columns=["source1_entity_id", "candidate_entity_ids"],
    )
    return matching, candidates


def write_submission(
    matching: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    matching.to_csv(matching_path, sep="\t", index=False, encoding="utf-8")
    candidates.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8")
    return matching_path, candidate_path
