"""
Unit tests for `src/analyze_candidates.py` (blocking candidate-pair quality
analysis: `scan_candidate_file`, `compute_report`, `format_report_text`,
`main`).

Note on location: the module under test lives at `src/analyze_candidates.py`
(not `scripts/analyze_candidates.py` -- there is no such file in this repo).
This test module imports from `src.analyze_candidates` accordingly.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analyze_candidates import (
    compute_report,
    format_report_text,
    main,
    scan_candidate_file,
)


def _write_tsv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False, encoding="utf-8")
    return path


@pytest.fixture
def candidate_file(tmp_path):
    """A small, deliberately imperfect candidate file:
    - S1-1: 2 candidates, both true matches survive as candidates.
    - S1-2: 1 candidate, but the true match is NOT among them (recall loss).
    - S1-3: correct singleton, zero candidates.
    - S1-4: duplicate candidate IDs within the same row (dedup exercised).
    """
    path = tmp_path / "candidate_pairs.tsv"
    return _write_tsv(
        path,
        [
            {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "candidate_entity_ids": "S2-99"},
            {"source1_entity_id": "S1-3", "candidate_entity_ids": ""},
            {"source1_entity_id": "S1-4", "candidate_entity_ids": "S2-4,S2-4,S3-4"},
        ],
        columns=["source1_entity_id", "candidate_entity_ids"],
    )


@pytest.fixture
def ground_truth_file(tmp_path):
    path = tmp_path / "train_ground_truth.tsv"
    return _write_tsv(
        path,
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-7"},  # not in candidates
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""},  # correct singleton
            {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-4"},
        ],
        columns=["source1_entity_id", "matched_entity_ids"],
    )


class TestScanCandidateFile:
    def test_basic_scan_counts_and_columns(self, candidate_file):
        candidates_by_s1, raw_counts_by_s1, columns = scan_candidate_file(candidate_file)

        assert columns == ["source1_entity_id", "candidate_entity_ids"]
        assert candidates_by_s1["S1-1"] == {"S2-1", "S3-1"}
        assert candidates_by_s1["S1-2"] == {"S2-99"}
        assert candidates_by_s1["S1-3"] == set()  # present, zero candidates
        assert raw_counts_by_s1["S1-3"] == 0

    def test_duplicate_candidate_ids_deduplicated_but_counted_raw(self, candidate_file):
        candidates_by_s1, raw_counts_by_s1, _ = scan_candidate_file(candidate_file)

        # S2-4 appears twice in the source row; the set collapses it, but the
        # raw (pre-dedup) count must still reflect 3 tokens written.
        assert candidates_by_s1["S1-4"] == {"S2-4", "S3-4"}
        assert raw_counts_by_s1["S1-4"] == 3

    def test_streams_in_chunks_smaller_than_file(self, candidate_file):
        """chunksize=1 forces multiple chunks; result must match a single
        large-chunk read, proving the chunked accumulation is correct."""
        chunked, chunked_raw, _ = scan_candidate_file(candidate_file, chunksize=1)
        whole, whole_raw, _ = scan_candidate_file(candidate_file, chunksize=10_000)
        assert chunked == whole
        assert chunked_raw == whole_raw

    def test_empty_candidate_string_yields_empty_set_not_missing_entry(self, candidate_file):
        candidates_by_s1, _, _ = scan_candidate_file(candidate_file)
        # S1-3 has an empty candidate_entity_ids cell -- it must still be a
        # key (blocking produced a row for it), just with an empty set, to
        # distinguish "considered, zero candidates" from "never appeared".
        assert "S1-3" in candidates_by_s1
        assert candidates_by_s1["S1-3"] == set()

    def test_missing_s1_id_raises(self, tmp_path):
        bad = _write_tsv(
            tmp_path / "bad.tsv",
            [{"source1_entity_id": "", "candidate_entity_ids": "S2-1"}],
            columns=["source1_entity_id", "candidate_entity_ids"],
        )
        with pytest.raises(ValueError):
            scan_candidate_file(bad)

    def test_wrong_schema_raises(self, tmp_path):
        bad = _write_tsv(
            tmp_path / "wrong_schema.tsv",
            [{"s1": "S1-1", "candidates": "S2-1"}],
            columns=["s1", "candidates"],
        )
        with pytest.raises(ValueError):
            scan_candidate_file(bad)

    def test_truly_empty_file_raises(self, tmp_path):
        # A genuinely 0-byte file has no header to parse at all.
        empty = tmp_path / "empty.tsv"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            scan_candidate_file(empty)

    def test_header_only_file_yields_no_entities_without_raising(self, tmp_path):
        # A file with a valid header but zero data rows is a legitimate (if
        # unusual) "blocking produced nothing yet" state, not malformed input.
        header_only = _write_tsv(
            tmp_path / "header_only.tsv",
            [],
            columns=["source1_entity_id", "candidate_entity_ids"],
        )
        candidates_by_s1, raw_counts_by_s1, columns = scan_candidate_file(header_only)
        assert columns == ["source1_entity_id", "candidate_entity_ids"]
        assert candidates_by_s1 == {}
        assert raw_counts_by_s1 == {}

    def test_repeated_s1_row_merges_rather_than_overwrites(self, tmp_path):
        path = _write_tsv(
            tmp_path / "dup_s1.tsv",
            [
                {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"},
                {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-2"},
            ],
            columns=["source1_entity_id", "candidate_entity_ids"],
        )
        candidates_by_s1, raw_counts_by_s1, _ = scan_candidate_file(path)
        assert candidates_by_s1["S1-1"] == {"S2-1", "S2-2"}
        assert raw_counts_by_s1["S1-1"] == 2


class TestComputeReport:
    def test_candidate_recall_matches_evaluation_definition(
        self, candidate_file, ground_truth_file
    ):
        from src.evaluation import candidate_recall, parse_ground_truth

        report = compute_report(candidate_file, ground_truth_file)

        gt_map = parse_ground_truth(ground_truth_file)
        candidates_by_s1, _, _ = scan_candidate_file(candidate_file)
        expected_recall = candidate_recall(gt_map, candidates_by_s1)

        assert report["candidate_recall"] == pytest.approx(expected_recall)
        # Total true matches: S1-1 has 2, S1-2 has 1, S1-3 has 0, S1-4 has 1 = 4.
        assert report["total_true_matches"] == 4
        # Retained: S1-1's 2 matches survive, S1-4's 1 match survives,
        # S1-2's true match (S2-7) is NOT among its candidates -> 3 retained.
        assert report["retained_true_matches"] == 3
        assert report["candidate_recall"] == pytest.approx(3 / 4)

    def test_volume_and_zero_candidate_stats(self, candidate_file, ground_truth_file):
        report = compute_report(candidate_file, ground_truth_file)

        assert report["unique_s1_entities_in_candidates"] == 4
        assert report["s1_zero_candidates"] == 1  # S1-3
        assert report["s1_with_candidates"] == 3
        assert report["duplicate_candidate_assignments"] == 1  # the repeated S2-4

    def test_ground_truth_population_independent_of_candidate_file_extras(
        self, tmp_path, ground_truth_file
    ):
        """S1 entities present only in the candidate file (not in ground
        truth) must not inflate/shrink the ground-truth-based recall calc."""
        extra_candidates = _write_tsv(
            tmp_path / "extra.tsv",
            [
                {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1"},
                {"source1_entity_id": "S1-2", "candidate_entity_ids": "S2-7"},
                {"source1_entity_id": "S1-3", "candidate_entity_ids": ""},
                {"source1_entity_id": "S1-4", "candidate_entity_ids": "S2-4"},
                # Not present anywhere in ground truth:
                {"source1_entity_id": "S1-EXTRA", "candidate_entity_ids": "S2-123"},
            ],
            columns=["source1_entity_id", "candidate_entity_ids"],
        )
        report = compute_report(extra_candidates, ground_truth_file)
        assert report["ground_truth_s1_entities"] == 4
        # Now every true match is retained (S1-2's S2-7 is included this time).
        assert report["candidate_recall"] == pytest.approx(1.0)

    def test_candidates_per_s1_stats_present(self, candidate_file, ground_truth_file):
        report = compute_report(candidate_file, ground_truth_file)
        stats = report["candidates_per_s1"]
        for key in ("min", "median", "mean", "p90", "p95", "p99", "max"):
            assert key in stats
        assert stats["min"] == 0  # S1-3
        assert stats["max"] == 2  # S1-1 or S1-4

    def test_chunksize_does_not_change_report(self, candidate_file, ground_truth_file):
        small_chunks = compute_report(candidate_file, ground_truth_file, chunksize=1)
        default_chunks = compute_report(candidate_file, ground_truth_file)
        assert small_chunks == default_chunks


class TestFormatReportText:
    def test_contains_key_figures(self, candidate_file, ground_truth_file):
        report = compute_report(candidate_file, ground_truth_file)
        text = format_report_text(report)

        assert "CANDIDATE ANALYSIS" in text
        assert "Candidate Recall" in text
        assert f"{report['candidate_recall'] * 100:.2f}%" in text
        assert "source1_entity_id, candidate_entity_ids" in text


class TestMainCli:
    def test_main_prints_report_and_writes_json_output(
        self, candidate_file, ground_truth_file, tmp_path, capsys
    ):
        output_path = tmp_path / "report.json"
        main(
            [
                "--candidates",
                str(candidate_file),
                "--ground-truth",
                str(ground_truth_file),
                "--output",
                str(output_path),
            ]
        )
        captured = capsys.readouterr()
        assert "CANDIDATE ANALYSIS" in captured.out
        assert output_path.exists()

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["candidate_file"] == str(candidate_file)
        assert "candidate_recall" in saved

    def test_main_without_output_does_not_create_file(
        self, candidate_file, ground_truth_file, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        main(["--candidates", str(candidate_file), "--ground-truth", str(ground_truth_file)])
        captured = capsys.readouterr()
        assert "CANDIDATE ANALYSIS" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])