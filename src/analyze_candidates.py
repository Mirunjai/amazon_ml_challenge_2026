#!/usr/bin/env python3
"""Blocking candidate-pair quality analysis.

Sudarshan's blocking stage produces a candidate file in the official
challenge "wide" shape (`src.config.CANDIDATE_COLUMNS`):

    source1_entity_id \t candidate_entity_ids
    S1-00001          \t S2-00047,S2-00193,S3-00812
    S1-00002          \t S3-00004
    S1-00003          \t

i.e. one row per Source-1 entity, with a comma-separated list of candidate
Source-2/Source-3 entity IDs (empty string for a singleton with no
candidates). This is the exact shape already used by:

    src.config.CANDIDATE_COLUMNS
    src.labeling.explode_candidate_pairs
    src.pipeline.make_submission_frames / write_submission (candidate_pairs.tsv)

This script does NOT invent a different schema. It validates the file
against `src.config.CANDIDATE_COLUMNS` and fails loudly if that shape is not
what's on disk.

Given a candidate file and `train_ground_truth.tsv`, this script measures the
QUALITY OF BLOCKING before any ML training happens: candidate volume per S1
and, most importantly, candidate recall -- the fraction of ground-truth true
matches that actually survived into the candidate set. This is BLOCKING
candidate recall, not model recall.

Usage:

    python scripts/analyze_candidates.py \\
        --candidates <candidate_file.tsv> \\
        --ground-truth <train_ground_truth.tsv> \\
        [--output <report.json>] \\
        [--chunksize 200000]

The candidate file is streamed in chunks (see `src.io_utils.iter_tsv`) so
this remains usable on multi-million-row files; it is never loaded whole
into a single DataFrame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as `python scripts/analyze_candidates.py ...` from the repo
# root without requiring the package to be installed (mirrors the
# `sys.path.insert(...)` bootstrap already used in the main notebook).
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import CANDIDATE_COLUMNS  # noqa: E402
from src.evaluation import candidate_recall, parse_ground_truth  # noqa: E402
from src.io_utils import iter_tsv, validate_columns  # noqa: E402

S1_COL, CANDIDATES_COL = CANDIDATE_COLUMNS
DEFAULT_CHUNKSIZE = 200_000

_PERCENT_STATS = ("min", "median", "mean", "p90", "p95", "p99", "max")


# ============================================================
# Single pass over the candidate file
# ============================================================

def scan_candidate_file(
    path: str | Path,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[dict[str, set[str]], dict[str, int], list[str]]:
    """Stream the candidate TSV once and accumulate per-S1 candidate sets.

    Returns:
        candidates_by_s1: unique candidate IDs per S1 entity, i.e. exactly
            what downstream code (`src.labeling.group_pairs_by_s1`,
            `src.evaluation.candidate_recall`) actually consumes -- a set,
            not a raw list. Every S1 row seen in the file gets an entry here,
            even if its candidate list is empty, so presence in this dict's
            keys distinguishes "blocking produced a row for this S1" from
            "S1 is entirely absent from the file".
        raw_counts_by_s1: total (non-deduplicated) candidate-token count per
            S1 entity, i.e. how many (S1, candidate) assignments were written
            for that S1 before deduplication. Used to detect duplicate
            candidate assignments.
        columns: the candidate file's column names, as found on disk.

    If the same S1 entity appears on more than one row (a malformed but not
    impossible blocking output), its candidate lists across all such rows
    are merged rather than one silently overwriting another.

    IDs are kept as plain strings throughout; missing cells become "" rather
    than the pandas default NaN/"nan".
    """
    path = Path(path)
    candidates_by_s1: dict[str, set[str]] = {}
    raw_counts_by_s1: dict[str, int] = {}
    columns: list[str] | None = None
    saw_any_chunk = False

    for chunk in iter_tsv(
        path,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    ):
        saw_any_chunk = True
        if columns is None:
            columns = list(chunk.columns)
            validate_columns(columns, CANDIDATE_COLUMNS, context="candidate file")

        for s1_raw, cand_raw in zip(
            chunk[S1_COL], chunk[CANDIDATES_COL], strict=False
        ):
            s1 = (s1_raw or "").strip()
            if not s1 or s1.lower() == "nan":
                raise ValueError(
                    "Encountered a candidate row with an empty/malformed "
                    f"'{S1_COL}' value in {path}."
                )

            bucket = candidates_by_s1.setdefault(s1, set())
            raw_counts_by_s1.setdefault(s1, 0)

            tokens = [tok.strip() for tok in (cand_raw or "").split(",")]
            tokens = [tok for tok in tokens if tok]

            raw_counts_by_s1[s1] += len(tokens)
            bucket.update(tokens)

    if not saw_any_chunk or columns is None:
        raise ValueError(f"Candidate file is empty or unreadable: {path}")

    return candidates_by_s1, raw_counts_by_s1, columns


# ============================================================
# Report assembly
# ============================================================

def _candidates_per_s1_stats(counts: np.ndarray) -> dict[str, float]:
    if counts.size == 0:
        return {stat: 0 for stat in _PERCENT_STATS}
    return {
        "min": int(counts.min()),
        "median": float(np.median(counts)),
        "mean": float(counts.mean()),
        "p90": float(np.percentile(counts, 90)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
    }


def compute_report(
    candidates_path: str | Path,
    ground_truth_path: str | Path,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Build the full candidate-analysis report as a plain (JSON-safe) dict.

    Reuses `src.evaluation.parse_ground_truth` and
    `src.evaluation.candidate_recall` rather than re-implementing ground-truth
    parsing or the recall definition. Candidate recall is computed against
    the COMPLETE ground truth (every S1 in `ground_truth_path`), regardless
    of which S1 entities happen to appear in the candidate file: S1 entities
    that exist only in the candidate file never enter `parse_ground_truth`'s
    output and therefore cannot inflate or shrink the ground-truth
    population used for recall.
    """
    candidates_by_s1, raw_counts_by_s1, columns = scan_candidate_file(
        candidates_path, chunksize=chunksize
    )
    gt_map = parse_ground_truth(ground_truth_path)

    unique_s1_in_candidates = len(candidates_by_s1)
    total_assignments = sum(raw_counts_by_s1.values())
    duplicate_assignments = sum(
        raw_counts_by_s1[s1] - len(candidates_by_s1[s1]) for s1 in candidates_by_s1
    )

    per_s1_counts = np.array(
        [len(v) for v in candidates_by_s1.values()], dtype=np.int64
    )
    stats = _candidates_per_s1_stats(per_s1_counts)

    zero_candidate_count = int((per_s1_counts == 0).sum()) if per_s1_counts.size else 0
    nonzero_candidate_count = unique_s1_in_candidates - zero_candidate_count

    gt_s1_count = len(gt_map)
    gt_s1_represented = sum(1 for s1 in gt_map if s1 in candidates_by_s1)
    total_true_matches = sum(len(matches) for matches in gt_map.values())

    # Reuse the canonical recall definition instead of re-deriving it; only
    # recover the raw retained-match count (not exposed by candidate_recall)
    # by multiplying back through the exact total, which is safe at these
    # (millions-of-rows, not more) scales.
    recall = candidate_recall(gt_map, candidates_by_s1)
    retained_true_matches = (
        int(round(recall * total_true_matches)) if total_true_matches else 0
    )

    return {
        "candidate_file": str(candidates_path),
        "ground_truth_file": str(ground_truth_path),
        "candidate_schema": columns,
        "total_candidate_assignments": total_assignments,
        "unique_s1_entities_in_candidates": unique_s1_in_candidates,
        "candidates_per_s1": stats,
        "s1_zero_candidates": zero_candidate_count,
        "s1_with_candidates": nonzero_candidate_count,
        "duplicate_candidate_assignments": duplicate_assignments,
        "ground_truth_s1_entities": gt_s1_count,
        "ground_truth_s1_represented_in_candidates": gt_s1_represented,
        "total_true_matches": total_true_matches,
        "retained_true_matches": retained_true_matches,
        "candidate_recall": recall,
    }


# ============================================================
# Console rendering
# ============================================================

def _fmt(n: object) -> str:
    """Thousands-separated formatting for large counts; passes floats through."""
    if isinstance(n, (int, np.integer)):
        return f"{int(n):,}"
    return str(n)


def format_report_text(report: dict) -> str:
    stats = report["candidates_per_s1"]
    lines = [
        "=" * 60,
        "CANDIDATE ANALYSIS",
        "=" * 60,
        "",
        f"Candidate file: {report['candidate_file']}",
        f"Ground truth file: {report['ground_truth_file']}",
        "",
        f"Candidate schema: {', '.join(report['candidate_schema'])}",
        "",
        f"Total candidate assignments: {_fmt(report['total_candidate_assignments'])}",
        f"Unique S1 entities: {_fmt(report['unique_s1_entities_in_candidates'])}",
        f"Ground-truth S1 entities: {_fmt(report['ground_truth_s1_entities'])}",
        "Ground-truth S1s represented in candidates: "
        f"{_fmt(report['ground_truth_s1_represented_in_candidates'])}",
        "",
        "Candidates per S1:",
        f"    Min    : {_fmt(stats['min'])}",
        f"    Median : {stats['median']}",
        f"    Mean   : {stats['mean']:.3f}",
        f"    P90    : {stats['p90']}",
        f"    P95    : {stats['p95']}",
        f"    P99    : {stats['p99']}",
        f"    Max    : {_fmt(stats['max'])}",
        "",
        f"S1 entities with zero candidates: {_fmt(report['s1_zero_candidates'])}",
        f"S1 entities with >=1 candidate: {_fmt(report['s1_with_candidates'])}",
        "",
        f"Total true matches : {_fmt(report['total_true_matches'])}",
        f"Retained true matches: {_fmt(report['retained_true_matches'])}",
        "",
        "Candidate Recall:",
        f"    decimal: {report['candidate_recall']:.6f}",
        f"    percent: {report['candidate_recall'] * 100:.2f}%",
        "",
        f"Duplicate candidate assignments: {_fmt(report['duplicate_candidate_assignments'])}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure blocking candidate quality (volume + candidate recall) "
        "for a Sudarshan-shaped candidate_pairs.tsv against train_ground_truth.tsv."
    )
    parser.add_argument(
        "--candidates", required=True, help="Path to the candidate_pairs.tsv-shaped file."
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to train_ground_truth.tsv."
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to save the report as JSON."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"Rows per chunk when streaming the candidate file (default {DEFAULT_CHUNKSIZE}).",
    )
    args = parser.parse_args(argv)

    report = compute_report(
        args.candidates,
        args.ground_truth,
        chunksize=args.chunksize,
    )

    print(format_report_text(report))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()