"""Lightweight experiment tracking.

A single append-only CSV under outputs/experiments/ (already gitignored, see
.gitignore). Column names line up with the "Empty experiment log" scaffold in
the team notebook (blocking_version, features_version, model,
candidate_pairs, candidate_recall, threshold, macro_f05, global_precision,
global_recall, notes) plus a few fields needed to compare ML-side runs
(model_params, positive/negative counts).

Deliberately not a framework: one dataclass, one append function, one loader.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_LOG_PATH = "outputs/experiments/experiment_log.csv"

COLUMNS: list[str] = [
    "experiment_id",
    "timestamp",
    "blocking_version",
    "features_version",
    "model",
    "model_params",
    "threshold",
    "candidate_pairs",
    "positive_count",
    "negative_count",
    "candidate_recall",
    "precision",
    "recall",
    "macro_f05",
    "notes",
]


@dataclass
class ExperimentRecord:
    """One row of the experiment log. Any field can be left as None/default
    if it isn't known yet (e.g. macro_f05 before Narendra's evaluator runs)."""

    experiment_id: str
    features_version: str
    model: str
    model_params: dict = field(default_factory=dict)
    blocking_version: str | None = None
    threshold: float | None = None
    candidate_pairs: int | None = None
    positive_count: int | None = None
    negative_count: int | None = None
    candidate_recall: float | None = None
    precision: float | None = None
    recall: float | None = None
    macro_f05: float | None = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        row = asdict(self)
        row["model_params"] = json.dumps(row["model_params"], sort_keys=True)
        return row


def append_experiment(record: ExperimentRecord, path: str | Path = DEFAULT_LOG_PATH) -> Path:
    """Append one experiment record to the CSV log, creating it (with header) if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = record.to_row()
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in COLUMNS})
    return path


def load_experiment_log(path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """Load the experiment log, or an empty (correctly-columned) frame if none exists yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)
