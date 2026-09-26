from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Iterator

import pandas as pd


def read_tsv(path: str | Path, columns: list[str] | None = None, **kwargs) -> pd.DataFrame:
    """Read a TSV with explicit UTF-8 encoding."""
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        usecols=columns,
        **kwargs,
    )


def iter_tsv(
    path: str | Path,
    chunksize: int = 100_000,
    columns: list[str] | None = None,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """Stream a TSV in chunks to avoid loading large files all at once."""
    reader = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        usecols=columns,
        chunksize=chunksize,
        **kwargs,
    )
    yield from reader


def count_tsv_rows(path: str | Path) -> int:
    """Count data rows without materializing the file."""
    with Path(path).open("r", encoding="utf-8") as f:
        next(f, None)
        return sum(1 for line in f if line.strip())


def validate_columns(
    columns: Iterable[str],
    required: Iterable[str],
    *,
    context: str = "dataframe",
) -> None:
    have = set(columns)
    missing = [c for c in required if c not in have]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a plain UTF-8 tab-separated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
