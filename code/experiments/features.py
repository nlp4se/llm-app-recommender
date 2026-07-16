"""Feature list loading helpers."""

from __future__ import annotations

from pathlib import Path

from code.experiments.csv_utils import read_csv_column

FEATURES_FULL_CSV = "data/input/use-case/features_large.csv"
FEATURES_SMALL_CSV = "data/input/use-case/features_small.csv"


def load_features_list(path: str | Path) -> list[str]:
    return read_csv_column(str(path))


def load_all_features() -> list[str]:
    return load_features_list(FEATURES_FULL_CSV)
