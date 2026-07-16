"""Small CSV helpers shared across experiment modules."""

from __future__ import annotations

import csv
from pathlib import Path


def read_csv_column(path: str | Path, column: int = 0, skip_header: bool = True) -> list[str]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        return [row[column] for row in reader if row]
