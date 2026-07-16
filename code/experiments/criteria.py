"""Load RQ3 ranking criteria CSV in paper JSON shape (n, d)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RQ3_CRITERION_SCHEMA_KEYS = ("n", "d")


def _read_criteria_table(path: Path) -> pd.DataFrame:
    """Read semicolon template or comma export; reject unknown layouts."""
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    if "name" in df.columns or "n" in df.columns:
        return df

    df = pd.read_csv(path, sep=",", encoding="utf-8")
    if "name" in df.columns or "n" in df.columns:
        return df

    raise ValueError(
        f"Criteria file {path} must have columns 'name' and 'description' (or 'n' and 'd'). "
        "Generate it via run_criteria_elicitation.py (export-rq3-template) after reviewing "
        "rc_consolidated.csv — not from rc_extracted.csv."
    )


def _row_to_criterion(row: pd.Series) -> dict[str, str]:
    name = row.get("n") if "n" in row.index else row.get("name")
    desc = row.get("d") if "d" in row.index else row.get("description", "")
    if pd.isna(name) or not str(name).strip():
        raise ValueError("Criteria row missing name/n")
    if pd.isna(desc):
        desc = ""
    return {"n": str(name).strip(), "d": str(desc).strip()}


def load_rq3_criteria(criteria_csv: str | Path) -> list[dict[str, str]]:
    """
    Load criteria for RQ3 as list of {"n": name, "d": description}.

    Deduplicates by criterion name (first description kept). Sorted by name for stable keys.
    """
    path = Path(criteria_csv)
    if not path.is_file():
        raise FileNotFoundError(f"RQ3 criteria CSV not found: {path}")

    df = _read_criteria_table(path)
    if "family" in df.columns and "feature" in df.columns and len(df) > 50:
        raise ValueError(
            f"{path} looks like rc_extracted (experiment rows), not an RQ3 criteria list. "
            "Use elicitation export-rq3-template → rc_wo_id.csv with columns name;description."
        )

    seen: set[str] = set()
    criteria: list[dict[str, str]] = []
    for _, row in df.iterrows():
        try:
            item = _row_to_criterion(row)
        except ValueError:
            continue
        key = item["n"].lower()
        if key in seen:
            continue
        seen.add(key)
        criteria.append(item)

    if not criteria:
        raise ValueError(f"No valid criteria in {path}")

    criteria.sort(key=lambda c: c["n"].lower())
    return criteria


def criterion_id(criterion: object) -> str:
    """Stable string id for bundle keys from criterion object or legacy string."""
    if isinstance(criterion, dict):
        return str(criterion.get("n") or criterion.get("name") or "").strip()
    return str(criterion or "").strip()


def write_rq3_criteria_template(criteria: list[dict[str, str]], path: str | Path) -> None:
    """Write semicolon CSV (name, description) for manual edit."""
    rows = [{"name": c["n"], "description": c["d"]} for c in criteria]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, sep=";")
