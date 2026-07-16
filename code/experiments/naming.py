from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

DatasetScale = Literal["large", "small"]

FEATURES_LARGE_CSV = "data/input/use-case/features_large.csv"
FEATURES_SMALL_CSV = "data/input/use-case/features_small.csv"
FEATURES_CSV = FEATURES_LARGE_CSV


def to_camel_case(text: str) -> str:
    """Convert 'Photo effects' -> 'PhotoEffects'."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return "Unknown"
    return "".join(word[:1].upper() + word[1:].lower() for word in words)


def from_camel_case(text: str) -> str:
    """Convert 'PhotoEffects' -> 'Photo effects' (best-effort spacing)."""
    if not text:
        return ""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return spaced.strip()


@lru_cache(maxsize=1)
def _canonical_features() -> tuple[str, ...]:
    path = Path(FEATURES_CSV)
    if not path.exists():
        return ()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return tuple(row[0] for row in reader if row)


def resolve_feature_name(camel_token: str) -> str:
    """Map CamelCase token back to canonical feature label when possible."""
    for feature in _canonical_features():
        if to_camel_case(feature) == camel_token:
            return feature
    return from_camel_case(camel_token)


def rq_output_dir(root: str, rq_id: str) -> Path:
    return Path(root) / rq_id


def scale_from_features_path(path: str | Path) -> DatasetScale:
    """Infer large vs small feature set from a features CSV path."""
    stem = Path(path).stem.lower()
    if "small" in stem:
        return "small"
    if "proprietary" in stem and "large" not in stem:
        return "small"
    return "large"


def dataset_suite(family: str, scale: DatasetScale) -> str:
    """Output suite id, e.g. open_large or proprietary_small."""
    return f"{family}_{scale}"


def resolve_dataset_suite(
    family: str,
    *,
    dataset_suite_override: str | None = None,
    features_csv: str | None = None,
    features_csv_proprietary: str | None = None,
    default_csv: str = FEATURES_LARGE_CSV,
) -> str:
    if dataset_suite_override:
        return dataset_suite_override
    if family == "proprietary" and features_csv_proprietary:
        csv_path = features_csv_proprietary
    elif features_csv:
        csv_path = features_csv
    else:
        csv_path = default_csv
    return dataset_suite(family, scale_from_features_path(csv_path))


def apps_output_dir(root: str | Path, rq_id: str, suite: str) -> Path:
    """Bundled experiment JSON: {root}/{rq_id}/apps/{suite}/."""
    return Path(root) / rq_id / "apps" / suite


def rc_output_dir(root: str | Path, rq_id: str, suite: str) -> Path:
    """Ranking-criteria artifacts: {root}/{rq_id}/rc/{suite}/."""
    return Path(root) / rq_id / "rc" / suite


def rc_wo_id_csv_path(root: str | Path, rq_id: str, suite: str) -> Path:
    return rc_output_dir(root, rq_id, suite) / f"rc_wo_id_{suite}.csv"


def merged_rc_csv_path(root: str | Path) -> Path:
    """Cross-suite unified criteria from rq1/rc/merge/."""
    return Path(root) / "rq1" / "rc" / "merge" / "rc_merge_unified.csv"


def resolve_criteria_csv(
    *,
    output_root: str | Path,
    suite: str,
    criteria_csv_override: str | None = None,
    default_csv: str | None = None,
) -> str | None:
    """
    Resolve RQ3 ranking criteria CSV.

    Priority: explicit override → RQ3 default (usually merged unified) →
    suite-specific rq1/rc/{suite}/rc_wo_id_{suite}.csv.
    """
    if criteria_csv_override:
        return criteria_csv_override
    if default_csv and Path(default_csv).is_file():
        return default_csv
    merged = merged_rc_csv_path(output_root)
    if merged.is_file():
        return str(merged)
    suite_path = rc_wo_id_csv_path(output_root, "rq1", suite)
    if suite_path.is_file():
        return str(suite_path)
    return None


def family_output_dir(root: str, rq_id: str, family: str) -> Path:
    """Legacy layout: {root}/{rq_id}/{family}/ (prefer apps_output_dir)."""
    return Path(root) / rq_id / family


def build_bundle_filename(
    *,
    family: str,
    provider: str,
    model_key: str,
    mode: str,
    k: int,
) -> str:
    return f"{family}_{provider}_{model_key}_{mode}_k{k}_ALL.json"


_K_MODE_SUFFIX = re.compile(r"^(?P<body>.+)_(?P<mode>structured|prompt)_k(?P<k>\d+)$")


def parse_bundle_filename(filename: str) -> dict[str, str | int]:
    """
    Parse bundled experiment output filenames:
    {family}_{provider}_{modelKey}_{mode}_k{k}_ALL.json

    modelKey may contain underscores (e.g. llama31_8b).
    """
    stem = Path(filename).stem
    if not stem.endswith("_ALL"):
        raise ValueError(f"Expected bundled filename ending with _ALL: {filename}")

    match = _K_MODE_SUFFIX.match(stem[:-4])  # strip trailing _ALL
    if not match:
        raise ValueError(f"Unexpected bundled filename format: {filename}")

    mode = match.group("mode")
    k = int(match.group("k"))
    body = match.group("body")

    family, rest = body.split("_", 1)
    provider, model_key = rest.split("_", 1)

    return {
        "family": family,
        "provider": provider,
        "model": model_key,
        "model_key": model_key,
        "mode": mode,
        "k": k,
        "feature": "",
        "criterion": None,
        "run": -1,
        "prefix": body,
    }
