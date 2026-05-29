from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

FEATURES_CSV = "data/input/use-case/features.csv"


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


def family_output_dir(root: str, rq_id: str, family: str) -> Path:
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


def parse_bundle_filename(filename: str) -> dict[str, str | int]:
    """
    Parse bundled experiment output filenames:
    {family}_{provider}_{modelKey}_{mode}_k{k}_ALL.json
    """
    stem = Path(filename).stem
    if not stem.endswith("_ALL"):
        raise ValueError(f"Expected bundled filename ending with _ALL: {filename}")
    parts = stem[:-4].split("_")  # drop trailing _ALL
    if len(parts) < 5:
        raise ValueError(f"Unexpected bundled filename format: {filename}")

    family = parts[0]
    provider = parts[1]
    model_key = parts[2]
    mode = parts[3]
    k_token = parts[4]
    if not k_token.startswith("k") or not k_token[1:].isdigit():
        raise ValueError(f"Invalid k token in filename: {filename}")
    k = int(k_token[1:])

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
        "prefix": "_".join(parts),
    }
