from __future__ import annotations

import json
import re
from pathlib import Path

from jsonschema import ValidationError, validate


def sanitize_response_text(text: str) -> str:
    """Strip markdown fences and surrounding whitespace from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def save_json_response(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_json_response(content: str) -> dict:
    cleaned = sanitize_response_text(content)
    return json.loads(cleaned)


def validate_json_response(payload: dict, schema: dict) -> None:
    validate(instance=payload, schema=schema)


def parse_and_validate_response(content: str, schema: dict) -> dict:
    payload = parse_json_response(content)
    try:
        validate_json_response(payload, schema)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc.message}") from exc
    return payload


def list_experiment_json_files(directory: str | Path) -> list[Path]:
    """Return JSON files directly under an RQ output folder."""
    folder = Path(directory)
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.json") if p.is_file())
