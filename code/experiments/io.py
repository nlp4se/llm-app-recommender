from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from code.experiments.naming import parse_bundle_filename


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


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
        elif isinstance(item, dict):
            if len(item) == 1:
                out.append(str(next(iter(item))).strip())
            elif "name" in item:
                out.append(str(item["name"]).strip())
            elif "n" in item:
                out.append(str(item["n"]).strip())
        elif item is not None:
            out.append(str(item).strip())
    return out


def _coerce_criteria_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("n") or item.get("name")
        desc = item.get("d") or item.get("description")
        if name is not None and desc is not None:
            out.append({"n": str(name).strip(), "d": str(desc).strip()})
    return out


def _unwrap_payload(payload: dict[str, Any], prop_keys: list[str]) -> dict[str, Any]:
    """Hoist fields nested under a JSON Schema-style `properties` object."""
    if any(k in payload for k in prop_keys):
        return payload
    nested = payload.get("properties")
    if isinstance(nested, dict) and any(k in nested for k in prop_keys):
        return nested
    return payload


def normalize_payload_for_schema(payload: dict, schema: dict) -> dict:
    """Repair common local-model JSON shape mistakes before schema validation."""
    if not isinstance(payload, dict):
        return payload

    prop_keys = list(schema.get("properties", {}).keys())
    if not prop_keys:
        return payload

    payload = _unwrap_payload(payload, prop_keys)
    required = schema.get("required", prop_keys)
    extracted: dict[str, Any] = {}

    for key in prop_keys:
        if key not in payload:
            continue
        value = payload[key]
        if key == "a":
            extracted[key] = _coerce_string_list(value)
        elif key == "c":
            extracted[key] = _coerce_criteria_list(value)
        else:
            extracted[key] = value

    if all(k in extracted for k in required):
        return extracted
    return payload


def validate_json_response(payload: dict, schema: dict) -> None:
    validate(instance=payload, schema=schema)


def parse_and_validate_response(content: str, schema: dict) -> dict:
    payload = normalize_payload_for_schema(parse_json_response(content), schema)
    try:
        validate_json_response(payload, schema)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc.message}") from exc
    return payload


def list_experiment_json_files(directory: str | Path) -> list[Path]:
    """Return bundled experiment JSON files (*_ALL.json) in an RQ output folder."""
    folder = Path(directory)
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*_ALL.json") if p.is_file())


def expand_bundled_file(file_path: str | Path, rq_id: str) -> list[dict[str, Any]]:
    """
    Expand one bundled output file into flat rows for downstream CSV/analysis.

    Each row contains metadata + payload (validated ranking JSON for one feature[/criterion]/run).
    """
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("records"), list):
        raise ValueError(f"Not a bundled experiment file: {path}")

    file_meta = parse_bundle_filename(path.name)
    family = str(data.get("family", file_meta["family"]))
    provider = str(data.get("provider", file_meta["provider"]))
    model_key = str(data.get("model_key", file_meta["model"]))
    mode = str(data.get("mode", file_meta["mode"]))
    k = int(data.get("k", file_meta["k"]))

    rows: list[dict[str, Any]] = []
    for record in data["records"]:
        if not isinstance(record, dict):
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        feature = record.get("feature", "")
        run = record.get("run", -1)
        criterion = record.get("criterion")
        rows.append(
            {
                "family": family,
                "model": model_key,
                "provider": provider,
                "mode": mode,
                "k": k,
                "feature": feature,
                "run": run,
                "criterion": criterion,
                "prefix": f"{family}_{provider}_{model_key}_{mode}_k{k}_{feature}_{criterion or 'na'}_{run}",
                "json_data": payload,
                "file_path": str(path),
                "rq": rq_id,
            }
        )
    return rows


def expand_bundled_folders(input_folders: list[str], rq_id: str) -> tuple[list[dict[str, Any]], int]:
    """Load all bundled files from one or more RQ family folders."""
    all_rows: list[dict[str, Any]] = []
    failed = 0
    for input_folder in input_folders:
        folder = Path(input_folder)
        if not folder.exists():
            print(f"Warning: Input folder {input_folder} does not exist. Skipping.")
            continue
        for file_path in list_experiment_json_files(folder):
            try:
                all_rows.extend(expand_bundled_file(file_path, rq_id))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Error reading {file_path}: {exc}")
                failed += 1
    return all_rows, failed
