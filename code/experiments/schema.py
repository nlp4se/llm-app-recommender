from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_base_schema(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_k_constraints(schema: dict[str, Any], k: int) -> dict[str, Any]:
    """Inject minItems/maxItems on the ranked-apps array for exactly k apps."""
    result = copy.deepcopy(schema)
    apps = result.get("properties", {}).get("a")
    if apps is not None:
        apps["minItems"] = k
        apps["maxItems"] = k
    return result


def openai_response_format(schema: dict[str, Any], name: str = "rank_apps") -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def mistral_response_format(schema: dict[str, Any], name: str = "rank_apps") -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def perplexity_response_format(schema: dict[str, Any], name: str = "rank_apps") -> dict[str, Any]:
    return openai_response_format(schema, name)


def anthropic_output_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "schema": schema,
    }


def load_schema_for_run(schema_base_path: str | Path, k: int) -> dict[str, Any]:
    base = load_base_schema(schema_base_path)
    return apply_k_constraints(base, k)
