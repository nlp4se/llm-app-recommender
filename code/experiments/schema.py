from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from code.experiments.config import Provider

# Keywords rejected by specific provider structured-output APIs.
# Local validation (jsonschema + io.py) still uses the full schema.
_OPENAI_UNSUPPORTED = frozenset({"uniqueItems"})
_GEMINI_UNSUPPORTED = frozenset({"additionalProperties", "uniqueItems"})


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


def load_schema_for_run(schema_base_path: str | Path, k: int) -> dict[str, Any]:
    """Full JSON Schema used for post-response validation."""
    return apply_k_constraints(load_base_schema(schema_base_path), k)


def _prune_schema_tree(schema: dict[str, Any], *, remove: frozenset[str]) -> dict[str, Any]:
    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: walk(value) for key, value in obj.items() if key not in remove}
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        return obj

    return walk(copy.deepcopy(schema))


def _strip_apps_array_bounds(schema: dict[str, Any]) -> dict[str, Any]:
    """Anthropic only allows minItems 0 or 1 on arrays; k bounds are enforced locally."""
    result = copy.deepcopy(schema)
    apps = result.get("properties", {}).get("a")
    if isinstance(apps, dict):
        for key in ("minItems", "maxItems", "uniqueItems"):
            apps.pop(key, None)
    return result


def adapt_schema_for_provider(provider: Provider, schema: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``schema`` suitable for a provider's structured-output API.

    The experiment runner validates responses against the unmodified schema from
    ``load_schema_for_run`` (including k bounds and duplicate checks).
    """
    if provider == Provider.GEMINI:
        return _prune_schema_tree(schema, remove=_GEMINI_UNSUPPORTED)
    if provider == Provider.ANTHROPIC:
        return _strip_apps_array_bounds(schema)
    if provider in (Provider.OPENAI, Provider.MISTRAL, Provider.PERPLEXITY):
        return _prune_schema_tree(schema, remove=_OPENAI_UNSUPPORTED)
    return copy.deepcopy(schema)
