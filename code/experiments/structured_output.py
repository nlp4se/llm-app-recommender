"""
Provider-native structured output wiring for experiment runs.

Each proprietary client uses the vendor's documented JSON-schema mechanism when
``structured=True``. Post-response validation always uses the full schema from
``load_schema_for_run`` plus ``parse_and_validate_response``.

References:
- OpenAI: https://developers.openai.com/api/docs/guides/structured-outputs
- Gemini: https://ai.google.dev/gemini-api/docs/structured-output
- Anthropic: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Mistral: https://docs.mistral.ai/studio-api/conversations/structured-output
- Perplexity Agent API: https://docs.perplexity.ai/docs/agent-api/output-control
  (``POST /v1/agent`` with ``preset`` for Sonar models; ``response_format`` json_schema)
- DeepSeek JSON Output: https://api-docs.deepseek.com/guides/json_mode
  (``response_format: {type: json_object}`` + ``json`` keyword and schema in prompt)
"""

from __future__ import annotations

import json
from typing import Any

from code.experiments.config import Provider
from code.experiments.schema import adapt_schema_for_provider

SCHEMA_NAME = "rank_apps"


class StructuredOutputError(ValueError):
    """Raised when structured output is requested without a schema."""


def require_schema(structured: bool, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if not structured:
        return None
    if schema is None:
        raise StructuredOutputError("schema is required when structured=True")
    return schema


def openai_json_schema_format(provider: Provider, schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-compatible ``response_format`` (OpenAI, Mistral, Perplexity Sonar)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": SCHEMA_NAME,
            "strict": True,
            "schema": adapt_schema_for_provider(provider, schema),
        },
    }


def anthropic_output_config(schema: dict[str, Any]) -> dict[str, Any]:
    """Anthropic ``output_config.format`` JSON schema block."""
    return {
        "format": {
            "type": "json_schema",
            "schema": adapt_schema_for_provider(Provider.ANTHROPIC, schema),
        }
    }


def gemini_json_config(schema: dict[str, Any]) -> dict[str, str | dict[str, Any]]:
    """Gemini ``generateContent`` JSON response settings."""
    return {
        "response_mime_type": "application/json",
        "response_schema": adapt_schema_for_provider(Provider.GEMINI, schema),
    }


def perplexity_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Perplexity Agent API ``response_format`` (no ``strict`` flag in docs)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": SCHEMA_NAME,
            "schema": adapt_schema_for_provider(Provider.PERPLEXITY, schema),
        },
    }


def apply_perplexity_structured(payload: dict[str, Any], *, schema: dict[str, Any]) -> None:
    payload["response_format"] = perplexity_response_format(schema)


def apply_openai_compatible_structured(
    payload: dict[str, Any],
    *,
    provider: Provider,
    schema: dict[str, Any],
) -> None:
    payload["response_format"] = openai_json_schema_format(provider, schema)


def apply_anthropic_structured(payload: dict[str, Any], *, schema: dict[str, Any]) -> None:
    payload["output_config"] = anthropic_output_config(schema)


def apply_gemini_structured(config: dict[str, Any], *, schema: dict[str, Any]) -> None:
    config.update(gemini_json_config(schema))


def augment_system_prompt_for_deepseek_json(system_prompt: str, schema: dict[str, Any]) -> str:
    """
    DeepSeek JSON Output requires the word ``json`` and an example schema in the prompt.
    """
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    return (
        f"{system_prompt.rstrip()}\n\n"
        "Return your answer as valid JSON (json format) only — no markdown, no prose.\n"
        "The JSON object must match this schema:\n"
        f"{schema_text}"
    )
