from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

from code.experiments.providers.base import LLMClient
from code.experiments.structured_output import apply_perplexity_structured, require_schema

load_dotenv()

PERPLEXITY_AGENT_URL = "https://api.perplexity.ai/v1/agent"
# Sonar models are Agent API presets, not ``model`` ids (see Perplexity output-control docs).
_SONAR_PRESETS = frozenset(
    {"sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research", "fast-search", "pro-search"}
)
logger = logging.getLogger(__name__)


def _agent_target(model_id: str) -> dict[str, str]:
    if model_id.startswith("preset:"):
        return {"preset": model_id.removeprefix("preset:")}
    if model_id in _SONAR_PRESETS:
        return {"preset": model_id}
    return {"model": model_id}


def _extract_output_text(data: dict[str, Any]) -> str:
    text = data.get("output_text")
    if isinstance(text, str) and text:
        return text
    parts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "".join(parts)


class PerplexityClient(LLMClient):
    """Perplexity Agent API (``/v1/agent``) with optional JSON-schema structured output."""

    def __init__(self, model: str, *, web_search: bool = False):
        super().__init__(model, web_search=web_search)
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not found")
        self.api_key = api_key

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            **_agent_target(self.model),
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": 20000,
        }
        if self.web_search:
            payload["tools"] = [{"type": "web_search"}]
        if require_schema(structured, schema) is not None:
            apply_perplexity_structured(payload, schema=schema)

        response = requests.post(
            PERPLEXITY_AGENT_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=600,
        )
        if not response.ok:
            detail = response.text[:500]
            raise requests.HTTPError(
                f"{response.status_code} Client Error for {PERPLEXITY_AGENT_URL}: {detail}",
                response=response,
            )

        data = response.json()
        if data.get("status") == "failed":
            raise RuntimeError(f"Perplexity Agent API failed: {data.get('error')}")

        return _extract_output_text(data)
