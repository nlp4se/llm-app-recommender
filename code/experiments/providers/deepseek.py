from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from code.experiments.config import Provider
from code.experiments.providers.base import LLMClient
from code.experiments.structured_output import (
    augment_system_prompt_for_deepseek_json,
    require_schema,
)

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(LLMClient):
    """
    DeepSeek chat API (OpenAI-compatible) with JSON Output mode.

    Uses ``response_format: {type: json_object}`` plus schema guidance in the
    system prompt (required by DeepSeek). Post-response validation uses jsonschema.
    """

    provider = Provider.DEEPSEEK

    def __init__(self, model: str, *, web_search: bool = False):
        super().__init__(model, web_search=web_search)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not found")
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        system = system_prompt
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 20000,
        }
        if require_schema(structured, schema) is not None:
            kwargs["response_format"] = {"type": "json_object"}
            system = augment_system_prompt_for_deepseek_json(system, schema)

        kwargs["messages"][0]["content"] = system
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
