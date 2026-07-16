from __future__ import annotations

import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from code.experiments.providers.base import LLMClient
from code.experiments.structured_output import apply_anthropic_structured, require_schema

load_dotenv()


class AnthropicClient(LLMClient):
    def __init__(self, model: str, *, web_search: bool = False):
        super().__init__(model, web_search=web_search)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 20000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if require_schema(structured, schema) is not None:
            apply_anthropic_structured(kwargs, schema=schema)

        response = self.client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
