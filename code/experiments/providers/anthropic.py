from __future__ import annotations

import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from code.experiments.providers.base import LLMClient
from code.experiments.schema import anthropic_output_format

load_dotenv()


class AnthropicClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(model)
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
        if structured:
            if schema is None:
                raise ValueError("schema is required for structured output")
            kwargs["output_config"] = {
                "format": anthropic_output_format(schema),
            }

        response = self.client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
