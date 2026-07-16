from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from code.experiments.config import Provider
from code.experiments.providers.base import LLMClient
from code.experiments.structured_output import apply_openai_compatible_structured, require_schema

load_dotenv()


class OpenAIClient(LLMClient):
    provider = Provider.OPENAI

    def __init__(self, model: str, *, web_search: bool = False):
        super().__init__(model, web_search=web_search)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if require_schema(structured, schema) is not None:
            apply_openai_compatible_structured(kwargs, provider=self.provider, schema=schema)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
