from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from code.experiments.providers.base import LLMClient
from code.experiments.schema import perplexity_response_format

load_dotenv()

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(model)
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
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 20000,
        }
        if structured:
            if schema is None:
                raise ValueError("schema is required for structured output")
            payload["response_format"] = perplexity_response_format(schema)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""
