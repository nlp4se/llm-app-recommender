from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral

from code.experiments.providers.base import LLMClient
from code.experiments.schema import mistral_response_format

load_dotenv()


class MistralClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

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
            "tools": [{"type": "web_search"}],
        }
        if structured:
            if schema is None:
                raise ValueError("schema is required for structured output")
            kwargs["response_format"] = mistral_response_format(schema)

        response = self.client.chat.complete(**kwargs)
        return response.choices[0].message.content or ""
