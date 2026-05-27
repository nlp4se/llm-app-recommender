from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from code.experiments.providers.base import LLMClient

load_dotenv()


class HuggingFaceClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(model)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is required for open models")
        self.client = InferenceClient(api_key=token)

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
            "max_tokens": 20000,
        }
        if structured and schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rank_apps", "schema": schema},
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError:
            # Some hosted models do not accept response_format yet.
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
