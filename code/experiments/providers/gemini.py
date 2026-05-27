from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

from code.experiments.providers.base import LLMClient

load_dotenv()


class GeminiClient(LLMClient):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "tools": [Tool(google_search=GoogleSearch())],
            "response_modalities": ["TEXT"],
            "candidate_count": 1,
        }
        if structured:
            if schema is None:
                raise ValueError("schema is required for structured output")
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema

        config = GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return response.text or ""
