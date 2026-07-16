from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

from code.experiments.providers.base import LLMClient
from code.experiments.structured_output import apply_gemini_structured, require_schema

load_dotenv()


class GeminiClient(LLMClient):
    def __init__(self, model: str, *, web_search: bool = False):
        super().__init__(model, web_search=web_search)
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
            "response_modalities": ["TEXT"],
            "candidate_count": 1,
        }
        if self.web_search:
            config_kwargs["tools"] = [Tool(google_search=GoogleSearch())]
        if require_schema(structured, schema) is not None:
            apply_gemini_structured(config_kwargs, schema=schema)

        config = GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return response.text or ""
