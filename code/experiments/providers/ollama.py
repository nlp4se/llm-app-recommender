from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

from code.experiments.config import ModelSpec, OllamaSettings
from code.experiments.providers.base import LLMClient

load_dotenv()


class OllamaClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec.model_id)
        self.spec = spec
        self.ollama = spec.ollama or OllamaSettings()
        base = os.getenv("OLLAMA_BASE_URL", self.ollama.base_url).rstrip("/")
        self.chat_url = f"{base}/api/chat"
        self.session = requests.Session()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        system = system_prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"num_predict": self.ollama.max_tokens},
        }

        if structured:
            if schema is None:
                raise ValueError("schema is required for structured output")
            # Prompt with a concrete example; dumping JSON Schema confuses many local models.
            k = schema.get("properties", {}).get("a", {}).get("minItems", 20)
            system = (
                f"{system}\n\n"
                "Return ONLY valid JSON in exactly this shape (no extra keys, no JSON Schema):\n"
                "{\n"
                f'  "a": ["App 1", "App 2", ... exactly {k} app names as plain strings],\n'
                '  "c": [{"n": "Criterion name", "d": "Criterion description"}]\n'
                "}\n"
                "Rules:\n"
                "- Top-level keys must be only: a, c\n"
                '- "a" must be an array of strings (app names), not objects\n'
                '- "c" must be an array of objects with keys n and d only'
            )
            payload["messages"][0]["content"] = system
            payload["format"] = "json"

        response = self.session.post(self.chat_url, json=payload, timeout=300)
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {response.text}")

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return content
