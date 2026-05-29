from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from code.experiments.config import HuggingFaceSettings, ModelSpec, hf_inference_model_id
from code.experiments.providers.base import LLMClient

load_dotenv()


class HuggingFaceClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec.model_id)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is required for open models")
        self.spec = spec
        self.hf = spec.hf or HuggingFaceSettings()
        self.api_model = hf_inference_model_id(spec)
        self.client = InferenceClient(
            api_key=token,
            provider=self.hf.provider,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        system = system_prompt
        if structured and schema is not None and not self.hf.use_json_schema:
            system += (
                "\n\nReturn ONLY valid JSON that follows this schema exactly:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )

        kwargs: dict[str, Any] = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.hf.max_tokens,
        }
        if structured and schema is not None and self.hf.use_json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rank_apps", "schema": schema},
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not (structured and kwargs.get("response_format") is not None):
                raise
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
