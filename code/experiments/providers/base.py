from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """
    Provider client for experiment API calls.

    Proprietary runs always call ``complete(..., structured=True, schema=...)``.
    Each implementation must use the vendor's native JSON-schema structured output
    API (see ``code.experiments.structured_output``).
    """

    def __init__(self, model: str, *, web_search: bool = False):
        self.model = model
        self.web_search = web_search

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Return raw text content from the model."""
        pass
