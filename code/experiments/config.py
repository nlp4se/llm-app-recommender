from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

OutputMode = Literal["structured", "prompt"]
Family = Literal["proprietary", "open"]


class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class HuggingFaceSettings:
    """Per-model Hugging Face Inference Providers routing."""

    inference_model_id: str | None = None
    provider: str | None = None
    use_json_schema: bool = True
    max_tokens: int = 20000


@dataclass(frozen=True)
class OllamaSettings:
    """Per-model local Ollama runtime settings."""

    base_url: str = "http://localhost:11434"
    use_json_schema: bool = True
    max_tokens: int = 8192


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: Family
    provider: Provider
    model_id: str
    hf: HuggingFaceSettings | None = None
    ollama: OllamaSettings | None = None


def hf_inference_model_id(spec: ModelSpec) -> str:
    hf = spec.hf or HuggingFaceSettings()
    return hf.inference_model_id or spec.model_id


def hf_api_model_id(spec: ModelSpec) -> str:
    model = hf_inference_model_id(spec)
    hf = spec.hf or HuggingFaceSettings()
    if hf.provider:
        return f"{model}@{hf.provider}"
    return model


MODEL_SPECS: dict[str, ModelSpec] = {
    "openai": ModelSpec("openai", "proprietary", Provider.OPENAI, "gpt-5.3-chat-latest"),
    "gemini": ModelSpec("gemini", "proprietary", Provider.GEMINI, "gemini-3-flash-preview"),
    "anthropic": ModelSpec("anthropic", "proprietary", Provider.ANTHROPIC, "claude-opus-4-6-thinking"),
    "mistral": ModelSpec("mistral", "proprietary", Provider.MISTRAL, "mistral-large-latest"),
    "perplexity": ModelSpec("perplexity", "proprietary", Provider.PERPLEXITY, "perplexity/sonar"),
    "llama31_8b": ModelSpec(
        "llama31_8b",
        "open",
        Provider.OLLAMA,
        "llama3.1:8b",
        ollama=OllamaSettings(use_json_schema=False, max_tokens=8192),
    ),
    "gemma3_4b": ModelSpec(
        "gemma3_4b",
        "open",
        Provider.OLLAMA,
        "gemma3:4b",
        ollama=OllamaSettings(use_json_schema=False, max_tokens=8192),
    ),
    "qwen3_8b": ModelSpec(
        "qwen3_8b",
        "open",
        Provider.OLLAMA,
        "qwen3:8b",
        ollama=OllamaSettings(use_json_schema=False, max_tokens=8192),
    ),
    "gptoss20b": ModelSpec(
        "gptoss20b",
        "open",
        Provider.OLLAMA,
        "gpt-oss:20b",
        ollama=OllamaSettings(use_json_schema=False, max_tokens=8192),
    ),
    "mistral_open": ModelSpec(
        "mistral_open",
        "open",
        Provider.OLLAMA,
        "mistral",
        ollama=OllamaSettings(use_json_schema=False, max_tokens=8192),
    ),
    "deepseekr1_8b": ModelSpec(
        "deepseekr1_8b",
        "open",
        Provider.OLLAMA,
        "deepseek-r1:8b",
        ollama=OllamaSettings(use_json_schema=False, max_tokens=8192),
    ),
}

DEFAULT_MODEL_KEYS_BY_FAMILY: dict[Family, list[str]] = {
    "proprietary": ["openai", "gemini", "anthropic", "mistral", "perplexity"],
    "open": ["llama31_8b", "gemma3_4b", "qwen3_8b", "gptoss20b", "mistral_open", "deepseekr1_8b"],
}

ENV_KEYS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.MISTRAL: "MISTRAL_API_KEY",
    Provider.PERPLEXITY: "PERPLEXITY_API_KEY",
    Provider.HUGGINGFACE: "HF_TOKEN",
    Provider.OLLAMA: "OLLAMA_BASE_URL",
}


@dataclass(frozen=True)
class RQConfig:
    id: str
    user_prompt: str
    system_prompt_structured: str
    system_prompt_output: str
    schema_base: str
    search_items_csv: str
    runs_per_item: int
    default_criteria_csv: str | None = None


RQ1 = RQConfig(
    id="rq1",
    user_prompt="data/input/prompts/user-prompt-feature-rq1.txt",
    system_prompt_structured="data/input/prompts/system-prompt-rq1.txt",
    system_prompt_output="data/input/prompts/system-prompt-output-rq1.txt",
    schema_base="data/input/schema/rq1.base.json",
    search_items_csv="data/input/use-case/features.csv",
    runs_per_item=10,
)

RQ3 = RQConfig(
    id="rq3",
    user_prompt="data/input/prompts/user-prompt-feature-rq3.txt",
    system_prompt_structured="data/input/prompts/system-prompt-rq3.txt",
    system_prompt_output="data/input/prompts/system-prompt-output-rq3.txt",
    schema_base="data/input/schema/rq3.base.json",
    search_items_csv="data/input/use-case/features.csv",
    runs_per_item=4,
    default_criteria_csv="data/output/features/rq1/rc_wo_id.csv",
)

RQ_CONFIGS: dict[str, RQConfig] = {
    "rq1": RQ1,
    "rq3": RQ3,
}

K_VALUES_CSV = "data/input/use-case/k.csv"
OUTPUT_ROOT = "data/output/features"
MAX_ATTEMPTS_PER_RUN = 3
