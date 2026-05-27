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


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: Family
    provider: Provider
    model_id: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "openai": ModelSpec("openai", "proprietary", Provider.OPENAI, "gpt-5.3-chat-latest"),
    "gemini": ModelSpec("gemini", "proprietary", Provider.GEMINI, "gemini-3-flash-preview"),
    "anthropic": ModelSpec("anthropic", "proprietary", Provider.ANTHROPIC, "claude-opus-4-6-thinking"),
    "mistral": ModelSpec("mistral", "proprietary", Provider.MISTRAL, "mistral-large-latest"),
    "perplexity": ModelSpec("perplexity", "proprietary", Provider.PERPLEXITY, "perplexity/sonar"),
    "llama4scout": ModelSpec("llama4scout", "open", Provider.HUGGINGFACE, "meta-llama/Llama-4-Scout-17B-16E"),
    "gemma4": ModelSpec("gemma4", "open", Provider.HUGGINGFACE, "google/gemma-4-31B"),
    "qwen3": ModelSpec("qwen3", "open", Provider.HUGGINGFACE, "Qwen/Qwen3-30B-A3B"),
    "gptoss20b": ModelSpec("gptoss20b", "open", Provider.HUGGINGFACE, "openai/gpt-oss-20b"),
    "mistralsmall31": ModelSpec("mistralsmall31", "open", Provider.HUGGINGFACE, "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
    "deepseekv3": ModelSpec("deepseekv3", "open", Provider.HUGGINGFACE, "deepseek-ai/DeepSeek-V3"),
}

DEFAULT_MODEL_KEYS_BY_FAMILY: dict[Family, list[str]] = {
    "proprietary": ["openai", "gemini", "anthropic", "mistral", "perplexity"],
    "open": ["llama4scout", "gemma4", "qwen3", "gptoss20b", "mistralsmall31", "deepseekv3"],
}

ENV_KEYS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.MISTRAL: "MISTRAL_API_KEY",
    Provider.PERPLEXITY: "PERPLEXITY_API_KEY",
    Provider.HUGGINGFACE: "HF_TOKEN",
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
