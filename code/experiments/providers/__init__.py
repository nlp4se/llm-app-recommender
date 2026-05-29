from code.experiments.config import ModelSpec, Provider
from code.experiments.providers.base import LLMClient


def get_client(spec: ModelSpec) -> LLMClient:
    if spec.provider == Provider.OPENAI:
        from code.experiments.providers.openai import OpenAIClient

        return OpenAIClient(spec.model_id)
    if spec.provider == Provider.GEMINI:
        from code.experiments.providers.gemini import GeminiClient

        return GeminiClient(spec.model_id)
    if spec.provider == Provider.ANTHROPIC:
        from code.experiments.providers.anthropic import AnthropicClient

        return AnthropicClient(spec.model_id)
    if spec.provider == Provider.MISTRAL:
        from code.experiments.providers.mistral import MistralClient

        return MistralClient(spec.model_id)
    if spec.provider == Provider.PERPLEXITY:
        from code.experiments.providers.perplexity import PerplexityClient

        return PerplexityClient(spec.model_id)
    if spec.provider == Provider.HUGGINGFACE:
        from code.experiments.providers.huggingface import HuggingFaceClient

        return HuggingFaceClient(spec)
    if spec.provider == Provider.OLLAMA:
        from code.experiments.providers.ollama import OllamaClient

        return OllamaClient(spec)
    raise ValueError(f"Unknown provider: {spec.provider}")


__all__ = ["get_client", "LLMClient"]
