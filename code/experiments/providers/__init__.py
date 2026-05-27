from code.experiments.config import Provider
from code.experiments.providers.base import LLMClient


def get_client(provider: Provider, model: str) -> LLMClient:
    if provider == Provider.OPENAI:
        from code.experiments.providers.openai import OpenAIClient

        return OpenAIClient(model)
    if provider == Provider.GEMINI:
        from code.experiments.providers.gemini import GeminiClient

        return GeminiClient(model)
    if provider == Provider.ANTHROPIC:
        from code.experiments.providers.anthropic import AnthropicClient

        return AnthropicClient(model)
    if provider == Provider.MISTRAL:
        from code.experiments.providers.mistral import MistralClient

        return MistralClient(model)
    if provider == Provider.PERPLEXITY:
        from code.experiments.providers.perplexity import PerplexityClient

        return PerplexityClient(model)
    if provider == Provider.HUGGINGFACE:
        from code.experiments.providers.huggingface import HuggingFaceClient

        return HuggingFaceClient(model)
    raise ValueError(f"Unknown provider: {provider}")


__all__ = ["get_client", "LLMClient"]
