
from app.gpt.base import GPT
from app.gpt.model_capabilities import infer_model_capabilities
from app.gpt.provider.OpenAI_compatible_provider import (
    OpenAICompatibleProvider,
    normalize_openai_base_url,
)
from app.gpt.universal_gpt import UniversalGPT
from app.models.model_config import ModelConfig


class GPTFactory:
    @staticmethod
    def from_config(config: ModelConfig) -> GPT:
        client = OpenAICompatibleProvider(
            api_key=config.api_key,
            base_url=normalize_openai_base_url(config.base_url),
        ).get_client
        provider_name = (config.name or "").lower()
        model_name = (config.model_name or "").lower()
        capabilities = infer_model_capabilities(provider_name, model_name)
        return UniversalGPT(client=client, model=config.model_name, supports_vision=capabilities.supports_vision)

    @staticmethod
    def _supports_vision(provider_name: str, model_name: str) -> bool:
        return infer_model_capabilities(provider_name, model_name).supports_vision
