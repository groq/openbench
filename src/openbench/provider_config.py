"""
Centralized configuration for model providers.

This module provides a unified configuration system for all supported model providers,
including their API keys, base URLs, and provider-specific settings.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ProviderType(str, Enum):
    """Supported model provider types."""

    AI21 = "ai21"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BASETEN = "baseten"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    CRUSOE = "crusoe"
    DEEPINFRA = "deepinfra"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    FRIENDLI = "friendli"
    GOOGLE = "google"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    HYPERBOLIC = "hyperbolic"
    LAMBDA = "lambda"
    MINIMAX = "minimax"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    NOUS = "nous"
    NOVITA = "novita"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    PARASAIL = "parasail"
    PERPLEXITY = "perplexity"
    REKA = "reka"
    SAMBANOVA = "sambanova"
    SILICONFLOW = "siliconflow"
    TOGETHER = "together"
    VERCEL = "vercel"
    WANDB = "wandb"
    XAI = "xai"


@dataclass
class ProviderConfig:
    """Configuration for a specific model provider."""

    name: str
    display_name: str
    api_key_env: str
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    additional_env_vars: Optional[List[str]] = None
    supports_vision: bool = False
    supports_function_calling: bool = True
    requires_auth: bool = True

    def get_api_key(self) -> Optional[str]:
        """Get the API key from environment variables."""
        return os.getenv(self.api_key_env)

    def get_base_url(self) -> Optional[str]:
        """Get the base URL, checking environment variable if specified."""
        if self.base_url_env:
            env_url = os.getenv(self.base_url_env)
            if env_url:
                return env_url
        return self.base_url

    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        if not self.requires_auth:
            return True
        return self.get_api_key() is not None

    def get_all_env_vars(self) -> List[str]:
        """Get all environment variables related to this provider."""
        env_vars = [self.api_key_env]
        if self.base_url_env:
            env_vars.append(self.base_url_env)
        if self.additional_env_vars:
            env_vars.extend(self.additional_env_vars)
        return env_vars


# Centralized provider configurations
PROVIDER_CONFIGS: Dict[ProviderType, ProviderConfig] = {
    ProviderType.AI21: ProviderConfig(
        name="ai21",
        display_name="AI21 Labs",
        api_key_env="AI21_API_KEY",
        base_url="https://api.ai21.com/studio/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.ANTHROPIC: ProviderConfig(
        name="anthropic",
        display_name="Anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.AZURE: ProviderConfig(
        name="azure",
        display_name="Azure OpenAI",
        api_key_env="AZURE_OPENAI_API_KEY",
        additional_env_vars=["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_VERSION"],
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.BASETEN: ProviderConfig(
        name="baseten",
        display_name="Baseten",
        api_key_env="BASETEN_API_KEY",
        base_url="https://api.baseten.co/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.CEREBRAS: ProviderConfig(
        name="cerebras",
        display_name="Cerebras",
        api_key_env="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.COHERE: ProviderConfig(
        name="cohere",
        display_name="Cohere",
        api_key_env="COHERE_API_KEY",
        base_url="https://api.cohere.ai/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.CRUSOE: ProviderConfig(
        name="crusoe",
        display_name="Crusoe",
        api_key_env="CRUSOE_API_KEY",
        base_url="https://api.crusoe.ai/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.DEEPINFRA: ProviderConfig(
        name="deepinfra",
        display_name="DeepInfra",
        api_key_env="DEEPINFRA_API_KEY",
        base_url="https://api.deepinfra.com/v1/openai",
        base_url_env="DEEPINFRA_BASE_URL",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.DEEPSEEK: ProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.FIREWORKS: ProviderConfig(
        name="fireworks",
        display_name="Fireworks AI",
        api_key_env="FIREWORKS_API_KEY",
        base_url="https://api.fireworks.ai/inference/v1",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.FRIENDLI: ProviderConfig(
        name="friendli",
        display_name="Friendli",
        api_key_env="FRIENDLI_TOKEN",
        base_url="https://inference.friendli.ai/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.GOOGLE: ProviderConfig(
        name="google",
        display_name="Google AI",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.GROQ: ProviderConfig(
        name="groq",
        display_name="Groq",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.HUGGINGFACE: ProviderConfig(
        name="huggingface",
        display_name="Hugging Face",
        api_key_env="HF_TOKEN",
        base_url="https://api-inference.huggingface.co",
        supports_vision=True,
        supports_function_calling=False,
    ),
    ProviderType.HYPERBOLIC: ProviderConfig(
        name="hyperbolic",
        display_name="Hyperbolic",
        api_key_env="HYPERBOLIC_API_KEY",
        base_url="https://api.hyperbolic.xyz/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.LAMBDA: ProviderConfig(
        name="lambda",
        display_name="Lambda",
        api_key_env="LAMBDA_API_KEY",
        base_url="https://api.lambdalabs.com/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.MINIMAX: ProviderConfig(
        name="minimax",
        display_name="MiniMax",
        api_key_env="MINIMAX_API_KEY",
        base_url="https://api.minimax.io/v1",
        base_url_env="MINIMAX_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.MISTRAL: ProviderConfig(
        name="mistral",
        display_name="Mistral AI",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.MOONSHOT: ProviderConfig(
        name="moonshot",
        display_name="Moonshot",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        base_url_env="MOONSHOT_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.NEBIUS: ProviderConfig(
        name="nebius",
        display_name="Nebius",
        api_key_env="NEBIUS_API_KEY",
        base_url="https://api.studio.nebius.ai/v1",
        base_url_env="NEBIUS_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.NOUS: ProviderConfig(
        name="nous",
        display_name="Nous Research",
        api_key_env="NOUS_API_KEY",
        base_url="https://inference-api.nousresearch.com/v1",
        base_url_env="NOUS_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.NOVITA: ProviderConfig(
        name="novita",
        display_name="Novita AI",
        api_key_env="NOVITA_API_KEY",
        base_url="https://api.novita.ai/v3/openai",
        base_url_env="NOVITA_BASE_URL",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.OPENAI: ProviderConfig(
        name="openai",
        display_name="OpenAI",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.OPENROUTER: ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.PARASAIL: ProviderConfig(
        name="parasail",
        display_name="Parasail",
        api_key_env="PARASAIL_API_KEY",
        base_url="https://api.parasail.ai/v1",
        base_url_env="PARASAIL_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.PERPLEXITY: ProviderConfig(
        name="perplexity",
        display_name="Perplexity",
        api_key_env="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
        supports_vision=False,
        supports_function_calling=False,
    ),
    ProviderType.REKA: ProviderConfig(
        name="reka",
        display_name="Reka",
        api_key_env="REKA_API_KEY",
        base_url="https://api.reka.ai/v1",
        base_url_env="REKA_BASE_URL",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.SAMBANOVA: ProviderConfig(
        name="sambanova",
        display_name="SambaNova",
        api_key_env="SAMBANOVA_API_KEY",
        base_url="https://api.sambanova.ai/v1",
        base_url_env="SAMBANOVA_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.SILICONFLOW: ProviderConfig(
        name="siliconflow",
        display_name="SiliconFlow",
        api_key_env="SILICONFLOW_API_KEY",
        base_url="https://api.siliconflow.com/v1",
        base_url_env="SILICONFLOW_BASE_URL",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.TOGETHER: ProviderConfig(
        name="together",
        display_name="Together AI",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.VERCEL: ProviderConfig(
        name="vercel",
        display_name="Vercel AI Gateway",
        api_key_env="AI_GATEWAY_API_KEY",
        supports_vision=True,
        supports_function_calling=True,
    ),
    ProviderType.WANDB: ProviderConfig(
        name="wandb",
        display_name="W&B Inference",
        api_key_env="WANDB_API_KEY",
        base_url="https://api.inference.wandb.ai/v1",
        base_url_env="WANDB_INFERENCE_BASE_URL",
        supports_vision=False,
        supports_function_calling=True,
    ),
    ProviderType.XAI: ProviderConfig(
        name="xai",
        display_name="xAI",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        supports_vision=False,
        supports_function_calling=True,
    ),
}

# Special AWS provider handling
AWS_ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]


class ProviderManager:
    """Manager class for provider configurations and operations."""

    @staticmethod
    def get_valid_providers() -> List[str]:
        """Get list of all valid provider names."""
        return [provider.value for provider in ProviderType]

    @staticmethod
    def get_config(provider: str) -> ProviderConfig:
        """Get configuration for a specific provider.

        Args:
            provider: Name of the provider

        Returns:
            ProviderConfig for the specified provider

        Raises:
            ValueError: If provider is not supported
        """
        try:
            provider_type = ProviderType(provider.lower())
            return PROVIDER_CONFIGS[provider_type]
        except ValueError:
            valid_providers = ProviderManager.get_valid_providers()
            raise ValueError(
                f"Invalid provider: {provider}. "
                f"Valid options: {', '.join(valid_providers)}"
            )

    @staticmethod
    def validate_provider(provider: str) -> bool:
        """Validate if a provider name is supported.

        Args:
            provider: Name of the provider to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            ProviderManager.get_config(provider)
            return True
        except ValueError:
            return False

    @staticmethod
    def extract_provider_from_model(model_string: str) -> Optional[str]:
        """Extract provider name from a model string like 'provider/model-name'.

        Args:
            model_string: Full model string

        Returns:
            Provider name if found, None otherwise
        """
        if "/" not in model_string:
            return None
        return model_string.split("/", 1)[0]

    @staticmethod
    def validate_model_string(model_string: str) -> bool:
        """Validate if a model string has correct format and valid provider.

        Args:
            model_string: Model string to validate

        Returns:
            True if valid, False otherwise
        """
        provider = ProviderManager.extract_provider_from_model(model_string)
        if not provider:
            return False
        return ProviderManager.validate_provider(provider)

    @staticmethod
    def get_configured_providers() -> List[str]:
        """Get list of providers that are properly configured (have API keys).

        Returns:
            List of provider names that have API keys set
        """
        configured = []
        for provider_type, config in PROVIDER_CONFIGS.items():
            if config.is_configured():
                configured.append(provider_type.value)
        return configured

    @staticmethod
    def get_all_env_vars() -> List[str]:
        """Get all environment variables used by all providers.

        Returns:
            List of all environment variable names
        """
        env_vars = set()
        for config in PROVIDER_CONFIGS.values():
            env_vars.update(config.get_all_env_vars())

        # Add AWS variables
        env_vars.update(AWS_ENV_VARS)

        return sorted(list(env_vars))

    @staticmethod
    def get_env_vars_dict() -> Dict[str, str]:
        """Get all provider environment variables as a dictionary.

        Returns:
            Dictionary mapping env var names to their values (empty string if not set)
        """
        env_vars = ProviderManager.get_all_env_vars()
        return {key: os.getenv(key, "") for key in env_vars}

    @staticmethod
    def get_provider_display_name(provider: str) -> str:
        """Get the display name for a provider.

        Args:
            provider: Provider name

        Returns:
            Display name for the provider
        """
        try:
            config = ProviderManager.get_config(provider)
            return config.display_name
        except ValueError:
            return provider.title()

    @staticmethod
    def get_all_configs() -> Dict[str, ProviderConfig]:
        """Get all provider configurations.

        Returns:
            Dictionary mapping provider names to their configurations
        """
        return {provider.value: config for provider, config in PROVIDER_CONFIGS.items()}
