"""Provider factories for embeddings and chat completions."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import httpx

from src.config import EmbeddingProviderConfig, LLMProviderConfig

DEFAULT_EMBEDDING_MODELS = {
    "mock": "mock:384",
    "huggingface": "BAAI/bge-small-en-v1.5",
    "mistral": "mistral-embed",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "custom": "text-embedding-3-small",
}

DEFAULT_EMBEDDING_BASE_URLS = {
    "ollama": "http://localhost:11434",
}


DEFAULT_CHAT_MODELS = {
    "mistral": "mistral-small-latest",
    "openai": "gpt-4.1-mini",
    "openrouter": "openai/gpt-4.1-mini",
    "ollama": "llama3.1",
    "custom": "gpt-4.1-mini",
}


DEFAULT_API_KEY_ENVS = {
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "custom": "AI_API_KEY",
}


DEFAULT_CHAT_BASE_URLS = {
    "mistral": "https://api.mistral.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


def provider_catalog() -> dict[str, list[dict[str, str | None]]]:
    """Return supported provider metadata for CLI inspection."""
    return {
        "embeddings": [
            {
                "id": provider,
                "default_model": model,
                "api_key_env": DEFAULT_API_KEY_ENVS.get(provider),
                "base_url": DEFAULT_EMBEDDING_BASE_URLS.get(provider),
            }
            for provider, model in DEFAULT_EMBEDDING_MODELS.items()
        ],
        "llms": [
            {
                "id": provider,
                "default_model": model,
                "api_key_env": DEFAULT_API_KEY_ENVS.get(provider),
                "base_url": DEFAULT_CHAT_BASE_URLS.get(provider),
            }
            for provider, model in DEFAULT_CHAT_MODELS.items()
        ],
    }


def _env_or_value(value: str | None, env_name: str | None) -> str | None:
    if value:
        return value
    if env_name:
        return os.getenv(env_name)
    return None


def embedding_config_from_env(
    provider: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
) -> EmbeddingProviderConfig:
    """Build embedding provider config from CLI values and environment."""
    provider_id = (provider or os.getenv("EMBEDDING_PROVIDER") or "huggingface").lower()
    if provider_id not in DEFAULT_EMBEDDING_MODELS:
        supported = ", ".join(sorted(DEFAULT_EMBEDDING_MODELS))
        raise ValueError(f"Unsupported embedding provider: {provider_id}. Supported: {supported}")

    resolved_model = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS.get(provider_id)
    resolved_api_key_env = api_key_env or os.getenv("EMBEDDING_API_KEY_ENV") or DEFAULT_API_KEY_ENVS.get(provider_id)
    resolved_base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URLS.get(provider_id)

    if not resolved_model:
        raise ValueError(f"No default embedding model configured for provider: {provider_id}")
    if provider_id == "custom" and not resolved_base_url:
        raise ValueError("EMBEDDING_BASE_URL is required for custom embedding provider")

    return EmbeddingProviderConfig(
        provider=provider_id,
        model=resolved_model,
        api_key_env=resolved_api_key_env,
        api_key=None,
        base_url=resolved_base_url,
    )


def llm_config_from_env(
    provider: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
) -> LLMProviderConfig:
    """Build chat provider config from CLI values and environment."""
    provider_id = (provider or os.getenv("LLM_PROVIDER") or "mistral").lower()
    if provider_id not in DEFAULT_CHAT_MODELS:
        supported = ", ".join(sorted(DEFAULT_CHAT_MODELS))
        raise ValueError(f"Unsupported LLM provider: {provider_id}. Supported: {supported}")

    resolved_model = model or os.getenv("LLM_MODEL") or DEFAULT_CHAT_MODELS.get(provider_id)
    resolved_api_key_env = api_key_env or os.getenv("LLM_API_KEY_ENV") or DEFAULT_API_KEY_ENVS.get(provider_id)
    resolved_base_url = (
        base_url
        or os.getenv("LLM_BASE_URL")
        or (os.getenv("OLLAMA_BASE_URL") + "/v1" if provider_id == "ollama" and os.getenv("OLLAMA_BASE_URL") else None)
        or DEFAULT_CHAT_BASE_URLS.get(provider_id)
    )

    if not resolved_model:
        raise ValueError(f"No default chat model configured for provider: {provider_id}")
    if not resolved_base_url:
        raise ValueError(f"No chat base URL configured for provider: {provider_id}")

    return LLMProviderConfig(
        provider=provider_id,
        model=resolved_model,
        api_key_env=resolved_api_key_env,
        api_key=None,
        base_url=resolved_base_url.rstrip("/"),
    )


def build_embedding_model(config: EmbeddingProviderConfig):
    """Instantiate a LlamaIndex embedding model from provider config."""
    provider = config.provider.lower()
    api_key = _env_or_value(config.api_key, config.api_key_env)

    if provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=config.model)

    if provider == "mock":
        from llama_index.core.embeddings import MockEmbedding

        return MockEmbedding(embed_dim=_parse_mock_embed_dim(config.model))

    if provider == "mistral":
        from llama_index.embeddings.mistralai import MistralAIEmbedding

        if not api_key:
            raise ValueError(f"{config.api_key_env or 'MISTRAL_API_KEY'} not set")
        return MistralAIEmbedding(model_name=config.model, api_key=api_key)

    if provider in {"openai", "custom"}:
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not api_key:
            raise ValueError(f"{config.api_key_env or 'OPENAI_API_KEY'} not set")
        if provider == "custom" and not config.base_url:
            raise ValueError("EMBEDDING_BASE_URL is required for custom embedding provider")
        kwargs: dict[str, Any] = {"model": config.model, "api_key": api_key}
        if config.base_url:
            kwargs["api_base"] = config.base_url
        return OpenAIEmbedding(**kwargs)

    if provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=config.model,
            base_url=config.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
        )

    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def _parse_mock_embed_dim(model: str) -> int:
    value = model.removeprefix("mock:")
    try:
        dim = int(value)
    except ValueError as exc:
        raise ValueError("Mock embedding model must be an integer dimension, e.g. mock:384") from exc
    if dim <= 0:
        raise ValueError("Mock embedding dimension must be positive")
    return dim


def _auth_headers(config: LLMProviderConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = _env_or_value(config.api_key, config.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if config.provider == "openrouter":
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

    return headers


def chat_completion_json(
    prompt: str,
    config: LLMProviderConfig,
    temperature: float = 0.5,
    timeout: float = 90.0,
) -> dict:
    """Call an OpenAI-compatible chat endpoint and parse JSON content."""
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    response = httpx.post(
        f"{config.base_url}/chat/completions",
        headers=_auth_headers(config),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return _parse_json_content(response.json())


async def chat_completion_json_async(
    prompt: str,
    config: LLMProviderConfig,
    temperature: float = 0.5,
    timeout: float = 90.0,
) -> dict:
    """Async version of chat_completion_json."""
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{config.base_url}/chat/completions",
            headers=_auth_headers(config),
            json=payload,
        )
        response.raise_for_status()
        return _parse_json_content(response.json())


def with_model(config: LLMProviderConfig, model: str | None) -> LLMProviderConfig:
    """Return config with optional model override."""
    return replace(config, model=model) if model else config


def _parse_json_content(payload: dict) -> dict:
    import json

    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, dict):
        return content
    return json.loads(content)
