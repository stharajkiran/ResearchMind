from abc import ABC, abstractmethod
import os
import os
from typing import Type, TypeVar
import instructor
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Abstract base — enforces a consistent signature across all providers
# ---------------------------------------------------------------------------


class LLMProvider(ABC):

    @abstractmethod
    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...

    @abstractmethod
    def complete_structured(
        self,
        model: str,
        user_prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T: ...


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):

    def __init__(self, api_key: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._instructor = instructor.from_anthropic(self._client)

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def complete_structured(
        self,
        model: str,
        user_prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T:
        kwargs = dict(
            model=model,
            response_model=response_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        return self._instructor.chat.completions.create(**kwargs)


class OllamaProvider(LLMProvider):

    def __init__(self, think: bool = False, num_ctx: int = 4096):
        import ollama

        self._client = ollama.Client()
        self._think = think
        self._num_ctx = num_ctx
        self._instructor_cache: dict[str, instructor.Instructor] = {}

    def _get_instructor(self, model: str):
        if model not in self._instructor_cache:
            self._instructor_cache[model] = instructor.from_provider(
                f"ollama/{model}", mode=instructor.Mode.JSON
            )
        return self._instructor_cache[model]

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat(
            model=model,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
                "temperature": temperature,
            },
            think=self._think,
            keep_alive=-1,
        )
        return response.message.content.strip()

    def complete_structured(
        self,
        model: str,
        user_prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T:
        instructor_client = self._get_instructor(model)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return instructor_client.create(
            response_model=response_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"keep_alive": -1, },
        )


class OpenAICompatibleProvider(LLMProvider):

    def __init__(self, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._instructor = instructor.from_openai(self._client)

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def complete_structured(
        self,
        model: str,
        user_prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return self._instructor.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Composer — the only thing the rest of the codebase imports
# ---------------------------------------------------------------------------


class ResearchMindLLM:
    """
    Single entry point for all LLM calls in ResearchMind.

    Tier routing:
        fast   -> Qwen3.5-9B   (Ollama)      — router, query rewriting
        medium -> Qwen3.6-27B  (Ollama)      — synthesis, comparison
        strong -> DeepSeek V4  (OpenAI-compat)— complex reasoning
        best   -> Claude        (Anthropic)   — gap detection only
    """

    TIERS: dict[str, tuple[str, str]] = {
        "fast": ("qwen3.5:9b", "ollama"),
        "medium": ("qwen3.6:27b", "ollama"),
        "strong": ("deepseek-chat", "openai"),
        "best": ("claude-sonnet-4-6", "anthropic"),
    }

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        deepseek_api_key: str | None = None,
        deepseek_base_url: str = "https://api.deepseek.com",
        ollama_think: bool = False,
        ollama_num_ctx: int = 4096,
    ):
        self._providers: dict[str, LLMProvider] = {
            "anthropic": AnthropicProvider(
                api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            ),
            "ollama": OllamaProvider(think=ollama_think, num_ctx=ollama_num_ctx),
            "openai": OpenAICompatibleProvider(
                api_key=deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url=deepseek_base_url,
            ),
        }

    def _resolve(self, tier: str) -> tuple[LLMProvider, str]:
        if tier not in self.TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Choose from: {list(self.TIERS)}")
        model, provider_key = self.TIERS[tier]
        return self._providers[provider_key], model

    def complete(
        self,
        user_prompt: str,
        tier: str = "fast",
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Get the response text from the specified tier.

        Tier routing:
        - fast   -> Qwen3.5-9B   (Ollama)      — router, query rewriting
        - medium -> Qwen3.6-27B  (Ollama)      — synthesis, comparison
        - strong -> DeepSeek V4  (OpenAI-compat)— complex reasoning
        - best   -> Claude        (Anthropic)   — gap detection only
        """
        provider, model = self._resolve(tier)
        return provider.complete(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_structured(
        self,
        user_prompt: str,
        response_model: Type[T],
        tier: str = "medium",
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T:
        """Get a structured response from the specified tier.

        Args:
            user_prompt (str): The prompt to send to the LLM.
            response_model (Type[T]): The model/class to use for the structured response.
            tier (str, optional): The tier to use. Defaults to "medium".
            system_prompt (str | None, optional): The system prompt to guide the LLM. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2048.
            temperature (float, optional): The sampling temperature. Defaults to 0.0.

        Returns:
            T: The structured response.
        """
        provider, model = self._resolve(tier)
        return provider.complete_structured(
            model=model,
            user_prompt=user_prompt,
            response_model=response_model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
