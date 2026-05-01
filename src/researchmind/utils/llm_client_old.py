from abc import ABC, abstractmethod
import instructor



class LLMClient(ABC):
    """Abstract interface for LLM clients used in query transformations."""

    @abstractmethod
    def complete(self) -> str:
        """Send a prompt and return the text response."""
        ...

    @abstractmethod
    def complete_structured(self):
        """Send a prompt and return a structured response (e.g. JSON)."""
        ...


class AnthropicClient(LLMClient):
    """LLM client for Anthropic's Messages API."""

    def __init__(self, api_key: str):
        import anthropic          # only runs when AnthropicClient() is actually instantiated
        self._client = anthropic.Anthropic(api_key=api_key)
        self._instructor_client = instructor.from_anthropic(self._client)

    def complete(
        self, model: str, user_prompt: str, system_prompt: str, max_tokens: int = 512
    ) -> str:
        response = self._client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def complete_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
        max_tokens=2048,
        temperature=0,
    ):
        """Get structured response from Anthropic API, parsing the content as JSON into the specified Pydantic model."""
        response = self._instructor_client.chat.completions.create(
            model=model,
            system=system_prompt,
            response_model=response_model,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,  # some creativity allowed for synthesis
        )
        return response


class OllamaClient(LLMClient):
    """LLM client for local Ollama models."""

    def __init__(self, think: bool = False, num_ctx: int = 4096):
        import ollama          # only runs when OllamaClient() is actually instantiated
        self._client = ollama.Client()
        self._think = think
        self._num_ctx = num_ctx
        self._instructor_cache: dict[str, instructor.Instructor] = {}
    
    def _get_instructor(self, model: str):
        if model not in self._instructor_cache:
            self._instructor_cache[model] = instructor.from_provider(f"ollama/{model}")
        return self._instructor_cache[model]

    def complete(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        response = self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
            },
            think=self._think,
            keep_alive=-1,
        )
        return response.message.content.strip()

    def complete_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
        max_tokens=2048,
        temperature=0,
    ):
        """Get structured response from Anthropic API, parsing the content as JSON into the specified Pydantic model."""
        instructor_client = self._get_instructor(model)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = instructor_client.create(
            response_model=response_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"keep_alive": -1},
        )
        return response


class OpenAIClient(LLMClient):
    """LLM client for OpenAI's Chat Completions API."""

    def __init__(self, api_key: str, base_url: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._instructor_client = instructor.from_openai(self._client)

    def complete(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def complete_structured(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
        max_tokens=2048,
        temperature=0,
    ):
        """Get structured response from Anthropic API, parsing the content as JSON into the specified Pydantic model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._instructor_client.chat.completions.create(
            model=model_name,
            response_model=response_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,  # some creativity allowed for synthesis
        )
        return response
