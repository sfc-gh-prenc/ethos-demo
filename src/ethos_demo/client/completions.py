"""Completion client for the vLLM backend (used by OutcomeEstimator)."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from ..config import API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL_CONTEXT_SIZE


class CompletionClient:
    """Holds model / connection config and exposes sync + async completion methods."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        max_model_len: int = DEFAULT_MODEL_CONTEXT_SIZE,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_model_len = max_model_len
        self._sync: OpenAI | None = None
        self._async: AsyncOpenAI | None = None

    @property
    def _client(self) -> OpenAI:
        if self._sync is None:
            self._sync = OpenAI(base_url=self.base_url, api_key=API_KEY)
        return self._sync

    @property
    def _async_client(self) -> AsyncOpenAI:
        if self._async is None:
            self._async = AsyncOpenAI(base_url=self.base_url, api_key=API_KEY)
        return self._async

    def send(
        self,
        prompt: str,
        *,
        n_input_tokens: int,
        n: int = 1,
        stop: list[str] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs,
    ) -> list[tuple[str, str]]:
        """Send a completion request and return (text, finish_reason) per choice."""
        max_tokens = self.max_model_len - n_input_tokens
        response = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            n=n,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={"include_stop_str_in_output": True},
            **kwargs,
        )
        return [(c.text, c.finish_reason) for c in response.choices]

    async def send_async(
        self,
        prompt: str,
        *,
        n_input_tokens: int,
        n: int = 1,
        stop: list[str] | None = None,
        allowed_token_ids: list[int] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs,
    ) -> list[tuple[str, str]]:
        """Async completion request - return (text, finish_reason) per choice."""
        max_tokens = self.max_model_len - n_input_tokens
        extra_body: dict = {"include_stop_str_in_output": True}
        if allowed_token_ids is not None:
            extra_body["allowed_token_ids"] = allowed_token_ids
        response = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            n=n,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            **kwargs,
        )
        return [(c.text, c.finish_reason) for c in response.choices]

    async def stream_async(
        self,
        prompt: str,
        *,
        n_input_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a text completion, yielding text deltas as they arrive."""
        max_tokens = self.max_model_len - n_input_tokens
        stream = await self._async_client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            extra_body={"include_stop_str_in_output": True},
            **kwargs,
        )
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    yield chunk.choices[0].text
        finally:
            await stream.close()
