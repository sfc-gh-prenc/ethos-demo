"""Chat client for the vLLM backend (used by SummaryGenerator)."""

import logging
from collections.abc import Generator

from openai import AsyncOpenAI, OpenAI

from ..config import API_KEY, DEFAULT_BASE_URL

_logger = logging.getLogger(__name__)


class ChatClient:
    """Holds model / connection config and exposes chat completion methods."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self._sync: OpenAI | None = None
        self._async: AsyncOpenAI | None = None

    @property
    def _client(self) -> OpenAI:
        if self._sync is None:
            self._sync = OpenAI(base_url=self.base_url, api_key=API_KEY)
        return self._sync

    @property
    def _aclient(self) -> AsyncOpenAI:
        if self._async is None:
            self._async = AsyncOpenAI(base_url=self.base_url, api_key=API_KEY)
        return self._async

    @property
    def _extra_body(self) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

    def send(self, messages: list[dict], **kwargs) -> str:
        """Non-streaming chat completion — return the full response text."""
        extra_body = {**self._extra_body, **kwargs.pop("extra_body", {})}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
            **kwargs,
        )
        return response.choices[0].message.content

    async def async_send(self, messages: list[dict], **kwargs) -> str:
        """Async non-streaming chat completion — return the full response text."""
        extra_body = {**self._extra_body, **kwargs.pop("extra_body", {})}
        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
            **kwargs,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """Stream a chat completion, yielding text deltas."""
        extra_body = {**self._extra_body, **kwargs.pop("extra_body", {})}
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            extra_body=extra_body,
            **kwargs,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
