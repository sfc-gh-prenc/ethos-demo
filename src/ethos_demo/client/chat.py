"""Chat client for the vLLM backend (used by SummaryGenerator)."""

import json
import logging
import re
from collections.abc import Callable, Generator

from openai import OpenAI

from ..config import API_KEY, DEFAULT_BASE_URL

_logger = logging.getLogger(__name__)

_INLINE_TOOL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"parameters"\s*:\s*(?P<args>\{.*\})\s*\}',
    re.DOTALL,
)


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

    @property
    def _client(self) -> OpenAI:
        if self._sync is None:
            self._sync = OpenAI(base_url=self.base_url, api_key=API_KEY)
        return self._sync

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

    def with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_handler: Callable[[str, dict], str],
        *,
        max_rounds: int = 1,
        finalize: bool = False,
    ) -> tuple[list[dict], str]:
        """Run a non-streaming chat loop that resolves tool calls.

        Returns ``(final_messages, content_text)`` where *final_messages*
        includes all intermediate assistant/tool messages and *content_text*
        is the model's final textual answer.

        When *finalize* is True and tool-call rounds are exhausted, one
        extra ``tool_choice="none"`` call is made so the model produces a
        text answer that incorporates the tool results.  When False
        (default) the caller is expected to follow up (e.g. via streaming).
        """
        messages = list(messages)

        for round_idx in range(max_rounds):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_body=self._extra_body,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                inline = _INLINE_TOOL_RE.search(text)
                if inline:
                    _logger.debug(
                        "Round %d: model wrote tool call as text, executing inline",
                        round_idx,
                    )
                    name = inline.group("name")
                    args = json.loads(inline.group("args"))
                    result = tool_handler(name, args)
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                    continue
                return messages, text

            messages.append(
                {"role": "assistant", "tool_calls": msg.tool_calls},
            )
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                _logger.debug("Tool call: %s(%s)", call.function.name, args)
                result = tool_handler(call.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call.id,
                        "name": call.function.name,
                    }
                )

        _logger.debug("with_tools: max_rounds (%d) reached", max_rounds)
        if finalize:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="none",
                extra_body=self._extra_body,
            )
            return messages, response.choices[0].message.content or ""
        return messages, ""

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
