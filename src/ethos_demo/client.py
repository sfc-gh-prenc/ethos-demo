"""Thin OpenAI-compatible client for talking to the Ray Serve vLLM backend."""

from collections.abc import AsyncIterator, Generator

import httpx
from openai import AsyncOpenAI, OpenAI

from ethos_demo.config import (
    API_KEY,
    DEFAULT_BASE_URL,
    HEALTH_TIMEOUT_SECONDS,
    MODEL_CONTEXT_SIZE,
)


def get_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=API_KEY)


def get_async_client(base_url: str = DEFAULT_BASE_URL) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=API_KEY)


def check_health(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Ping the Ray Serve health endpoint."""
    root = base_url.removesuffix("/v1").removesuffix("/")
    try:
        resp = httpx.get(f"{root}/-/healthz", timeout=HEALTH_TIMEOUT_SECONDS)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def list_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Return available model IDs from the /v1/models endpoint."""
    client = get_client(base_url)
    return sorted(m.id for m in client.models.list())


def send_completion_request(
    prompt: str,
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    **kwargs,
) -> str:
    """Send a text completion request and return the generated text."""
    client = get_client(base_url)
    response = client.completions.create(model=model, prompt=prompt, **kwargs)
    return response.choices[0].text


def send_raw_completion(
    prompt: str,
    *,
    n_input_tokens: int,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    n: int = 1,
    stop: list[str] | None = None,
    **kwargs,
) -> list[tuple[str, str]]:
    """Send a completion request and return (text, finish_reason) for each choice."""
    client = get_client(base_url)
    max_tokens = MODEL_CONTEXT_SIZE - n_input_tokens
    response = client.completions.create(
        model=model,
        prompt=prompt,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        extra_body={"include_stop_str_in_output": True},
        **kwargs,
    )
    return [(c.text, c.finish_reason) for c in response.choices]


async def send_raw_completion_async(
    prompt: str,
    *,
    n_input_tokens: int,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    n: int = 1,
    stop: list[str] | None = None,
    allowed_token_ids: list[int] | None = None,
    **kwargs,
) -> list[tuple[str, str]]:
    """Async version — send a completion request, return (text, finish_reason) per choice."""
    client = get_async_client(base_url)
    max_tokens = MODEL_CONTEXT_SIZE - n_input_tokens
    extra_body: dict = {"include_stop_str_in_output": True}
    if allowed_token_ids is not None:
        extra_body["allowed_token_ids"] = allowed_token_ids
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        extra_body=extra_body,
        **kwargs,
    )
    return [(c.text, c.finish_reason) for c in response.choices]


async def stream_completion_async(
    prompt: str,
    *,
    n_input_tokens: int,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    stop: list[str] | None = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream a text completion, yielding text deltas as they arrive."""
    client = get_async_client(base_url)
    max_tokens = MODEL_CONTEXT_SIZE - n_input_tokens
    stream = await client.completions.create(
        model=model,
        prompt=prompt,
        stream=True,
        max_tokens=max_tokens,
        stop=stop,
        extra_body={"include_stop_str_in_output": True},
        **kwargs,
    )
    try:
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text
    finally:
        await stream.close()


def send_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    enable_thinking: bool = False,
    **kwargs,
) -> str:
    """Send a non-streaming chat completion and return the full response text."""
    client = get_client(base_url)
    extra_body = kwargs.pop("extra_body", {})
    extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )
    return response.choices[0].message.content


def stream_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    enable_thinking: bool = False,
    **kwargs,
) -> Generator[str, None, None]:
    """Stream a chat completion, yielding text deltas."""
    client = get_client(base_url)
    extra_body = kwargs.pop("extra_body", {})
    extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body=extra_body,
        **kwargs,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
