"""Thin OpenAI-compatible client for talking to the Ray Serve vLLM backend."""

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "ethos"


def get_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="fake-key")


def send_completion_request(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    **kwargs,
) -> str:
    """Send a text completion request and return the generated text."""
    client = get_client(base_url)
    response = client.completions.create(model=model, prompt=prompt, **kwargs)
    return response.choices[0].text
