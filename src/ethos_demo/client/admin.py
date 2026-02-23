"""Administrative client functions — health checks and model listing."""

from dataclasses import dataclass

import httpx
from openai import OpenAI

from ..config import API_KEY, DEFAULT_BASE_URL, HEALTH_TIMEOUT_SECONDS


@dataclass
class ModelInfo:
    id: str
    model_type: str  # "ethos" | "llm" | "unknown"
    max_model_len: int | None


def check_health(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Ping the Ray Serve health endpoint."""
    root = base_url.removesuffix("/v1").removesuffix("/")
    try:
        resp = httpx.get(f"{root}/-/healthz", timeout=HEALTH_TIMEOUT_SECONDS)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _parse_model_type(model_id: str) -> str:
    if model_id.startswith("ethos/"):
        return "ethos"
    if model_id.startswith("llm/"):
        return "llm"
    return "unknown"


def list_models(base_url: str = DEFAULT_BASE_URL) -> list[ModelInfo]:
    """Return available models with parsed type and context size."""
    client = OpenAI(base_url=base_url, api_key=API_KEY)
    return sorted(
        (
            ModelInfo(
                id=m.id,
                model_type=_parse_model_type(m.id),
                max_model_len=(m.model_extra or {}).get("max_model_len"),
            )
            for m in client.models.list()
        ),
        key=lambda mi: mi.id,
    )
