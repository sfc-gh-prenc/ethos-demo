"""Client subpackage — re-exports all public names for backward compatibility."""

from .admin import ModelInfo, check_health, list_models
from .chat import ChatClient
from .completions import CompletionClient

__all__ = [
    "ChatClient",
    "CompletionClient",
    "ModelInfo",
    "check_health",
    "list_models",
]
