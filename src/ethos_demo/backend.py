"""Backend monitoring for the ETHOS deployment."""

import logging
import time
from enum import StrEnum, auto

import streamlit as st

from .client import ModelInfo, check_health, list_models
from .config import DEFAULT_MODEL_CONTEXT_SIZE, HEALTH_POLL_SECONDS

_logger = logging.getLogger(__name__)


class BackendEvent(StrEnum):
    UNCHANGED = auto()
    CONNECTION_RESTORED = auto()
    CONNECTION_LOST = auto()


class BackendMonitor:
    """Tracks ETHOS deployment health and fires events on state transitions.

    Components query the monitor's properties (``healthy``, ``models``,
    ``has_ethos_model``, …) instead of reading raw session-state keys.
    The ``poll()`` method returns a ``BackendEvent`` so the UI fragment
    can render the appropriate status indicator.
    """

    _KEY_HEALTHY = "health_result"
    _KEY_MODELS = "_available_models"
    _KEY_LAST_CHECK = "_health_last_check"

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    # ── read-only state ───────────────────────────────────────────

    @property
    def healthy(self) -> bool:
        return bool(st.session_state.get(self._KEY_HEALTHY))

    @property
    def _all_models(self) -> list[ModelInfo]:
        return st.session_state.get(self._KEY_MODELS, [])

    @property
    def all_model_ids(self) -> list[str]:
        return [m.id for m in self._all_models]

    def _model_ids_by_type(self, model_type: str) -> list[str]:
        typed = [m.id for m in self._all_models if m.model_type == model_type]
        return typed if typed else self.all_model_ids

    @property
    def ethos_models(self) -> list[str]:
        return self._model_ids_by_type("ethos")

    @property
    def llm_models(self) -> list[str]:
        return self._model_ids_by_type("llm")

    @property
    def ethos_model(self) -> str | None:
        val = st.session_state.get("ethos_model_id")
        return val if val and val != "Could not fetch models" else None

    @property
    def llm_model(self) -> str | None:
        val = st.session_state.get("llm_model_id")
        return val if val and val != "Could not fetch models" else None

    @property
    def has_ethos_model(self) -> bool:
        return self.healthy and self.ethos_model is not None

    @property
    def has_llm_model(self) -> bool:
        return self.healthy and self.llm_model is not None

    @property
    def ethos_max_model_len(self) -> int:
        selected = self.ethos_model
        if selected:
            for m in self._all_models:
                if m.id == selected and m.max_model_len is not None:
                    return m.max_model_len
        return DEFAULT_MODEL_CONTEXT_SIZE

    # ── actions ───────────────────────────────────────────────────

    @property
    def busy(self) -> bool:
        """True when backend requests are in-flight (summary, estimation, etc.)."""
        return any(
            o is not None and o.running
            for o in [
                st.session_state.get("_summarizer"),
                st.session_state.get("_estimator"),
            ]
        )

    def poll(self) -> BackendEvent:
        """Run one health-check cycle and return what happened.

        The fragment fires every HEALTH_RETRY_SECONDS, but when healthy we throttle to only check
        every HEALTH_POLL_SECONDS.
        """
        if self.busy:
            return BackendEvent.UNCHANGED

        now = time.monotonic()
        last_check = st.session_state.get(self._KEY_LAST_CHECK, 0.0)
        was_healthy = st.session_state.get(self._KEY_HEALTHY, None)

        if was_healthy and (now - last_check) < HEALTH_POLL_SECONDS:
            return BackendEvent.UNCHANGED

        st.session_state[self._KEY_LAST_CHECK] = now
        healthy = check_health(self._base_url)
        st.session_state[self._KEY_HEALTHY] = healthy

        if healthy == was_healthy:
            return BackendEvent.UNCHANGED

        if healthy:
            self._on_connection_restored()
            return BackendEvent.CONNECTION_RESTORED

        self._on_connection_lost()
        return BackendEvent.CONNECTION_LOST

    # ── internal event handlers ───────────────────────────────────

    def _on_connection_restored(self) -> None:
        _logger.info("Connection restored — fetching models")
        try:
            models = list_models(self._base_url)
        except Exception:
            models = []
        st.session_state[self._KEY_MODELS] = models

        model_ids = {m.id for m in models}
        for key in ("ethos_model_id", "llm_model_id"):
            current = st.session_state.get(key)
            saved = st.session_state.get(f"_saved_{key}")
            # Clear stale placeholder or unavailable model
            if current == "Could not fetch models" or (saved and saved not in model_ids):
                if saved and saved not in model_ids:
                    _logger.info("Model %r no longer available — resetting %s", saved, key)
                    st.session_state.pop(f"_saved_{key}", None)
                st.session_state.pop(key, None)

    def _on_connection_lost(self) -> None:
        _logger.info("Connection lost")
        st.session_state[self._KEY_MODELS] = []
