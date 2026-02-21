"""Backend monitoring for the ETHOS deployment."""

import logging
import time
from enum import StrEnum, auto

import streamlit as st

from ethos_demo.client import check_health, list_models
from ethos_demo.config import HEALTH_POLL_SECONDS

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
    def models(self) -> list[str]:
        return st.session_state.get(self._KEY_MODELS, [])

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

        for key in ("ethos_model_id", "llm_model_id"):
            saved = st.session_state.get(f"_saved_{key}")
            if saved and saved not in models:
                _logger.info("Model %r no longer available — resetting %s", saved, key)
                st.session_state.pop(f"_saved_{key}", None)
                st.session_state.pop(key, None)

    def _on_connection_lost(self) -> None:
        _logger.info("Connection lost")
        st.session_state[self._KEY_MODELS] = []
