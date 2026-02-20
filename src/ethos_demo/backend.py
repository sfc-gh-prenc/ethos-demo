"""Backend monitoring for the ETHOS deployment."""

import logging
from enum import StrEnum, auto

import streamlit as st

from ethos_demo.client import check_health, list_models

_logger = logging.getLogger(__name__)


class BackendEvent(StrEnum):
    UNCHANGED = auto()
    CHECKING = auto()
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
    _KEY_LOADING = "_health_loading"
    _KEY_MODELS = "_available_models"

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    # ── read-only state ───────────────────────────────────────────

    @property
    def healthy(self) -> bool:
        return bool(st.session_state.get(self._KEY_HEALTHY))

    @property
    def loading(self) -> bool:
        return bool(st.session_state.get(self._KEY_LOADING))

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

    def request_check(self) -> None:
        """Request a manual health check on the next fragment tick."""
        st.session_state[self._KEY_LOADING] = True

    def poll(self) -> BackendEvent:
        """Run one health-check cycle and return what happened.

        Call this from inside a ``@st.fragment(run_every=…)`` function.
        """
        loading = st.session_state.pop(self._KEY_LOADING, False)
        was_healthy = st.session_state.get(self._KEY_HEALTHY, None)

        # Show "Checking…" only when recovering from an unreachable state;
        # if already connected, run the check silently.
        if loading and not was_healthy:
            return BackendEvent.CHECKING

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

    def _on_connection_lost(self) -> None:
        _logger.info("Connection lost")
        st.session_state[self._KEY_MODELS] = []
