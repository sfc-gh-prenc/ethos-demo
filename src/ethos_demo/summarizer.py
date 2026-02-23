"""Two-stage streaming EHR summary generator backed by a chat LLM.

Stage 1 (optional): Summarize the patient's past medical history via a
    non-streaming LLM call. Skipped when there are no past-history tokens.
Stage 2: Stream a present-event summary that incorporates the past-history
    summary (or a first-encounter note).
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar

import yaml
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from ethos_demo.client import send_chat_completion, stream_chat_completion
from ethos_demo.config import DEFAULT_BASE_URL, PROMPTS_DIR
from ethos_demo.data import (
    format_tokens_as_dicts,
    get_last_24h_history,
    get_patient_demographics,
    get_stay_history,
    get_timeline_times_us,
    get_triage_history,
)
from ethos_demo.scenarios import Scenario

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ethos.datasets import InferenceDataset

_NO_PAST_HISTORY = (
    "This patient's prior medical history is not known because this is "
    "their first encounter with our healthcare system."
)


class SummaryGenerator:
    """Build an LLM prompt from patient data, stream the response, and surface status updates.

    Designed to be stored in session state. The UI reads ``running``,
    ``status``, and ``text`` at render time instead of relying on callbacks.
    """

    _HISTORY_FN: ClassVar[dict] = {
        Scenario.TRIAGE: get_triage_history,
        Scenario.HOSPITAL_ADMISSION: get_last_24h_history,
        Scenario.HOSPITAL_DISCHARGE: get_stay_history,
    }

    def __init__(
        self,
        *,
        dataset: "InferenceDataset",
        selected_idx: int,
        scenario: Scenario,
        scenario_context: str,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
        enable_thinking: bool = False,
    ) -> None:
        self.dataset = dataset
        self.selected_idx = selected_idx
        self.scenario = scenario
        self.scenario_context = scenario_context
        self.model_id = model_id
        self.base_url = base_url
        self.enable_thinking = enable_thinking

        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: str | None = None
        self._text: str | None = None

    # ── public properties (read at render time) ────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def text(self) -> str | None:
        return self._text

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Launch summary generation in a background thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        add_script_run_ctx(self._thread, get_script_run_ctx())
        self._thread.start()

    def cancel(self) -> None:
        """Signal cancellation.

        Safe to call multiple times.
        """
        self._cancel_event.set()

    # ── internals ──────────────────────────────────────────────────

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _wait(self, seconds: float) -> None:
        scaled = seconds * (1.0 if self.enable_thinking else 0.2)
        if scaled > 0:
            time.sleep(scaled)

    @staticmethod
    def _fmt_demographics(
        demographics: dict[str, str],
        scenario_context: str | None = None,
    ) -> dict[str, str]:
        fmt: dict[str, str] = {
            "marital_status": demographics.get("Marital Status", "unknown"),
            "race": demographics.get("Race", "unknown"),
            "gender": demographics.get("Gender", "unknown"),
            "age": demographics.get("Age", "unknown"),
        }
        if scenario_context is not None:
            fmt["scenario_context"] = scenario_context
        return fmt

    def _build_past_history_messages(
        self, past_dicts: list[dict], fmt: dict[str, str]
    ) -> list[dict[str, str]]:
        with open(PROMPTS_DIR / "past_history.yaml") as f:
            tpl = yaml.safe_load(f)
        kwargs = {**fmt, "past_timeline_events": past_dicts}
        return [
            {"role": "system", "content": tpl["system"].format(**kwargs)},
            {"role": "user", "content": tpl["user"].format(**kwargs)},
        ]

    def _build_present_messages(
        self, present_dicts: list[dict], past_summary: str, fmt: dict[str, str]
    ) -> list[dict[str, str]]:
        with open(PROMPTS_DIR / "ehr_summary.yaml") as f:
            tpl = yaml.safe_load(f)
        kwargs = {
            **fmt,
            "past_history_summary": past_summary,
            "timeline_events": present_dicts,
        }
        return [
            {"role": "system", "content": tpl["system"].format(**kwargs)},
            {"role": "user", "content": tpl["user"].format(**kwargs)},
        ]

    @staticmethod
    def _strip_think(text: str, enable_thinking: bool) -> str:
        """Remove the <think>...</think> block if present."""
        if not enable_thinking:
            return text
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return ""

    def _run(self) -> None:
        # ── 1. Split & format timeline ──────────────────────────
        self._status = "Pulling records…"
        self._wait(2.0)
        if self._cancelled():
            return

        history_fn = self._HISTORY_FN[self.scenario]
        past_tokens, present_tokens = history_fn(self.dataset, self.selected_idx)

        self._status = "Reading…"
        self._wait(1.5)
        if self._cancelled():
            return

        past_dicts = format_tokens_as_dicts(past_tokens)
        present_dicts = format_tokens_as_dicts(present_tokens)

        timeline_start_us, _prediction_us = get_timeline_times_us(
            self.dataset,
            self.selected_idx,
        )
        past_demo = get_patient_demographics(
            self.dataset,
            self.selected_idx,
            reference_time_us=timeline_start_us,
        )
        present_demo = get_patient_demographics(self.dataset, self.selected_idx)

        past_fmt = self._fmt_demographics(past_demo)
        present_fmt = self._fmt_demographics(present_demo, self.scenario_context)

        _logger.debug(
            "Timeline split (%s): %d past events, %d present events",
            self.scenario,
            len(past_dicts),
            len(present_dicts),
        )

        # ── 2. Stage 1: Summarize past history (non-streaming) ──
        if past_dicts:
            self._status = "Reviewing past history…"
            if self._cancelled():
                return

            past_messages = self._build_past_history_messages(past_dicts, past_fmt)
            raw = send_chat_completion(
                past_messages,
                model=self.model_id,
                base_url=self.base_url,
                enable_thinking=self.enable_thinking,
            )
            past_summary = self._strip_think(raw, self.enable_thinking)
            _logger.debug("Past history summary: %s", past_summary)
        else:
            past_summary = _NO_PAST_HISTORY

        if self._cancelled():
            return

        # ── 3. Stage 2: Stream present summary ──────────────────
        self._status = "Focusing on present…"
        if self._cancelled():
            return

        present_messages = self._build_present_messages(present_dicts, past_summary, present_fmt)

        full_text = ""
        think_done = not self.enable_thinking

        stream = stream_chat_completion(
            present_messages,
            model=self.model_id,
            base_url=self.base_url,
            enable_thinking=self.enable_thinking,
        )
        try:
            for delta in stream:
                if self._cancelled():
                    return

                full_text += delta

                if not think_done:
                    if "</think>" in full_text:
                        think_done = True
                        full_text = full_text.split("</think>", 1)[1]
                    else:
                        continue

                visible = full_text.strip()
                if visible:
                    self._text = visible
        finally:
            stream.close()

        self._status = None
        if think_done:
            self._text = full_text.strip() or None
