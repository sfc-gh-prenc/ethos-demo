"""Streaming EHR summary generator backed by a chat LLM."""

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import yaml

from ethos_demo.client import stream_chat_completion
from ethos_demo.config import DEFAULT_BASE_URL, PROMPTS_DIR
from ethos_demo.data import (
    get_last_24h_history,
    get_patient_demographics,
    get_stay_history,
    get_triage_history,
)
from ethos_demo.scenarios import Scenario

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ethos.datasets import InferenceDataset


class SummaryGenerator:
    """Build an LLM prompt from patient data, stream the response, and surface status updates."""

    _STAGES: ClassVar[list[tuple[str, float]]] = [
        ("Pulling records…", 2.0),
        ("Reading…", 1.5),
        ("Summarizing…", 0.0),
    ]

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
        on_status: Callable[[str], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.dataset = dataset
        self.selected_idx = selected_idx
        self.scenario = scenario
        self.scenario_context = scenario_context
        self.model_id = model_id
        self.base_url = base_url
        self.on_status = on_status
        self.on_chunk = on_chunk
        self.cancel_event = cancel_event
        self.enable_thinking = enable_thinking

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _build_messages(self) -> list[dict[str, str]]:
        demographics = get_patient_demographics(self.dataset, self.selected_idx)
        history_fn = self._HISTORY_FN[self.scenario]
        timeline_events = history_fn(self.dataset, self.selected_idx)
        _logger.debug("EHR summary event tokens (%s): %s", self.scenario, timeline_events)

        with open(PROMPTS_DIR / "ehr_summary.yaml") as f:
            prompt_tpl = yaml.safe_load(f)

        fmt_kwargs = {
            "scenario_context": self.scenario_context,
            "marital_status": demographics.get("Marital Status", "unknown"),
            "race": demographics.get("Race", "unknown"),
            "gender": demographics.get("Gender", "unknown"),
            "age": demographics.get("Age", "unknown"),
            "timeline_events": timeline_events,
        }
        return [
            {"role": "system", "content": prompt_tpl["system"].format(**fmt_kwargs)},
            {"role": "user", "content": prompt_tpl["user"].format(**fmt_kwargs)},
        ]

    def run(self) -> str | None:
        """Stream summary, pushing visible chunks via *on_chunk*.

        Returns final text.
        """
        messages = self._build_messages()

        time_scale = 1.0 if self.enable_thinking else 0.2
        for label, duration in self._STAGES:
            if self._cancelled():
                return None
            if self.on_status:
                self.on_status(label)
            time.sleep(duration * time_scale)

        if self._cancelled():
            return None

        full_text = ""
        think_done = not self.enable_thinking

        stream = stream_chat_completion(
            messages,
            model=self.model_id,
            base_url=self.base_url,
            enable_thinking=self.enable_thinking,
        )
        try:
            for delta in stream:
                if self._cancelled():
                    return None

                full_text += delta

                if not think_done:
                    if "</think>" in full_text:
                        think_done = True
                        full_text = full_text.split("</think>", 1)[1]
                    else:
                        continue

                visible = full_text.strip()
                if visible and self.on_chunk:
                    self.on_chunk(visible)
        finally:
            stream.close()

        return full_text.strip() if think_done else None
