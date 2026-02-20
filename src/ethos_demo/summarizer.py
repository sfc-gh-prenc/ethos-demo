"""Streaming EHR summary generator backed by a chat LLM."""

import queue
import threading
import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, ClassVar

import yaml

from ethos_demo.client import stream_chat_completion
from ethos_demo.config import DEFAULT_BASE_URL, PROMPTS_DIR
from ethos_demo.data import get_last_24h_history, get_patient_demographics

if TYPE_CHECKING:
    from ethos.datasets import InferenceDataset

_SENTINEL = None


class SummaryGenerator:
    """Build an LLM prompt from patient data, stream the response, and surface status updates."""

    _STAGES: ClassVar[list[tuple[str, float]]] = [
        ("Pulling records…", 2.0),
        ("Reading…", 2.0),
        ("Thinking…", 0.0),
    ]

    def __init__(
        self,
        *,
        dataset: "InferenceDataset",
        selected_idx: int,
        scenario_context: str,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
        on_status: Callable[[str], None] | None = None,
        min_warmup_seconds: float = 3.0,
    ) -> None:
        self.dataset = dataset
        self.selected_idx = selected_idx
        self.scenario_context = scenario_context
        self.model_id = model_id
        self.base_url = base_url
        self.on_status = on_status
        self.min_warmup_seconds = min_warmup_seconds

    def _build_messages(self) -> list[dict[str, str]]:
        demographics = get_patient_demographics(self.dataset, self.selected_idx)
        timeline_events = get_last_24h_history(self.dataset, self.selected_idx)

        with open(PROMPTS_DIR / "ehr_summary.yaml") as f:
            prompt_tpl = yaml.safe_load(f)

        user_msg = prompt_tpl["user"].format(
            scenario_context=self.scenario_context,
            marital_status=demographics.get("Marital Status", "unknown"),
            race=demographics.get("Race", "unknown"),
            gender=demographics.get("Gender", "unknown"),
            age=demographics.get("Age", "unknown"),
            timeline_events=timeline_events,
        )
        return [
            {"role": "system", "content": prompt_tpl["system"]},
            {"role": "user", "content": user_msg},
        ]

    def _stream_worker(
        self, messages: list[dict[str, str]], delta_q: queue.Queue[str | None]
    ) -> None:
        try:
            for delta in stream_chat_completion(
                messages, model=self.model_id, base_url=self.base_url
            ):
                delta_q.put(delta)
        finally:
            delta_q.put(_SENTINEL)

    def run(self) -> Generator[str, None, None]:
        """Yield visible summary text chunks (thinking content is hidden).

        Status callbacks fire during the warmup phase before yielding begins.
        """
        messages = self._build_messages()

        delta_q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._stream_worker, args=(messages, delta_q), daemon=True).start()
        t0 = time.monotonic()

        for label, duration in self._STAGES:
            if self.on_status:
                self.on_status(label)
            time.sleep(duration)

        remaining = max(0, self.min_warmup_seconds - (time.monotonic() - t0))
        time.sleep(remaining)

        full_text = ""
        think_done = False
        while True:
            try:
                delta = delta_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if delta is _SENTINEL:
                break
            full_text += delta

            if not think_done:
                if "</think>" in full_text:
                    think_done = True
                else:
                    continue

            visible = full_text.split("</think>", 1)[1].strip()
            if visible:
                yield visible
