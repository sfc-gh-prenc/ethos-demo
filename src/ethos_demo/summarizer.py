"""Two-stage streaming EHR summary generator backed by a chat LLM.

Stage 1 (optional): Summarize the patient's past medical history via a
    non-streaming LLM call. Skipped when there are no past-history tokens.
Stage 2: Stream a present-event summary that incorporates the past-history
    summary (or a first-encounter note).
"""

import asyncio
import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, ClassVar, TypeVar

import yaml
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from .client import ChatClient
from .config import DEFAULT_BASE_URL, PROMPTS_DIR
from .data import format_tokens_as_dicts_async, get_decile_ranges, get_patient_demographics
from .scenarios import SCENARIOS, Scenario, get_timeline_times_us

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

    _TOOL_SCHEMA: ClassVar[list[dict]] = [
        {
            "type": "function",
            "function": {
                "name": "get_decile_ranges",
                "description": (
                    "Look up the actual numeric value ranges for each decile "
                    "of specified measurements (labs, vitals, or BMI). Use "
                    "this ONLY for measurements whose decile is clinically "
                    "significant and where knowing the real range would help "
                    "you describe the result more accurately (e.g., critically "
                    "low vs mildly low). Do NOT query every measurement — "
                    "only those where the distinction matters. Returns a "
                    "mapping of decile labels (D1, D2, ...) to value ranges. "
                    "Note: not all measurements have 10 deciles; when "
                    "population values lack differentiation, fewer deciles "
                    "are returned. For blood pressure, pass "
                    "'BLOOD_PRESSURE' to get both SBP and DBP ranges."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "IMPORTANT: Use the EXACT full measurement "
                                "name including all '//' separators as it "
                                "appears in the LAB or VITAL timeline data. "
                                "Lab names follow the pattern "
                                "NAME//UNIT//SPECIMEN, e.g. "
                                "'HEMOGLOBIN//G/DL//BLOOD', "
                                "'HEMATOCRIT//%//BLOOD', "
                                "'SODIUM//MMOL/L//BLOOD', "
                                "'POTASSIUM//MEQ/L//BLOOD', "
                                "'UREA_NITROGEN//MG/DL//BLOOD', "
                                "'CHOLESTEROL_TOTAL//MG/DL//BLOOD'. "
                                "Vitals: 'HEART_RATE', 'BLOOD_PRESSURE'. "
                                "Short names like 'HEMOGLOBIN' or "
                                "'SODIUM' will NOT work."
                            ),
                        },
                    },
                    "required": ["names"],
                },
            },
        }
    ]

    def __init__(
        self,
        *,
        dataset: "InferenceDataset",
        selected_idx: int,
        scenario: Scenario,
        model: str,
        dataset_name: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """Create a summary generator for a single patient encounter.

        Args:
            dataset: Loaded ETHOS inference dataset.
            selected_idx: Index of the patient sample inside *dataset*.
            scenario: Clinical scenario that determines timeline splitting
                and the context injected into the LLM prompt.
            model: vLLM model ID for the chat LLM (e.g. ``"llm/llama-3.1"``).
            dataset_name: Dataset directory name, used for quantile look-ups.
            base_url: OpenAI-compatible API endpoint.
        """
        self.dataset = dataset
        self.selected_idx = selected_idx
        self.scenario = scenario
        self._chat = ChatClient(model=model, base_url=base_url)
        self.dataset_name = dataset_name

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

    _T = TypeVar("_T")

    _SUMMARY_RE = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL)

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @classmethod
    def _extract_summary(cls, text: str) -> str:
        """Extract content between <SUMMARY> tags, or empty string if absent."""
        m = cls._SUMMARY_RE.search(text)
        return m.group(1).strip() if m else ""

    def _call_or_cancel(self, fn: Callable[..., _T], *args, **kwargs) -> _T | None:
        """Run *fn* off-thread, polling for cancellation every 0.2 s.

        Returns the result of *fn*, or ``None`` if cancelled first.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, *args, **kwargs)
        pool.shutdown(wait=False)
        while True:
            if self._cancelled():
                return None
            try:
                return future.result(timeout=0.2)
            except FutureTimeout:
                continue

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

    def _handle_tool_call(self, name: str, args: dict) -> str:
        """Dispatch a tool call and return the JSON result string."""
        if name == "get_decile_ranges":
            result = get_decile_ranges(self.dataset_name, args["names"])
            return json.dumps(result)
        return json.dumps({"error": f"unknown tool: {name}"})

    def _run(self) -> None:
        # ── 1. Split & format timeline ──────────────────────────
        self._status = "Pulling records…"

        sc = SCENARIOS[self.scenario]
        split = sc.history_fn(self.dataset, self.selected_idx)

        async def _format_both():
            return await asyncio.gather(
                format_tokens_as_dicts_async(split.past_tokens),
                format_tokens_as_dicts_async(split.present_tokens),
            )

        past_dicts, present_dicts = asyncio.run(_format_both())

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
        present_fmt = self._fmt_demographics(present_demo, sc.context)

        _logger.debug(
            "Timeline split (%s): %d past events, %d present events",
            self.scenario,
            len(past_dicts),
            len(present_dicts),
        )
        _logger.debug("Current encounter tokens (%s): %s", self.scenario, split.present_tokens)

        # ── 2. Stage 1: Summarize past history (with tool calling) ──
        if past_dicts:
            self._status = "Reviewing past history…"
            if self._cancelled():
                return

            past_messages = self._build_past_history_messages(past_dicts, past_fmt)
            result = self._call_or_cancel(
                self._chat.with_tools,
                past_messages,
                self._TOOL_SCHEMA,
                self._handle_tool_call,
                max_rounds=1,
                finalize=True,
            )
            if result is None:
                return
            _msgs, raw = result
            past_summary = self._extract_summary(raw) or _NO_PAST_HISTORY
            _logger.debug("Past history summary: %s", past_summary)
        else:
            past_summary = _NO_PAST_HISTORY

        # ── 3. Stage 2: Present summary (tool loop then stream) ──
        self._status = "Focusing on current encounter…"
        if self._cancelled():
            return

        present_messages = self._build_present_messages(present_dicts, past_summary, present_fmt)

        result = self._call_or_cancel(
            self._chat.with_tools,
            present_messages,
            self._TOOL_SCHEMA,
            self._handle_tool_call,
            max_rounds=1,
        )
        if result is None:
            return
        resolved_messages, preflight_text = result

        if preflight_text:
            self._text = self._extract_summary(preflight_text) or None
            self._status = None
            return

        # Stream the final response with tool_choice="none"
        full_text = ""
        stream = self._chat.stream(resolved_messages, tool_choice="none")
        try:
            for delta in stream:
                if self._cancelled():
                    return
                full_text += delta
                extracted = self._extract_summary(full_text)
                if extracted and self._SUMMARY_RE.search(full_text):
                    self._text = extracted
        finally:
            stream.close()

        self._status = None
        self._text = self._extract_summary(full_text) or None
