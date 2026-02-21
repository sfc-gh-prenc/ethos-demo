"""Outcome estimator using batch completions with post-hoc token analysis."""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import timedelta

from ethos.datasets import InferenceDataset
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from ethos_demo.client import send_raw_completion_async
from ethos_demo.config import DEFAULT_BASE_URL, MAX_CONCURRENT_STREAMS, N_STREAMS
from ethos_demo.data import get_sample_prompt
from ethos_demo.scenarios import SCENARIOS, OutcomeRule, Scenario

logger = logging.getLogger(__name__)


# ── StreamTracker ────────────────────────────────────────────────────────


@dataclass
class StreamTracker:
    """Token-by-token state machine that resolves multiple outcomes from one stream.

    Accumulates timeline time from TIME_INTERVAL tokens and resolves each outcome as
    positive/negative based on event tokens and time windows.
    """

    rules: list[OutcomeRule]
    interval_estimates: dict[str, float]
    max_time: timedelta | None

    _buffer: str = field(default="", init=False)
    _accumulated_us: float = field(default=0.0, init=False)
    _outcomes: dict[str, bool | None] = field(default_factory=dict, init=False)
    _done: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._outcomes = {r.name: None for r in self.rules}

    @property
    def done(self) -> bool:
        return self._done

    @property
    def outcomes(self) -> dict[str, bool]:
        """Final resolved outcomes (undetermined treated as negative)."""
        return {name: (val is True) for name, val in self._outcomes.items()}

    @property
    def accumulated_time(self) -> timedelta:
        return timedelta(microseconds=self._accumulated_us)

    # ── public API ────────────────────────────────────────────────

    def process_chunk(self, text: str) -> bool:
        """Feed a text chunk.

        Returns True when the stream should be cancelled.
        """
        if self._done:
            return True

        self._buffer += text
        tokens = self._buffer.split()

        # Keep the last partial token in the buffer unless text ends with whitespace
        if text and not text[-1].isspace():
            self._buffer = tokens.pop() if tokens else ""
        else:
            self._buffer = ""

        for token in tokens:
            self._process_token(token)
            if self._done:
                return True
        return False

    def flush(self) -> None:
        """Process any remaining partial token when the stream ends."""
        if self._buffer and not self._done:
            self._process_token(self._buffer)
            self._buffer = ""
        self._finalize()

    # ── internal ──────────────────────────────────────────────────

    def _process_token(self, token: str) -> None:
        if token in self.interval_estimates:
            self._accumulated_us += self.interval_estimates[token]

        for rule in self.rules:
            if self._outcomes[rule.name] is not None:
                continue
            if token in rule.positive_events and (
                rule.time_window is None or self.accumulated_time <= rule.time_window
            ):
                self._outcomes[rule.name] = True

        # Resolve outcomes whose time window has been exceeded
        for rule in self.rules:
            if self._outcomes[rule.name] is not None:
                continue
            if rule.time_window is not None and self.accumulated_time > rule.time_window:
                self._outcomes[rule.name] = False

        # Check if max time exceeded
        if self.max_time is not None and self.accumulated_time > self.max_time:
            self._finalize()
            return

        if all(v is not None for v in self._outcomes.values()):
            self._done = True

    def _finalize(self) -> None:
        """Mark all undetermined outcomes as negative and set done."""
        for name, val in self._outcomes.items():
            if val is None:
                self._outcomes[name] = False
        self._done = True


# ── OutcomeEstimator ─────────────────────────────────────────────────────


@dataclass
class _AggResult:
    positives: int = 0
    valid: int = 0

    @property
    def probability(self) -> float | None:
        return self.positives / self.valid if self.valid else None


class OutcomeEstimator:
    """Fire N concurrent completions and derive outcome probabilities.

    Designed to be stored in session state. The UI reads ``running``,
    ``progress``, ``probabilities``, and ``log_summary`` at render time
    instead of relying on callbacks.
    """

    def __init__(
        self,
        *,
        dataset: InferenceDataset,
        sample_idx: int,
        scenario: Scenario,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 1.0,
        n_requests: int = N_STREAMS,
        max_concurrent: int = MAX_CONCURRENT_STREAMS,
    ) -> None:
        self.dataset = dataset
        self.sample_idx = sample_idx
        self.scenario = scenario
        self.model_id = model_id
        self.base_url = base_url
        self.temperature = temperature
        self.n_requests = n_requests
        self.max_concurrent = max_concurrent

        sc = SCENARIOS[scenario]
        self._rules = sc.outcomes
        self._outcome_names = [r.name for r in self._rules]
        self._max_time = self._derive_max_time()
        self._stop_tokens = list(sc.stop_tokens)
        self._interval_estimates: dict[str, float] = dataset.vocab.interval_estimates.get(
            "mean", {}
        )

        self._prompt, self._n_input_tokens = get_sample_prompt(dataset, sample_idx)

        self._results: dict[str, _AggResult] = {name: _AggResult() for name in self._outcome_names}
        self._completed = 0
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── public properties (read at render time) ────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def progress(self) -> tuple[int, int] | None:
        if self._completed == 0:
            return None
        return (self._completed, self.n_requests)

    @property
    def probabilities(self) -> dict[str, tuple[float, int, int] | None]:
        """Per-outcome ``(probability, positives, valid)`` or None."""
        out: dict[str, tuple[float, int, int] | None] = {}
        for name, agg in self._results.items():
            if agg.probability is not None:
                out[name] = (agg.probability, agg.positives, agg.valid)
            else:
                out[name] = None
        return out

    @property
    def log_summary(self) -> dict:
        done = self._completed
        return {
            name: {
                "processed": done,
                "valid": agg.valid,
                "yield": f"{100 * agg.valid / done:.1f}%" if done else "n/a",
                "prob": f"{agg.probability:.1%}" if agg.probability is not None else "n/a",
            }
            for name, agg in self._results.items()
        }

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the estimation in a background thread."""
        if self._thread is not None:
            return

        def _run_wrapper() -> None:
            try:
                asyncio.run(self._run())
            except Exception:
                logger.exception("Estimation failed")
            finally:
                logger.debug("estimation done — %s", self.log_summary)

        self._thread = threading.Thread(target=_run_wrapper, daemon=True)
        add_script_run_ctx(self._thread, get_script_run_ctx())
        self._thread.start()

    def cancel(self) -> None:
        """Signal cancellation.

        Safe to call multiple times or on a finished estimator.
        """
        self._cancel_event.set()

    # ── internals ──────────────────────────────────────────────────

    def _derive_max_time(self) -> timedelta | None:
        windows = [r.time_window for r in self._rules if r.time_window is not None]
        return max(windows) if windows else None

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    async def _request_one(self, sem: asyncio.Semaphore) -> StreamTracker:
        tracker = StreamTracker(
            rules=self._rules,
            interval_estimates=self._interval_estimates,
            max_time=self._max_time,
        )
        async with sem:
            if self._is_cancelled():
                tracker.flush()
                return tracker

            pairs = await send_raw_completion_async(
                self._prompt,
                n_input_tokens=self._n_input_tokens,
                model=self.model_id,
                base_url=self.base_url,
                stop=self._stop_tokens,
                temperature=self.temperature,
            )
            text, _finish = pairs[0]
            tracker.process_chunk(text)

        tracker.flush()
        return tracker

    async def _run(self) -> None:
        sem = asyncio.Semaphore(self.max_concurrent)
        futures = [asyncio.ensure_future(self._request_one(sem)) for _ in range(self.n_requests)]
        pending = set(futures)

        while pending:
            if self._is_cancelled():
                for f in pending:
                    f.cancel()
                logger.debug(
                    "Estimation cancelled — %d/%d completed",
                    self._completed,
                    self.n_requests,
                )
                await asyncio.gather(*pending, return_exceptions=True)
                break

            done, pending = await asyncio.wait(
                pending, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                try:
                    tracker = future.result()
                except Exception:
                    logger.debug("Request failed", exc_info=True)
                    self._completed += 1
                    continue

                self._completed += 1
                for name, positive in tracker.outcomes.items():
                    agg = self._results[name]
                    agg.valid += 1
                    if positive:
                        agg.positives += 1
