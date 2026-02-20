"""Async outcome estimator that fires all requests concurrently."""

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ethos_demo.client import send_raw_completion_async
from ethos_demo.data import (
    compute_task_counts,
    find_sample_idx,
    get_sample_prompt,
    load_dataset,
)

if TYPE_CHECKING:
    from ethos.datasets import InferenceDataset

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Incrementally accumulated outcome counts and probability for a single task."""

    positives: int = 0
    valid: int = 0

    @property
    def probability(self) -> float | None:
        if self.valid == 0:
            return None
        return self.positives / self.valid

    def update(self, batch_positives: int, batch_valid: int) -> None:
        self.positives += batch_positives
        self.valid += batch_valid


class OutcomeEstimator:
    """Fire N concurrent requests per task and report progress via callbacks."""

    def __init__(
        self,
        *,
        dataset_name: str,
        patient_id: int,
        prediction_time: int,
        tasks: list[str],
        model_id: str,
        base_url: str,
        n_requests: int = 10,
        n_per_request: int = 10,
        on_progress: Callable[[int, int], None] | None = None,
        on_task_update: Callable[[str, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.patient_id = patient_id
        self.prediction_time = prediction_time
        self.tasks = tasks
        self.model_id = model_id
        self.base_url = base_url
        self.n_requests = n_requests
        self.n_per_request = n_per_request
        self.on_progress = on_progress
        self.on_task_update = on_task_update
        self.cancel_event = cancel_event
        self.cancelled = False

        self._task_data: dict[str, tuple[InferenceDataset, str, int, list[str]]] = {}
        self.results: dict[str, TaskResult] = {t: TaskResult() for t in tasks}

    def _prepare(self) -> None:
        """Load datasets, resolve per-task sample index, and extract prompts."""
        for t in self.tasks:
            ds = load_dataset(self.dataset_name, t)
            idx = find_sample_idx(ds, self.patient_id, self.prediction_time)
            prompt, n_input_tokens, stop_tokens = get_sample_prompt(ds, idx)
            self._task_data[t] = (ds, prompt, n_input_tokens, stop_tokens)

    async def _send_one(self, task_name: str) -> tuple[str, list[tuple[str, str]]]:
        _ds, prompt, n_tok, stops = self._task_data[task_name]
        batch = await send_raw_completion_async(
            prompt,
            n_input_tokens=n_tok,
            model=self.model_id,
            base_url=self.base_url,
            n=self.n_per_request,
            stop=stops,
        )
        return task_name, batch

    async def _run(self) -> None:
        futures = [
            asyncio.ensure_future(self._send_one(t))
            for t in self.tasks
            for _ in range(self.n_requests)
        ]
        total = len(futures)
        pending = set(futures)
        completed = 0

        while pending:
            if self.cancel_event and self.cancel_event.is_set():
                for f in pending:
                    f.cancel()
                self.cancelled = True
                logger.debug("Estimation cancelled — %d/%d completed", completed, total)
                break

            done, pending = await asyncio.wait(
                pending,
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                task_name, batch = future.result()
                completed += 1

                ds = self._task_data[task_name][0]
                batch_pos, batch_valid = compute_task_counts(batch, task_name, ds.vocab)

                result = self.results[task_name]
                result.update(batch_pos, batch_valid)

                logger.debug(
                    "[%s] batch %d/%d — pos=%d valid=%d (cumul pos=%d valid=%d prob=%.2f%%)",
                    task_name,
                    completed,
                    total,
                    batch_pos,
                    batch_valid,
                    result.positives,
                    result.valid,
                    (result.probability or 0) * 100,
                )

                if self.on_progress:
                    self.on_progress(completed, total)

                if self.on_task_update and result.probability is not None:
                    self.on_task_update(task_name, result.probability)

    def run(self) -> dict[str, TaskResult]:
        """Prepare data, fire all requests, and return results."""
        self._prepare()
        asyncio.run(self._run())
        return self.results
