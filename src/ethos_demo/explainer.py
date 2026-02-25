"""Trajectory-based outcome explanation.

Samples generated trajectories from the estimator, splits them into positive and negative groups for
a given outcome rule, summarizes each via LLM, then generates a streamed probability-conditioned
explanation.
"""

import asyncio
import logging
import re
import threading

import numpy as np
import yaml
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from .client import ChatClient
from .config import (
    DEFAULT_BASE_URL,
    MAX_CONCURRENT_TRAJECTORY_SUMMARIES,
    N_EXPLANATION_TRAJECTORIES,
    PROMPTS_DIR,
)
from .data import build_decile_label_maps, format_events_text, format_tokens_as_dicts_async
from .scenarios import OutcomeRule, Scenario

_logger = logging.getLogger(__name__)

_TOKEN_GROUP_SIZES: dict[str, int] = {
    "HOSPITAL_ADMISSION": 3,
    "HOSPITAL_DISCHARGE": 2,
    "ICU_ADMISSION": 2,
    "SOFA": 2,
    "ED_ADMISSION": 2,
    "ACUITY": 2,
}

_SUMMARY_RE = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL)


def _extract_relevant_tokens(tokens: list[str], rule: OutcomeRule, is_positive: bool) -> list[str]:
    """Trim a trajectory to the clinically relevant portion for an outcome."""
    if not is_positive:
        return tokens
    for i, tok in enumerate(tokens):
        if tok in rule.positive_events:
            group_size = _TOKEN_GROUP_SIZES.get(tok, 1)
            return tokens[: i + group_size]
    return tokens


def _probability_guidance(probability: float, outcome_title: str) -> str:
    pct = probability * 100
    if 40 <= pct <= 60:
        return (
            "The probability is in an uncertain range (40-60%). Provide a balanced "
            f"discussion — what clinical factors could lead to {outcome_title} and "
            "what factors suggest it will not occur. Reference patterns from both "
            "types of scenarios equally."
        )
    if pct > 60:
        return (
            f"The probability is elevated ({pct:.0f}%). Emphasize what clinical "
            f"factors in scenarios where {outcome_title} occurs drive this risk — "
            "what events, lab trends, or complications are involved. Also note what "
            "distinguishes scenarios where the outcome is avoided."
        )
    return (
        f"The probability is low ({pct:.0f}%). Emphasize what clinical factors "
        f"in scenarios where {outcome_title} does not occur make it unlikely — "
        "what indicates a stable or recovering course. Also note what risk factors "
        "in the minority of scenarios could still lead to the outcome."
    )


def _outcome_guidance(is_positive: bool, outcome_title: str) -> str:
    if is_positive:
        return (
            f"This trajectory resulted in {outcome_title}. Focus on what clinical "
            "developments — events, lab changes, medications, procedures — escalate "
            "toward this outcome. Describe the pathway to deterioration."
        )
    return (
        f"This trajectory did NOT result in {outcome_title}. Summarize the full "
        "patient course — all treatments, lab trends, procedures, and transfers. "
        "Describe what the trajectory of recovery or stability looks like."
    )


class TrajectoryExplainer:
    """Explain an outcome probability using sampled trajectory summaries.

    Lifecycle mirrors ``SummaryGenerator`` and ``OutcomeEstimator``: expose
    ``running``, ``status``, and ``text`` for the UI to poll.
    """

    def __init__(
        self,
        *,
        outcome_rule: OutcomeRule,
        probability: float,
        margin: float,
        trajectories: list[tuple[list[str], dict[str, bool]]],
        past_summary: str,
        present_summary: str,
        demographics_context: dict[str, str],
        scenario: Scenario,
        dataset_name: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """Create an explainer for a single outcome rule.

        Args:
            outcome_rule: The outcome to explain.
            probability: Estimated probability for this outcome.
            margin: Wilson CI half-width for the probability.
            trajectories: All stored trajectories from the estimator.
            past_summary: Past history summary text (or placeholder).
            present_summary: Present encounter summary text (or placeholder).
            demographics_context: Demographic fields for prompt formatting.
            scenario: Active clinical scenario.
            dataset_name: Dataset directory name, used for quantile look-ups.
            model: Chat LLM model ID.
            base_url: OpenAI-compatible API endpoint.
        """
        self._rule = outcome_rule
        self._probability = probability
        self._margin = margin
        self._trajectories = trajectories
        self._past_summary = past_summary
        self._present_summary = present_summary
        self._demographics = demographics_context
        self._scenario = scenario
        self._dataset_name = dataset_name
        self._chat = ChatClient(model=model, base_url=base_url)

        self._cancel_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: str | None = None
        self._text: str | None = None
        self._progress: tuple[int, int] | None = None

        # Persistent state across runs (survives stop + resume)
        self._sampled: list[tuple[list[str], bool]] | None = None
        self._summary_results: dict[int, str] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def text(self) -> str | None:
        return self._text

    @property
    def progress(self) -> tuple[int, int] | None:
        """``(completed, total)`` during trajectory summarization, else None."""
        return self._progress

    @property
    def n_summarized(self) -> int:
        """Number of trajectory summaries that were actually generated."""
        return sum(1 for s in self._summary_results.values() if s)

    @property
    def can_resume(self) -> bool:
        """True when stopped early and there are unsummarized trajectories."""
        return (
            not self.running
            and self._sampled is not None
            and len(self._summary_results) < len(self._sampled)
        )

    def start(self) -> None:
        """Launch explanation generation in a background thread."""
        if self._thread is not None:
            return
        self._launch()

    def resume(self) -> None:
        """Resume after an early stop — finish remaining trajectory summaries."""
        self._stop_event.clear()
        self._text = None
        self._thread = None
        self._launch()

    def restart(self) -> None:
        """Clear all cached state and re-run the full flow from scratch."""
        self._stop_event.clear()
        self._cancel_event.clear()
        self._sampled = None
        self._summary_results = {}
        self._text = None
        self._status = None
        self._progress = None
        self._thread = None
        self._launch()

    def _launch(self) -> None:
        def _run_wrapper() -> None:
            try:
                asyncio.run(self._run())
            except Exception:
                _logger.exception("Trajectory explanation failed")

        self._thread = threading.Thread(target=_run_wrapper, daemon=True)
        add_script_run_ctx(self._thread, get_script_run_ctx())
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    def stop(self) -> None:
        """Stop trajectory summarization early and proceed to final explanation."""
        self._stop_event.set()
        self._progress = None

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    def _should_abort(self) -> bool:
        return self._cancelled() or self._stopped()

    async def _call_or_cancel(self, coro):
        """Run an awaitable with periodic cancellation checks."""
        task = asyncio.create_task(coro)
        while not task.done():
            if self._should_abort():
                task.cancel()
                return None
            await asyncio.sleep(0.2)
        return task.result()

    @staticmethod
    def _extract_tag(pattern: re.Pattern, text: str) -> str:
        m = pattern.search(text)
        return m.group(1).strip() if m else text.strip()

    async def _run(self) -> None:
        if self._cancelled():
            return

        n_total = (
            len(self._sampled)
            if self._sampled
            else min(N_EXPLANATION_TRAJECTORIES, len(self._trajectories))
        )
        n_done = len(self._summary_results)
        self._progress = (n_done, n_total)
        self._status = f"Analyzing trajectories\u2026 {n_done}/{n_total}"

        decile_maps = build_decile_label_maps(self._dataset_name)

        # ── 1. Sample and split (first run only) ─────────────────
        if self._sampled is None:
            n_sample = min(N_EXPLANATION_TRAJECTORIES, len(self._trajectories))
            rng = np.random.default_rng()
            indices = rng.choice(len(self._trajectories), size=n_sample, replace=False)

            self._sampled = []
            for idx in indices:
                tokens, outcomes = self._trajectories[idx]
                is_pos = outcomes.get(self._rule.name, False)
                trimmed = _extract_relevant_tokens(tokens, self._rule, is_pos)
                self._sampled.append((trimmed, is_pos))

            _logger.debug(
                "Trajectory split for %s: %d positive, %d negative",
                self._rule.name,
                sum(1 for _, p in self._sampled if p),
                sum(1 for _, p in self._sampled if not p),
            )

        n_total = len(self._sampled)

        # ── 2. Summarize remaining trajectories ──────────────────
        remaining = [
            (i, t, p) for i, (t, p) in enumerate(self._sampled) if i not in self._summary_results
        ]

        if remaining:
            with open(PROMPTS_DIR / "trajectory_summary.yaml") as f:
                tpl = yaml.safe_load(f)

            sem = asyncio.Semaphore(MAX_CONCURRENT_TRAJECTORY_SUMMARIES)
            lock = asyncio.Lock()

            async def _summarize_one(idx: int, tokens: list[str], is_positive: bool) -> None:
                if self._should_abort():
                    return
                dicts = await format_tokens_as_dicts_async(tokens, decile_maps)
                label = "POSITIVE" if is_positive else "NEGATIVE"
                guidance = _outcome_guidance(is_positive, self._rule.title)
                kwargs = {
                    **self._demographics,
                    "past_summary": self._past_summary,
                    "present_summary": self._present_summary,
                    "trajectory_events": format_events_text(dicts),
                    "outcome_title": self._rule.title,
                    "outcome_label": label,
                    "outcome_guidance": guidance,
                }
                messages = [
                    {"role": "system", "content": tpl["system"].format(**kwargs)},
                    {"role": "user", "content": tpl["user"].format(**kwargs)},
                ]
                async with sem:
                    if self._should_abort():
                        return
                    result = await self._call_or_cancel(self._chat.async_send(messages))
                    if result is None:
                        return
                summary = self._extract_tag(_SUMMARY_RE, result)
                async with lock:
                    self._summary_results[idx] = summary
                    done = len(self._summary_results)
                    self._progress = (done, n_total)
                    self._status = f"Analyzing trajectories\u2026 {done}/{n_total}"
                    _logger.debug(
                        "[%s] %s #%d (%d/%d): %s",
                        self._rule.name,
                        label,
                        idx,
                        done,
                        n_total,
                        summary,
                    )

            await asyncio.gather(*[_summarize_one(i, t, p) for i, t, p in remaining])

        if self._cancelled():
            return

        # ── Collect summaries ─────────────────────────────────────
        pos_summaries = [
            self._summary_results[i]
            for i, (_, is_pos) in enumerate(self._sampled)
            if i in self._summary_results and is_pos and self._summary_results[i]
        ]
        neg_summaries = [
            self._summary_results[i]
            for i, (_, is_pos) in enumerate(self._sampled)
            if i in self._summary_results and not is_pos and self._summary_results[i]
        ]
        self._progress = None
        n_summarized = len(pos_summaries) + len(neg_summaries)

        _logger.debug(
            "Trajectory summaries for %s: %d positive, %d negative (total=%d/%d, stopped_early=%s)",
            self._rule.name,
            len(pos_summaries),
            len(neg_summaries),
            n_summarized,
            n_total,
            self._stopped(),
        )

        if n_summarized == 0:
            self._status = None
            return

        # ── 3. Final explanation (streamed) ───────────────────────
        self._status = "Generating score overview\u2026"

        with open(PROMPTS_DIR / "outcome_explanation.yaml") as f:
            expl_tpl = yaml.safe_load(f)

        pct = self._probability * 100
        margin_pct = self._margin * 100
        guidance = _probability_guidance(self._probability, self._rule.title)

        pos_text = (
            "\n\n".join(f"[POSITIVE trajectory {i + 1}] {s}" for i, s in enumerate(pos_summaries))
            or "No positive trajectories in the sample."
        )
        neg_text = (
            "\n\n".join(f"[NEGATIVE trajectory {i + 1}] {s}" for i, s in enumerate(neg_summaries))
            or "No negative trajectories in the sample."
        )

        kwargs = {
            "outcome_title": self._rule.title,
            "probability_pct": f"{pct:.0f}",
            "margin_pct": f"{margin_pct:.1f}",
            "n_positive": len(pos_summaries),
            "n_negative": len(neg_summaries),
            "probability_guidance": guidance,
            "positive_summaries": pos_text,
            "negative_summaries": neg_text,
        }
        messages = [
            {"role": "system", "content": expl_tpl["system"].format(**kwargs)},
            {"role": "user", "content": expl_tpl["user"].format(**kwargs)},
        ]

        full_text = ""
        tag_open = "<EXPLANATION>"
        stream = self._chat.stream(messages, stop=["</EXPLANATION>"])
        try:
            for delta in stream:
                if self._cancelled():
                    return
                full_text += delta
                if tag_open in full_text:
                    self._status = None
                    self._text = full_text.split(tag_open, 1)[1].strip()
        finally:
            stream.close()

        self._status = None
        if tag_open in full_text:
            self._text = full_text.split(tag_open, 1)[1].strip() or None
        else:
            self._text = full_text.strip() or None
