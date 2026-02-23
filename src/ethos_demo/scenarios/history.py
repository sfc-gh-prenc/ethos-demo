"""Scenario-specific timeline extraction functions."""

from dataclasses import dataclass
from datetime import timedelta

from ethos.datasets import InferenceDataset

_36H_US = int(timedelta(hours=36) / timedelta(microseconds=1))


@dataclass(frozen=True)
class HistorySplit:
    """Result of splitting a patient timeline into past history and current encounter."""

    past_tokens: list[str]
    present_tokens: list[str]
    past_time_span: timedelta
    present_time_span: timedelta


def _timeline_bounds(dataset: InferenceDataset, idx: int) -> tuple[int, int]:
    """Return (timeline_start, start_idx) for sample *idx*."""
    start_idx = dataset.start_indices[idx].item()
    timeline_start_idx = dataset.patient_offset_at_idx[start_idx].item()
    if start_idx - timeline_start_idx + 1 > dataset.timeline_size:
        timeline_start_idx = start_idx + 1 - dataset.timeline_size
    return timeline_start_idx, start_idx


def get_timeline_times_us(dataset: InferenceDataset, idx: int) -> tuple[int, int]:
    """Return (timeline_start_us, prediction_time_us) for sample *idx*."""
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    return int(dataset.times[timeline_start].item()), int(dataset.times[start_idx].item())


def _decode_tokens(dataset: InferenceDataset, lo: int, hi: int) -> list[str]:
    """Decode tokens in [lo, hi] and return non-None values."""
    tokens = dataset.vocab.decode(dataset.tokens[lo : hi + 1])
    return [t for t in tokens if t is not None]


def _time_span(dataset: InferenceDataset, lo: int, hi: int) -> timedelta:
    """Return the time span between indices *lo* and *hi*."""
    if hi <= lo:
        return timedelta(0)
    return timedelta(microseconds=int(dataset.times[hi].item()) - int(dataset.times[lo].item()))


def _build_split(
    dataset: InferenceDataset,
    timeline_start: int,
    window_start: int,
    start_idx: int,
) -> HistorySplit:
    """Construct a HistorySplit from the boundary indices.

    The leading TIME_INTERVAL token at the boundary is moved into history because it represents
    elapsed time *before* the current encounter. The history time span covers the full record up to
    prediction time.
    """
    interval_tokens = dataset.vocab.interval_estimates.get("mean", {})
    if window_start <= start_idx:
        first = dataset.vocab.decode(dataset.tokens[window_start : window_start + 1])
        if first and first[0] in interval_tokens:
            window_start += 1

    if window_start > timeline_start:
        past = _decode_tokens(dataset, timeline_start, window_start - 1)
        past_span = _time_span(dataset, timeline_start, window_start - 1)
    else:
        past = []
        past_span = timedelta(0)

    if window_start <= start_idx:
        present = _decode_tokens(dataset, window_start, start_idx)
        present_span = _time_span(dataset, window_start, start_idx)
    else:
        present = []
        present_span = timedelta(0)

    return HistorySplit(
        past_tokens=past,
        present_tokens=present,
        past_time_span=past_span,
        present_time_span=present_span,
    )


def get_triage_history(dataset: InferenceDataset, idx: int) -> HistorySplit:
    """Split timeline for a triage scenario.

    Current encounter = tokens sharing the same timestamp as prediction time. EHR history =
    everything before that timestamp.
    """
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    prediction_time_us = int(dataset.times[start_idx].item())

    times_slice = dataset.times[timeline_start : start_idx + 1]
    mask = (times_slice == prediction_time_us).nonzero(as_tuple=False)
    window_start = timeline_start + int(mask[0].item())

    return _build_split(dataset, timeline_start, window_start, start_idx)


def get_last_36h_history(dataset: InferenceDataset, idx: int) -> HistorySplit:
    """Split timeline for a hospital-admission scenario.

    Current encounter = tokens from the last 36 hours before prediction time. EHR history =
    everything before the 36-hour window.
    """
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    prediction_time_us = int(dataset.times[start_idx].item())
    cutoff_us = prediction_time_us - _36H_US

    times_slice = dataset.times[timeline_start : start_idx + 1]
    first_in_window = int((times_slice >= cutoff_us).nonzero(as_tuple=False)[0].item())
    window_start = timeline_start + first_in_window

    return _build_split(dataset, timeline_start, window_start, start_idx)


def get_stay_history(dataset: InferenceDataset, idx: int) -> HistorySplit:
    """Split timeline for a hospital-discharge scenario.

    Current encounter = 36h before the most recent HOSPITAL_ADMISSION + the entire stay up to
    prediction time. EHR history = everything before that expanded window.
    """
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    token_ids = dataset.tokens[timeline_start : start_idx + 1]
    decoded = dataset.vocab.decode(token_ids)

    admission_pos = None
    for i in range(len(decoded) - 1, -1, -1):
        if decoded[i] == "HOSPITAL_ADMISSION":
            admission_pos = i
            break

    admission_idx = timeline_start + admission_pos if admission_pos is not None else timeline_start

    # Expand window to 36h before the admission timestamp
    admission_time_us = int(dataset.times[admission_idx].item())
    pre_cutoff_us = admission_time_us - _36H_US

    times_slice = dataset.times[timeline_start : admission_idx + 1]
    hits = (times_slice >= pre_cutoff_us).nonzero(as_tuple=False)
    window_start = timeline_start + int(hits[0].item()) if len(hits) > 0 else timeline_start

    return _build_split(dataset, timeline_start, window_start, start_idx)
