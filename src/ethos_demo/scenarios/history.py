"""Scenario-specific timeline extraction functions."""

from datetime import timedelta

from ethos.datasets import InferenceDataset

_24H_US = int(timedelta(hours=24) / timedelta(microseconds=1))


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


def get_triage_history(dataset: InferenceDataset, idx: int) -> tuple[list[str], list[str]]:
    """Return (past_tokens, present_tokens) for a triage scenario.

    Present = tokens sharing the same timestamp as prediction time. Past = everything before that
    timestamp.
    """
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    prediction_time_us = int(dataset.times[start_idx].item())

    times_slice = dataset.times[timeline_start : start_idx + 1]
    mask = (times_slice == prediction_time_us).nonzero(as_tuple=False)
    window_start = timeline_start + int(mask[0].item())

    past = (
        _decode_tokens(dataset, timeline_start, window_start - 1)
        if window_start > timeline_start
        else []
    )
    present = _decode_tokens(dataset, window_start, start_idx)
    return past, present


def get_last_24h_history(dataset: InferenceDataset, idx: int) -> tuple[list[str], list[str]]:
    """Return (past_tokens, present_tokens) for a hospital-admission scenario.

    Present = tokens from the last 24 hours before prediction time. Past = everything before the
    24-hour window.
    """
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    prediction_time_us = int(dataset.times[start_idx].item())
    cutoff_us = prediction_time_us - _24H_US

    times_slice = dataset.times[timeline_start : start_idx + 1]
    first_in_window = int((times_slice >= cutoff_us).nonzero(as_tuple=False)[0].item())
    window_start = timeline_start + first_in_window

    past = (
        _decode_tokens(dataset, timeline_start, window_start - 1)
        if window_start > timeline_start
        else []
    )
    present = _decode_tokens(dataset, window_start, start_idx)
    return past, present


def get_stay_history(dataset: InferenceDataset, idx: int) -> tuple[list[str], list[str]]:
    """Return (past_tokens, present_tokens) for a hospital-discharge scenario.

    Present = tokens from the most recent HOSPITAL_ADMISSION to prediction time. Past = everything
    before that admission.
    """
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    token_ids = dataset.tokens[timeline_start : start_idx + 1]
    decoded = dataset.vocab.decode(token_ids)

    admission_pos = None
    for i in range(len(decoded) - 1, -1, -1):
        if decoded[i] == "HOSPITAL_ADMISSION":
            admission_pos = i
            break

    window_start = timeline_start + admission_pos if admission_pos is not None else timeline_start

    past = (
        _decode_tokens(dataset, timeline_start, window_start - 1)
        if window_start > timeline_start
        else []
    )
    present = _decode_tokens(dataset, window_start, start_idx)
    return past, present
