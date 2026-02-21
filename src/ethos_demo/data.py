"""Dataset loading and patient-level helpers for the ETHOS Demo app."""

import json
import logging
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import streamlit as st
from ethos.datasets import InferenceDataset

from ethos_demo.config import TOKENIZED_DATASETS_DIR

logger = logging.getLogger(__name__)


@st.cache_resource
def load_dataset(dataset_name: str, task: str) -> InferenceDataset:
    input_dir = TOKENIZED_DATASETS_DIR / dataset_name / "test"
    return InferenceDataset.from_task(task, input_dir=input_dir)


def sample_indices(
    dataset: InferenceDataset,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample `n` indices from *dataset*."""
    total = len(dataset)
    np.random.seed(seed)
    return np.random.choice(total, min(n, total), replace=False)


def get_sample_identity(dataset: InferenceDataset, idx: int) -> tuple[int, int]:
    """Return (patient_id, prediction_time_us) that uniquely identify a sample."""
    start_idx = dataset.start_indices[idx].item()
    patient_id = dataset.patient_id_at_idx[start_idx].item()
    prediction_time = int(dataset.times[start_idx].item())
    return patient_id, prediction_time


def find_sample_idx(
    dataset: InferenceDataset,
    patient_id: int,
    prediction_time: int,
) -> int:
    """Find the dataset index whose (patient_id, prediction_time) matches."""
    for idx in range(len(dataset)):
        si = dataset.start_indices[idx].item()
        if (
            dataset.patient_id_at_idx[si].item() == patient_id
            and int(dataset.times[si].item()) == prediction_time
        ):
            return idx
    raise ValueError(f"No matching sample for patient {patient_id} at time {prediction_time}")


def get_patient_demographics(dataset: InferenceDataset, idx: int) -> dict[str, str]:
    """Return Gender, Race, Marital Status and Age for sample *idx*."""
    start_idx = dataset.start_indices[idx].item()
    patient_id = dataset.patient_id_at_idx[start_idx].item()
    time_at_start = datetime.fromtimestamp(
        int(dataset.times[start_idx].item() / 1e6),
        tz=UTC,
    ).replace(tzinfo=None)
    static = dataset.static_data[patient_id]

    demographics: dict[str, str] = {}
    for prefix, data in static.items():
        code = data["code"][0]

        if code == "MEDS_BIRTH":
            age_years = (time_at_start - data["time"][0]).days / 365.25
            demographics["Age"] = f"{age_years:.0f}"
            continue

        # Time-varying fields: pick the value at prediction time
        if len(data["code"]) > 1:
            time_idx = _find_idx_of_last_le(data["time"], time_at_start)
            code = f"{prefix}//UNKNOWN" if time_idx == -1 else data["code"][time_idx]

        value = code.split("//", 1)[-1] if "//" in code else code
        if value == "UNKNOWN":
            value = "???"

        if prefix == "GENDER":
            demographics["Gender"] = "Male" if value == "M" else "Female"
        elif prefix == "RACE":
            demographics["Race"] = value.title()
        elif prefix in ("MARITAL", "MARITAL_STATUS"):
            demographics["Marital Status"] = value.title()

    return demographics


def _find_idx_of_last_le(times: list[datetime], value: datetime) -> int:
    indices = [i for i, t in enumerate(times) if t <= value]
    return indices[-1] if indices else -1


@st.cache_resource
def _load_bmi_quantiles(dataset_name: str) -> list[float] | None:
    """Load BMI quantile breaks (including min/max) from quantiles.json."""
    qf = TOKENIZED_DATASETS_DIR / dataset_name / "test" / "quantiles.json"
    if not qf.is_file():
        return None
    with qf.open() as f:
        data = json.load(f)
    for key in ("BMI//Q", "VITAL//BMI"):
        if key in data:
            return data[key]
    return None


def get_patient_bmi_group(dataset: InferenceDataset, idx: int, dataset_name: str) -> str:
    """Return BMI group label for sample *idx*, e.g. '23-25 (D3)'."""
    start_idx = dataset.start_indices[idx].item()
    timeline_start_idx = dataset.patient_offset_at_idx[start_idx].item()
    if start_idx - timeline_start_idx + 1 > dataset.timeline_size:
        timeline_start_idx = start_idx + 1 - dataset.timeline_size

    tokens = dataset.tokens[timeline_start_idx : start_idx + 1]
    decoded = dataset.vocab.decode(tokens)

    q_num = None
    for token in reversed(decoded):
        if token and "BMI" in token and "QUANTILE" in token:
            q_num = int(token.rsplit("//", 1)[-1])
            break

    if q_num is None:
        return "???"

    breaks = _load_bmi_quantiles(dataset_name)
    if breaks is None or q_num < 1 or q_num > len(breaks) - 1:
        return f"D{q_num}"

    lo = round(breaks[q_num - 1])
    hi = round(breaks[q_num])
    return f"{lo}-{hi} (D{q_num})"


def get_admission_order(dataset: InferenceDataset, idx: int) -> int:
    """1-based rank of the sample among the same patient's cases."""
    start_idx = dataset.start_indices[idx].item()
    pt_offset = dataset.patient_offset_at_idx[start_idx].item()
    pt_end = dataset.patient_data_end_at_idx[start_idx].item()

    mask = (dataset.start_indices >= pt_offset) & (dataset.start_indices < pt_end)
    patient_starts = dataset.start_indices[mask].sort().values
    return (patient_starts == start_idx).nonzero(as_tuple=False).item() + 1


def build_sample_labels(dataset: InferenceDataset, indices: np.ndarray) -> list[tuple[int, str]]:
    """Return (dataset_idx, display_label) pairs for the sampled indices.

    The label includes an admission-order suffix only when a patient appears more than once in the
    sample set, and always shows the number of events in the history.
    """
    patient_ids = [
        dataset.patient_id_at_idx[dataset.start_indices[int(i)].item()].item() for i in indices
    ]
    pid_counts = Counter(patient_ids)

    labels: list[tuple[int, str]] = []
    for i, pid in zip(indices, patient_ids, strict=False):
        x, _y = dataset[int(i)]
        n_events = len(x["input_ids"])
        if pid_counts[pid] > 1:
            order = get_admission_order(dataset, int(i))
            label = f"Patient {pid} — Case #{order} ({n_events} events)"
        else:
            label = f"Patient {pid} ({n_events} events)"
        labels.append((int(i), label))

    return labels


def get_sample_prompt(dataset: InferenceDataset, idx: int) -> tuple[str, int]:
    """Return (prompt_text, n_input_tokens) for sample *idx*."""
    x, _y = dataset[idx]
    input_ids = x["input_ids"]
    tokens = dataset.vocab.decode(input_ids)
    prompt = " ".join(tokens)
    return prompt, len(input_ids)


_24H_US = int(timedelta(hours=24) / timedelta(microseconds=1))


def _timeline_bounds(dataset: InferenceDataset, idx: int) -> tuple[int, int]:
    """Return (timeline_start, start_idx) for sample *idx*."""
    start_idx = dataset.start_indices[idx].item()
    timeline_start_idx = dataset.patient_offset_at_idx[start_idx].item()
    if start_idx - timeline_start_idx + 1 > dataset.timeline_size:
        timeline_start_idx = start_idx + 1 - dataset.timeline_size
    return timeline_start_idx, start_idx


def _decode_window(dataset: InferenceDataset, lo: int, hi: int) -> str:
    """Decode tokens in [lo, hi] and join non-None values."""
    tokens = dataset.vocab.decode(dataset.tokens[lo : hi + 1])
    return " ".join(t for t in tokens if t is not None)


def get_triage_history(dataset: InferenceDataset, idx: int) -> str:
    """Decode tokens sharing the same timestamp as the prediction time (triage events)."""
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    prediction_time_us = int(dataset.times[start_idx].item())

    times_slice = dataset.times[timeline_start : start_idx + 1]
    mask = (times_slice == prediction_time_us).nonzero(as_tuple=False)
    window_start = timeline_start + int(mask[0].item())
    return _decode_window(dataset, window_start, start_idx)


def get_last_24h_history(dataset: InferenceDataset, idx: int) -> str:
    """Decode the timeline tokens from the last 24 hours before prediction time."""
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    prediction_time_us = int(dataset.times[start_idx].item())
    cutoff_us = prediction_time_us - _24H_US

    times_slice = dataset.times[timeline_start : start_idx + 1]
    first_in_window = int((times_slice >= cutoff_us).nonzero(as_tuple=False)[0].item())
    window_start = timeline_start + first_in_window
    return _decode_window(dataset, window_start, start_idx)


def get_stay_history(dataset: InferenceDataset, idx: int) -> str:
    """Decode tokens from the most recent HOSPITAL_ADMISSION to prediction time."""
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    token_ids = dataset.tokens[timeline_start : start_idx + 1]
    decoded = dataset.vocab.decode(token_ids)

    # Walk backwards to find the most recent admission boundary
    admission_pos = None
    for i in range(len(decoded) - 1, -1, -1):
        if decoded[i] == "HOSPITAL_ADMISSION":
            admission_pos = i
            break

    window_start = timeline_start + admission_pos if admission_pos is not None else timeline_start
    return _decode_window(dataset, window_start, start_idx)


def get_sample_context_stats(dataset: InferenceDataset, idx: int) -> dict[str, str]:
    """Return EHR history time span and token count for sample *idx*."""
    x, _y = dataset[idx]
    n_tokens = len(x["input_ids"])

    start_idx = dataset.start_indices[idx].item()
    timeline_start_idx = dataset.patient_offset_at_idx[start_idx].item()
    if start_idx - timeline_start_idx + 1 > dataset.timeline_size:
        timeline_start_idx = start_idx + 1 - dataset.timeline_size

    time_start_us = int(dataset.times[timeline_start_idx].item())
    time_end_us = int(dataset.times[start_idx].item())
    span = timedelta(microseconds=time_end_us - time_start_us)

    return {
        "Time": _format_timedelta(span),
        "Tokens": str(n_tokens),
    }


def _format_timedelta(td: timedelta) -> str:
    total_days = td.days
    total_months = round(total_days / 30.44)
    years, months = divmod(total_months, 12)
    if years > 0:
        return f"{years}y {months}mt" if months else f"{years}y"
    if months > 0:
        days = total_days - round(months * 30.44)
        return f"{months}mt {days}d" if days > 0 else f"{months}mt"
    return f"{total_days}d"
