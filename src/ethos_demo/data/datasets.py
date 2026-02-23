"""Dataset loading, sampling, and sample-level helpers."""

import json
from collections import Counter
from datetime import timedelta

import numpy as np
import streamlit as st
from ethos.constants import SpecialToken
from ethos.datasets import InferenceDataset

from ..config import TOKENIZED_DATASETS_DIR


@st.cache_resource
def load_dataset(dataset_name: str, task: str) -> InferenceDataset:
    input_dir = TOKENIZED_DATASETS_DIR / dataset_name / "test"
    return InferenceDataset.from_task(task, input_dir=input_dir)


@st.cache_data
def get_allowed_token_ids(dataset_name: str, task: str) -> list[int] | None:
    """Return constrained-generation token IDs from code_counts.json, or None."""
    counts_fp = TOKENIZED_DATASETS_DIR / dataset_name / "test" / "code_counts.json"
    if not counts_fp.is_file():
        return None
    ds = load_dataset(dataset_name, task)
    with counts_fp.open() as f:
        codes = [c for c in json.load(f) if c != SpecialToken.TIMELINE_START]
    ids = [i for i in ds.vocab.encode(codes) if i is not None]
    return ids or None


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
        "Time": format_timedelta(span),
        "Tokens": str(n_tokens),
    }


def format_timedelta(td: timedelta) -> str:
    total_days = td.days
    total_months = round(total_days / 30.44)
    years, months = divmod(total_months, 12)
    if years > 0:
        return f"{years}y {months}mt" if months else f"{years}y"
    if months > 0:
        days = total_days - round(months * 30.44)
        return f"{months}mt {days}d" if days > 0 else f"{months}mt"
    return f"{total_days}d"
