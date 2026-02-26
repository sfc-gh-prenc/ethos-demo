"""Dataset loading, sampling, and sample-level helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st
from ethos.constants import SpecialToken
from ethos.datasets import InferenceDataset

from ..config import TOKENIZED_DATASETS_DIR

if TYPE_CHECKING:
    from ..scenarios import HistorySplit, Scenario


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
    min_events: int = 100,
) -> np.ndarray:
    """Sample `n` indices from *dataset*, keeping only samples with enough history."""
    eligible = np.array(
        [i for i in range(len(dataset)) if _n_timeline_tokens(dataset, i) >= min_events]
    )
    if len(eligible) == 0:
        eligible = np.arange(len(dataset))
    np.random.seed(seed)
    return np.random.choice(eligible, min(n, len(eligible)), replace=False)


def _n_timeline_tokens(dataset: InferenceDataset, idx: int) -> int:
    """Number of tokens in the timeline window for sample *idx*."""
    start_idx = dataset.start_indices[idx].item()
    timeline_start = dataset.patient_offset_at_idx[start_idx].item()
    return min(start_idx - timeline_start + 1, dataset.timeline_size)


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
        n_events = len(x["input_ids"]) - dataset.static_ctx_size
        if pid_counts[pid] > 1:
            order = get_admission_order(dataset, int(i))
            label = f"Patient {pid} — Case #{order} ({n_events:,} events)"
        else:
            label = f"Patient {pid} ({n_events:,} events)"
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
    total_seconds = td.total_seconds()
    if total_seconds == 0:
        return "\u2013"
    total_days = td.days
    if total_days < 1:
        hours = int(total_seconds // 3600)
        return f"{hours}h"
    total_months = round(total_days / 30.44)
    years, months = divmod(total_months, 12)
    if years > 0:
        return f"{years}y {months}mt" if months else f"{years}y"
    if months > 0:
        days = total_days - round(months * 30.44)
        return f"{months}mt {days}d" if days > 0 else f"{months}mt"
    return f"{total_days}d"


_DEMO_ORDER = ["Gender", "Race", "Age", "BMI", "Marital Status"]


class SampleContext:
    """Pre-computed sample pool for a dataset+task combination.

    Caches per-patient demographics, BMI, and history splits so they are computed at most once per
    sample selection.
    """

    def __init__(self, dataset_name: str, task: str, n_samples: int, seed: int):
        self.dataset = load_dataset(dataset_name, task)
        self.dataset_name = dataset_name
        self.task = task
        indices = sample_indices(self.dataset, n_samples, seed)
        self.labels = build_sample_labels(self.dataset, indices)
        self.label_to_idx: dict[str, int] = {label: idx for idx, label in self.labels}
        self._demo_cache: dict[int, dict[str, str]] = {}
        self._split_cache: dict[tuple[int, str], HistorySplit] = {}

    def demographics(self, idx: int) -> dict[str, str]:
        """Return ordered demographics dict for sample *idx* (cached)."""
        if idx not in self._demo_cache:
            from .demographics import get_patient_bmi_group, get_patient_demographics

            raw = get_patient_demographics(self.dataset, idx)
            raw["BMI"] = get_patient_bmi_group(self.dataset, idx, self.dataset_name)
            self._demo_cache[idx] = {k: raw.get(k, "???") for k in _DEMO_ORDER}
        return self._demo_cache[idx]

    def history_split(self, idx: int, scenario: Scenario) -> HistorySplit:
        """Return history/encounter split for sample *idx* under *scenario* (cached)."""
        key = (idx, str(scenario))
        if key not in self._split_cache:
            from ..scenarios import SCENARIOS

            self._split_cache[key] = SCENARIOS[scenario].history_fn(self.dataset, idx)
        return self._split_cache[key]


@st.cache_resource
def get_sample_context(
    dataset_name: str,
    task: str,
    n_samples: int,
    seed: int,
) -> SampleContext:
    """Return a cached `SampleContext` for the given dataset+task combination."""
    return SampleContext(dataset_name, task, n_samples, seed)
