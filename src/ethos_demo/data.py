"""Dataset loading and patient-level helpers for the ETHOS Demo app."""

import logging
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import streamlit as st
from ethos.datasets import InferenceDataset
from ethos.vocabulary import Vocabulary

from ethos_demo.config import TASK_DATASET_MAP, TOKENIZED_DATASETS_DIR

logger = logging.getLogger(__name__)


def _resolve_dataset_task(task: str) -> str:
    """Map an app-level task name to the underlying ethos dataset task."""
    return TASK_DATASET_MAP.get(task, task)


@st.cache_resource
def load_dataset(dataset_name: str, task: str) -> InferenceDataset:
    input_dir = TOKENIZED_DATASETS_DIR / dataset_name / "test"
    return InferenceDataset.from_task(_resolve_dataset_task(task), input_dir=input_dir)


@st.cache_data
def _extract_identities(dataset_name: str, task: str) -> pl.DataFrame:
    """Return (patient_id, prediction_time, idx) for every sample in a task dataset."""
    ds = load_dataset(dataset_name, _resolve_dataset_task(task))
    patient_ids, prediction_times = [], []
    for i in range(len(ds)):
        si = ds.start_indices[i].item()
        patient_ids.append(ds.patient_id_at_idx[si].item())
        prediction_times.append(int(ds.times[si].item()))
    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "prediction_time": prediction_times,
            "idx": list(range(len(ds))),
        }
    )


def sample_common_indices(
    dataset_name: str,
    tasks: list[str],
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample `n` indices (into the first task's dataset) from patients common to all tasks."""
    dfs = [_extract_identities(dataset_name, t).rename({"idx": f"idx_{t}"}) for t in tasks]
    common = dfs[0]
    for df in dfs[1:]:
        common = common.join(df, on=["patient_id", "prediction_time"], how="inner")

    all_indices = common.get_column(f"idx_{tasks[0]}").to_numpy()
    np.random.seed(seed)
    return np.random.choice(all_indices, min(n, len(all_indices)), replace=False)


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
    sample set.
    """
    patient_ids = [
        dataset.patient_id_at_idx[dataset.start_indices[int(i)].item()].item() for i in indices
    ]
    pid_counts = Counter(patient_ids)

    labels: list[tuple[int, str]] = []
    for i, pid in zip(indices, patient_ids, strict=False):
        if pid_counts[pid] > 1:
            order = get_admission_order(dataset, int(i))
            label = f"Patient {pid} — Case #{order}"
        else:
            label = f"Patient {pid}"
        labels.append((int(i), label))

    return labels


def get_sample_prompt(dataset: InferenceDataset, idx: int) -> tuple[str, int, list[str]]:
    """Return (prompt_text, n_input_tokens, stop_tokens) for sample *idx*."""
    x, _y = dataset[idx]
    input_ids = x["input_ids"]
    tokens = dataset.vocab.decode(input_ids)
    prompt = " ".join(tokens)
    stop_tokens = [str(s) for s in dataset.stop_tokens]
    return prompt, len(input_ids), stop_tokens


_24H_US = int(timedelta(hours=24) / timedelta(microseconds=1))


def get_last_24h_history(dataset: InferenceDataset, idx: int) -> str:
    """Decode the timeline tokens from the last 24 hours before prediction time."""
    start_idx = dataset.start_indices[idx].item()
    timeline_start_idx = dataset.patient_offset_at_idx[start_idx].item()
    if start_idx - timeline_start_idx + 1 > dataset.timeline_size:
        timeline_start_idx = start_idx + 1 - dataset.timeline_size

    prediction_time_us = int(dataset.times[start_idx].item())
    cutoff_us = prediction_time_us - _24H_US

    times_slice = dataset.times[timeline_start_idx : start_idx + 1]
    first_in_window = int((times_slice >= cutoff_us).nonzero(as_tuple=False)[0].item())
    window_start = timeline_start_idx + first_in_window

    tokens = dataset.vocab.decode(dataset.tokens[window_start : start_idx + 1])
    return " ".join(t for t in tokens if t is not None)


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
    years, remaining = divmod(total_days, 365)
    months = remaining // 30
    if years > 0:
        return f"{years}y {months}m" if months else f"{years}y"
    if months > 0:
        return f"{months}m {remaining - months * 30}d"
    return f"{total_days}d"


# ── Probability computation ─────────────────────────────────────────────────

_POSITIVE_OUTCOME: dict[str, tuple[set[str], timedelta | None]] = {
    "ed_hospitalization": ({"HOSPITAL_ADMISSION"}, timedelta(days=2)),
    "ed_critical_outcome": ({"ICU_ADMISSION", "MEDS_DEATH"}, timedelta(hours=12)),
    "icu_admission": ({"ICU_ADMISSION", "MEDS_DEATH"}, None),
    "icu_mortality": ({"MEDS_DEATH"}, None),
    "readmission_30d": ({"HOSPITAL_ADMISSION", "MEDS_DEATH"}, timedelta(days=30)),
    "readmission_90d": ({"HOSPITAL_ADMISSION", "MEDS_DEATH"}, timedelta(days=90)),
}


def compute_task_counts(
    responses: list[tuple[str, str]],
    task: str,
    vocab: Vocabulary,
) -> tuple[int, int]:
    """Return (positives, valid) counts from a batch of (generated_text, finish_reason)."""
    positive_tokens, time_limit = _POSITIVE_OUTCOME[task]
    positives = 0
    valid = 0

    for text, finish_reason in responses:
        generated = text.strip().split()
        if not generated:
            continue

        last_token = generated[-1]

        if time_limit is None:
            if finish_reason != "stop":
                continue
            valid += 1
            if last_token in positive_tokens:
                positives += 1
        else:
            valid += 1
            if last_token in positive_tokens:
                try:
                    token_time = vocab.get_timeline_total_time(generated)
                except Exception:
                    token_time = timedelta(0)
                if token_time <= time_limit:
                    positives += 1

    return positives, valid
