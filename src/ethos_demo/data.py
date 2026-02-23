"""Dataset loading and patient-level helpers for the ETHOS Demo app."""

import json
import logging
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import streamlit as st
from ethos.constants import SpecialToken
from ethos.datasets import InferenceDataset
from ethos.utils import group_tokens_by_info

from ethos_demo.config import TOKENIZED_DATASETS_DIR

logger = logging.getLogger(__name__)


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


def get_patient_demographics(
    dataset: InferenceDataset,
    idx: int,
    *,
    reference_time_us: int | None = None,
) -> dict[str, str]:
    """Return Gender, Race, Marital Status and Age for sample *idx*.

    If *reference_time_us* is given (microseconds), demographics are resolved at that point in time;
    otherwise the prediction time is used.
    """
    start_idx = dataset.start_indices[idx].item()
    patient_id = dataset.patient_id_at_idx[start_idx].item()
    ts_us = (
        reference_time_us if reference_time_us is not None else int(dataset.times[start_idx].item())
    )
    ref_time = datetime.fromtimestamp(int(ts_us / 1e6), tz=UTC).replace(tzinfo=None)
    static = dataset.static_data[patient_id]

    demographics: dict[str, str] = {}
    for prefix, data in static.items():
        code = data["code"][0]

        if code == "MEDS_BIRTH":
            age_years = (ref_time - data["time"][0]).days / 365.25
            demographics["Age"] = f"{age_years:.0f}"
            continue

        if len(data["code"]) > 1:
            time_idx = _find_idx_of_last_le(data["time"], ref_time)
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


def get_timeline_times_us(dataset: InferenceDataset, idx: int) -> tuple[int, int]:
    """Return (timeline_start_us, prediction_time_us) for sample *idx*."""
    timeline_start, start_idx = _timeline_bounds(dataset, idx)
    return int(dataset.times[timeline_start].item()), int(dataset.times[start_idx].item())


def _decode_tokens(dataset: InferenceDataset, lo: int, hi: int) -> list[str]:
    """Decode tokens in [lo, hi] and return non-None values."""
    tokens = dataset.vocab.decode(dataset.tokens[lo : hi + 1])
    return [t for t in tokens if t is not None]


def format_tokens_as_dicts(tokens: list[str]) -> list[dict]:
    """Convert raw decoded tokens into a list of dicts (one per clinical event).

    Uses the Polars pipeline ported from the ETHOS notebook to group related tokens, pivot
    categories into columns, and strip null values.
    """
    if not tokens:
        return []

    groups = group_tokens_by_info(tokens)

    df = (
        pl.DataFrame([groups, pl.Series("token", tokens)])
        .with_columns(
            token=pl.when(pl.col("token").str.starts_with("BMI//"))
            .then(pl.concat_list(pl.lit("BMI"), pl.col("token").str.slice(len("BMI//"))))
            .otherwise(pl.concat_list("token"))
        )
        .explode("token")
        .select(
            "groups",
            cat=pl.when(pl.col("token").str.starts_with("HOSPITAL//"))
            .then(pl.col("token").str.splitn("//", 3).struct[1])
            .otherwise(pl.col("token").str.splitn("//", 2).struct[0])
            .replace("QUANTILE", "DECILE"),
            token=pl.when(pl.col("token").str.starts_with("ICD"))
            .then(pl.col("token").str.split("//").list.last())
            .when(pl.col("token").str.starts_with("LAB//NAME//"))
            .then(pl.col("token").str.slice(len("LAB//NAME//")).str.splitn("//", 1).struct[0])
            .otherwise(pl.col("token").str.split("//").list.last()),
        )
        .with_columns(
            cat=pl.when(pl.col("cat") == pl.col("token")).then(pl.lit("EVENT")).otherwise("cat")
        )
        .group_by("groups", maintain_order=True)
        .agg(
            "cat",
            pl.when(pl.col("cat").is_in(["ATC", "ICD_CM"]))
            .then(pl.col("token").str.join(""))
            .otherwise(pl.col("token")),
        )
        .with_columns(
            cat=pl.when(pl.col("token").list[0] == "BLOOD_PRESSURE")
            .then(["VITAL", "SBP_DECILE", "DBP_DECILE"])
            .otherwise("cat")
        )
        .explode("cat", "token")
        .with_row_index("rid")
        .pivot(
            index=["rid", "groups"],
            on="cat",
            values="token",
            aggregate_function="first",
        )
        .drop("rid")
        .group_by("groups", maintain_order=True)
        .agg(pl.exclude("groups").drop_nulls().first())
        .drop("groups")
    )

    return [{k: v for k, v in d.items() if v is not None} for d in df.to_dicts()]


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
