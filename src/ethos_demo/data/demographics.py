"""Patient demographic extraction and BMI group labelling."""

from datetime import UTC, datetime

from ethos.datasets import InferenceDataset

from .quantiles import _format_decile_label, _inner_breaks, load_quantiles


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


def get_patient_bmi_group(dataset: InferenceDataset, idx: int, dataset_name: str) -> str:
    """Return BMI group label for sample *idx*, e.g. '< 21 (D1)'."""
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

    all_q = load_quantiles(dataset_name)
    raw = None
    for key in ("BMI//Q", "VITAL//BMI"):
        if key in all_q:
            raw = all_q[key]
            break

    if raw is None:
        return f"D{q_num}"

    inner = _inner_breaks(raw)
    n_deciles = len(inner) + 1 if len(raw) > 1 else 1
    label = _format_decile_label(inner, q_num, n_deciles)
    return f"{label} (D{q_num})"
