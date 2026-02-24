"""Quantile / decile loading, formatting, and label-map building."""

import json

import streamlit as st

from ..config import TOKENIZED_DATASETS_DIR


@st.cache_data
def load_quantiles(dataset_name: str) -> dict[str, list[float]]:
    """Load all quantile breaks from quantiles.json for the given dataset."""
    qf = TOKENIZED_DATASETS_DIR / dataset_name / "test" / "quantiles.json"
    if not qf.is_file():
        return {}
    with qf.open() as f:
        return json.load(f)


def _inner_breaks(raw_breaks: list[float]) -> list[float]:
    """Strip dataset-extreme min/max and return inner quantile boundaries.

    Follows the same convention as ``transform_to_quantiles`` in the ethos
    tokenization pipeline.
    """
    if len(raw_breaks) > 2:
        return raw_breaks[1:-1]
    return raw_breaks[:1]


def _format_decile_label(inner: list[float], q_num: int, total_deciles: int) -> str:
    """Return a human-readable range label for decile *q_num* (1-based)."""
    if total_deciles == 1:
        return f"= {inner[0]:g}"
    if q_num == 1:
        return f"< {inner[0]:g}"
    if q_num > len(inner):
        return f">= {inner[-1]:g}"
    return f"{inner[q_num - 2]:g} - {inner[q_num - 1]:g}"


@st.cache_data
def build_decile_label_maps(dataset_name: str) -> dict[str, dict[str, str]]:
    """Pre-compute ``{key: annotated_decile}`` mappings for ``pl.replace()``.

    Returns a dict with four sub-dicts:
    - ``"vital"``: composite keys ``"{name}||{d}" -> "{d} [{range}]"``
    - ``"lab"``:   composite keys ``"{name}||{d}" -> "{d} [{range}]"``
    - ``"sbp"``:   simple keys ``"{d}" -> "{d} [{range}]"``
    - ``"dbp"``:   simple keys ``"{d}" -> "{d} [{range}]"``
    """
    all_q = load_quantiles(dataset_name)
    vital: dict[str, str] = {}
    lab: dict[str, str] = {}
    sbp: dict[str, str] = {}
    dbp: dict[str, str] = {}

    for key, raw in all_q.items():
        inner = _inner_breaks(raw)
        n = len(inner) + 1 if len(raw) > 1 else 1
        labels = {str(d): f"{d} [{_format_decile_label(inner, d, n)}]" for d in range(1, n + 1)}

        if key.startswith("VITAL//Q//"):
            name = key.removeprefix("VITAL//Q//")
            if name == "SBP":
                sbp = labels
            elif name == "DBP":
                dbp = labels
            else:
                vital.update({f"{name}||{d}": v for d, v in labels.items()})
        elif key.startswith("LAB//Q//"):
            name = key.removeprefix("LAB//Q//")
            lab.update({f"{name}||{d}": v for d, v in labels.items()})

    return {"vital": vital, "lab": lab, "sbp": sbp, "dbp": dbp}
