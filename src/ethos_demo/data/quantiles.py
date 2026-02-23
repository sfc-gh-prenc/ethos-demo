"""Quantile / decile loading, formatting, and range lookup (LLM tool backend)."""

import json
import logging

import streamlit as st

from ..config import TOKENIZED_DATASETS_DIR

_logger = logging.getLogger(__name__)

_QUANTILE_KEY_PREFIXES = ("LAB//Q//", "VITAL//Q//", "")

_BP_ALIASES = {"BLOOD_PRESSURE", "BP"}


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


def _lookup_single(all_q: dict[str, list[float]], name: str) -> dict[str, str] | str:
    raw = None
    if name.upper() == "BMI":
        raw = all_q.get("BMI//Q")
    else:
        for prefix in _QUANTILE_KEY_PREFIXES:
            raw = all_q.get(f"{prefix}{name}")
            if raw is not None:
                break

    if raw is None:
        _logger.warning("Decile range not found for %r", name)
        return "not available"

    inner = _inner_breaks(raw)
    n_deciles = len(inner) + 1 if len(raw) > 1 else 1
    return {f"D{d}": _format_decile_label(inner, d, n_deciles) for d in range(1, n_deciles + 1)}


def get_decile_ranges(dataset_name: str, names: list[str] | str) -> dict[str, dict[str, str] | str]:
    """Return decile-range mappings for the requested measurement names.

    Accepts lab names (e.g. ``HEMOGLOBIN//G/DL//BLOOD``), vital names
    (e.g. ``HEART_RATE``, ``SBP``, ``DBP``), ``BMI``, or
    ``BLOOD_PRESSURE`` / ``BP`` (expands to SBP + DBP).

    *names* may be a list or a JSON-encoded string (LLMs sometimes
    double-serialize the argument).
    """
    if isinstance(names, str):
        names = json.loads(names)
    all_q = load_quantiles(dataset_name)
    result: dict[str, dict[str, str] | str] = {}

    for name in names:
        if name.upper() in _BP_ALIASES:
            result["SBP"] = _lookup_single(all_q, "SBP")
            result["DBP"] = _lookup_single(all_q, "DBP")
        else:
            result[name] = _lookup_single(all_q, name)

    return result
