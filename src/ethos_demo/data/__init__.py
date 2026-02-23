"""Data subpackage — re-exports all public names for backward compatibility."""

from .datasets import (
    build_sample_labels,
    find_sample_idx,
    get_admission_order,
    get_allowed_token_ids,
    get_sample_context_stats,
    get_sample_identity,
    get_sample_prompt,
    load_dataset,
    sample_indices,
)
from .demographics import (
    get_patient_bmi_group,
    get_patient_demographics,
)
from .quantiles import (
    get_decile_ranges,
    load_quantiles,
)
from .tokens import format_tokens_as_dicts

__all__ = [
    "build_sample_labels",
    "find_sample_idx",
    "format_tokens_as_dicts",
    "get_admission_order",
    "get_allowed_token_ids",
    "get_decile_ranges",
    "get_patient_bmi_group",
    "get_patient_demographics",
    "get_sample_context_stats",
    "get_sample_identity",
    "get_sample_prompt",
    "load_dataset",
    "load_quantiles",
    "sample_indices",
]
