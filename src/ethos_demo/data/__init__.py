"""Data subpackage — re-exports all public names for backward compatibility."""

from .datasets import (
    SampleContext,
    build_sample_labels,
    find_sample_idx,
    format_timedelta,
    get_admission_order,
    get_allowed_token_ids,
    get_sample_context,
    get_sample_context_stats,
    get_sample_identity,
    get_sample_prompt,
    get_token_ids_by_prefix,
    load_dataset,
    resolve_stop_token_ids,
    sample_indices,
)
from .demographics import (
    get_patient_bmi_group,
    get_patient_demographics,
)
from .quantiles import (
    build_decile_label_maps,
    load_quantiles,
)
from .tokens import format_events_text, format_tokens_as_dicts, format_tokens_as_dicts_async

__all__ = [
    "SampleContext",
    "build_decile_label_maps",
    "build_sample_labels",
    "find_sample_idx",
    "format_events_text",
    "format_timedelta",
    "format_tokens_as_dicts",
    "format_tokens_as_dicts_async",
    "get_admission_order",
    "get_allowed_token_ids",
    "get_patient_bmi_group",
    "get_patient_demographics",
    "get_sample_context",
    "get_sample_context_stats",
    "get_sample_identity",
    "get_sample_prompt",
    "get_token_ids_by_prefix",
    "load_dataset",
    "load_quantiles",
    "resolve_stop_token_ids",
    "sample_indices",
]
