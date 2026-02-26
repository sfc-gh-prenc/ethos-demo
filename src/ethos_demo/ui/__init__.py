"""UI layer — styles and reusable rendering components."""

from .components import (
    render_demographics,
    render_ehr_stats,
    render_estimate_button_fill,
    render_explainer_status,
    render_outcome_card,
    render_score_overview,
    render_status_indicator,
    render_summary,
)
from .styles import inject_styles

__all__ = [
    "inject_styles",
    "render_demographics",
    "render_ehr_stats",
    "render_estimate_button_fill",
    "render_explainer_status",
    "render_outcome_card",
    "render_score_overview",
    "render_status_indicator",
    "render_summary",
]
