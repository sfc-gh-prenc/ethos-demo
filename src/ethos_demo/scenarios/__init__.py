"""Scenarios subpackage — re-exports all public names for backward compatibility."""

from .config import (
    SCENARIOS,
    OutcomeRule,
    Scenario,
    ScenarioConfig,
)
from .history import (
    get_last_24h_history,
    get_stay_history,
    get_timeline_times_us,
    get_triage_history,
)

__all__ = [
    "SCENARIOS",
    "OutcomeRule",
    "Scenario",
    "ScenarioConfig",
    "get_last_24h_history",
    "get_stay_history",
    "get_timeline_times_us",
    "get_triage_history",
]
