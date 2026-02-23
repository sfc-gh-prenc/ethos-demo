"""Scenario definitions for the ETHOS Demo application."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

from ethos.constants import SpecialToken
from ethos.supported_tasks import Task

from .history import HistorySplit, get_last_36h_history, get_stay_history, get_triage_history

if TYPE_CHECKING:
    from ethos.datasets import InferenceDataset

HistoryFn = Callable[["InferenceDataset", int], HistorySplit]


class Scenario(StrEnum):
    TRIAGE = "triage"
    HOSPITAL_ADMISSION = "hospital_admission"
    HOSPITAL_DISCHARGE = "hospital_discharge"


@dataclass(frozen=True)
class OutcomeRule:
    name: str
    icon: str
    title: str
    positive_events: frozenset[str]
    time_window: timedelta | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    description: str
    dataset: Task
    stop_tokens: tuple[str, ...]
    outcomes: tuple[OutcomeRule, ...]
    context: str
    history_fn: HistoryFn

    @property
    def task_names(self) -> list[str]:
        return [o.name for o in self.outcomes]


_COMMON_STOP = (SpecialToken.DEATH, SpecialToken.TIMELINE_END)

SCENARIOS: dict[Scenario, ScenarioConfig] = {
    Scenario.TRIAGE: ScenarioConfig(
        description="Patient presents to ED and is triaged",
        dataset=Task.ED_HOSPITALIZATION,
        stop_tokens=(SpecialToken.ICU_ADMISSION, *_COMMON_STOP),
        outcomes=(
            OutcomeRule(
                "ed_hospitalization",
                "\U0001f3e5",
                "Hospitalization",
                frozenset({SpecialToken.ADMISSION}),
                timedelta(hours=36),
            ),
            OutcomeRule(
                "ed_critical_outcome",
                "\U0001f6cf\ufe0f",
                "12h Critical Event",
                frozenset({SpecialToken.ICU_ADMISSION, SpecialToken.DEATH}),
                timedelta(hours=12),
            ),
        ),
        context=(
            "The patient is presenting to the Emergency Department. They are being registered "
            "and triaged. Vitals are being measured, acuity level is being assessed, and the "
            "clinical team is determining the urgency of care needed. "
            "The present timeline covers events from the triage encounter."
        ),
        history_fn=get_triage_history,
    ),
    Scenario.HOSPITAL_ADMISSION: ScenarioConfig(
        description="Patient is admitted to hospital",
        dataset=Task.ICU_ADMISSION,
        stop_tokens=(SpecialToken.DISCHARGE, *_COMMON_STOP),
        outcomes=(
            OutcomeRule(
                "icu_admission",
                "\U0001f6cf\ufe0f",
                "ICU Admission",
                frozenset({SpecialToken.ICU_ADMISSION, SpecialToken.DEATH}),
            ),
            OutcomeRule(
                "icu_mortality",
                "\U0001f480",
                "Mortality",
                frozenset({SpecialToken.DEATH}),
            ),
        ),
        context=(
            "The patient is being admitted to the hospital. The admitting team is reviewing "
            "the clinical picture. "
            "The present timeline covers the last 36 hours before admission."
        ),
        history_fn=get_last_36h_history,
    ),
    Scenario.HOSPITAL_DISCHARGE: ScenarioConfig(
        description="Patient is discharged from hospital",
        dataset=Task.READMISSION,
        stop_tokens=(SpecialToken.ADMISSION, *_COMMON_STOP),
        outcomes=(
            OutcomeRule(
                "readmission_30d",
                "\U0001f3e5",
                "30-Day Readmission",
                frozenset({SpecialToken.ADMISSION, SpecialToken.DEATH}),
                timedelta(days=30),
            ),
            OutcomeRule(
                "readmission_90d",
                "\U0001f3e5",
                "90-Day Readmission",
                frozenset({SpecialToken.ADMISSION, SpecialToken.DEATH}),
                timedelta(days=90),
            ),
        ),
        context=(
            "The patient is being discharged from the hospital. The care team is reviewing "
            "the clinical course, assessing readiness for discharge, and evaluating the risk "
            "of short-term and medium-term readmission. "
            "The present timeline covers events from 36 hours before admission through the "
            "entire hospital stay to discharge."
        ),
        history_fn=get_stay_history,
    ),
}
