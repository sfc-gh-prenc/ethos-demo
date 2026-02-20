"""Shared constants for the ETHOS Demo application."""

from enum import StrEnum
from pathlib import Path


class Scenario(StrEnum):
    TRIAGE = "Triage"
    HOSPITAL_ADMISSION = "Hospital Admission"
    HOSPITAL_DISCHARGE = "Hospital Discharge"


SCENARIO_DESCRIPTION: dict[Scenario, str] = {
    Scenario.TRIAGE: "Triage — patient came to ED and was triaged",
    Scenario.HOSPITAL_ADMISSION: "Admission — patient was admitted to hospital",
    Scenario.HOSPITAL_DISCHARGE: "Discharge — patient was discharged from hospital",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_DATASETS_DIR = PROJECT_ROOT / "data" / "tokenized_datasets"

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_ETHOS_MODEL = "ethos-gpt"
DEFAULT_ETHOS_TEMPERATURE = 1.0
DEFAULT_LLM_MODEL = "deepseek"
API_KEY = "fake-key"
HEALTH_POLL_SECONDS = 15
HEALTH_TIMEOUT_SECONDS = 3

SCENARIO_TASKS: dict[Scenario, list[str]] = {
    Scenario.TRIAGE: ["ed_hospitalization", "ed_critical_outcome"],
    Scenario.HOSPITAL_ADMISSION: ["icu_admission", "icu_mortality"],
    Scenario.HOSPITAL_DISCHARGE: ["readmission_30d", "readmission_90d"],
}

# Maps app-level task names to the underlying ethos dataset task names.
# Tasks not listed here use their own name as the dataset task.
TASK_DATASET_MAP: dict[str, str] = {
    "readmission_30d": "readmission",
    "readmission_90d": "readmission",
}

SAMPLE_SEED = 42
N_SAMPLES = 10

TASK_DISPLAY: dict[str, dict[str, str]] = {
    "ed_hospitalization": {"icon": "\U0001f3e5", "title": "Hospitalization"},
    "ed_critical_outcome": {"icon": "\U0001f6cf\ufe0f", "title": "Critical Event"},
    "icu_admission": {"icon": "\U0001f6cf\ufe0f", "title": "ICU Admission"},
    "icu_mortality": {"icon": "\U0001f480", "title": "Mortality"},
    "readmission_30d": {"icon": "\U0001f3e5", "title": "30-Day Readmission"},
    "readmission_90d": {"icon": "\U0001f3e5", "title": "90-Day Readmission"},
}

SCENARIO_CONTEXT: dict[Scenario, str] = {
    Scenario.TRIAGE: (
        "The patient is presenting to the Emergency Department. They are being registered "
        "and triaged. Vitals are being measured, acuity level is being assessed, and the "
        "clinical team is determining the urgency of care needed. "
        "The timeline events below are from the triage encounter."
    ),
    Scenario.HOSPITAL_ADMISSION: (
        "The patient is being admitted to the hospital. The admitting team is reviewing "
        "the clinical picture, evaluating the need for ICU-level care, and prioritizing "
        "the initial workup. "
        "The timeline events below cover the last 24 hours before admission."
    ),
    Scenario.HOSPITAL_DISCHARGE: (
        "The patient is being discharged from the hospital. The care team is reviewing "
        "the clinical course, assessing readiness for discharge, and evaluating the risk "
        "of short-term and medium-term readmission. "
        "The timeline events below cover the entire hospital stay from admission to discharge."
    ),
}

MODEL_CONTEXT_SIZE = 4096

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

N_REQUESTS = 100
N_PER_REQUEST = 10
