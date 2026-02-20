"""Shared constants for the ETHOS Demo application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_DATASETS_DIR = PROJECT_ROOT / "data" / "tokenized_datasets"

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_ETHOS_MODEL = "ethos-gpt"
DEFAULT_LLM_MODEL = "deepseek"
API_KEY = "fake-key"
HEALTH_POLL_SECONDS = 30
HEALTH_TIMEOUT_SECONDS = 3

SCENARIO_TASKS: dict[str, list[str]] = {
    "Triage": ["ed_hospitalization", "ed_critical_outcome"],
    "Hospital Admission": ["icu_admission", "icu_mortality"],
    "Hospital Discharge": ["readmission_30d", "readmission_90d"],
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

SCENARIO_CONTEXT: dict[str, str] = {
    "Triage": (
        "This patient has just arrived at the Emergency Department and is being triaged. "
        "The summary should help the triage clinician rapidly assess acuity and identify "
        "immediate risks."
    ),
    "Hospital Admission": (
        "This patient is being admitted to the hospital from the Emergency Department. "
        "The summary should help the admitting team understand the clinical picture and "
        "prioritize initial workup."
    ),
    "Hospital Discharge": (
        "This patient is being discharged from the hospital. The summary should help "
        "the discharging clinician quickly understand the patient's recent clinical "
        "course and current status at the time of discharge."
    ),
}

MODEL_CONTEXT_SIZE = 4096

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

N_REQUESTS = 10
N_PER_REQUEST = 10
