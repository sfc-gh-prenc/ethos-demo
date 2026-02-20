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
    "Hospital Discharge": ["readmission"],
}

SAMPLE_SEED = 42
N_SAMPLES = 10

TASK_DISPLAY: dict[str, dict[str, str]] = {
    "ed_hospitalization": {"icon": "\U0001f3e5", "title": "Hospitalization"},
    "ed_critical_outcome": {"icon": "\U0001f6cf\ufe0f", "title": "Critical Event"},
    "icu_admission": {"icon": "\U0001f6cf\ufe0f", "title": "ICU Admission"},
    "icu_mortality": {"icon": "\U0001f480", "title": "Mortality"},
    "readmission": {"icon": "\U0001f3e5", "title": "Readmission"},
}

MODEL_CONTEXT_SIZE = 4096

N_REQUESTS = 10
N_PER_REQUEST = 10
