"""Shared constants for the ETHOS Demo application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_DATASETS_DIR = PROJECT_ROOT / "data" / "tokenized_datasets"

DEFAULT_BASE_URL = "http://localhost:8000/v1"
API_KEY = "fake-key"

DEFAULT_ETHOS_TEMPERATURE = 1.0

HEALTH_POLL_SECONDS = 15
HEALTH_RETRY_SECONDS = 3
HEALTH_TIMEOUT_SECONDS = 3

SAMPLE_SEED = 42
N_SAMPLES = 10

MODEL_CONTEXT_SIZE = 4096

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

N_STREAMS = 1000
MAX_CONCURRENT_STREAMS = 200
