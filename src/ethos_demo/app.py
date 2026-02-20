"""ETHOS Demo - Streamlit application."""

import logging
from datetime import timedelta

import streamlit as st

from ethos_demo.client import check_health, list_models
from ethos_demo.config import (
    DEFAULT_BASE_URL,
    HEALTH_POLL_SECONDS,
    N_PER_REQUEST,
    N_REQUESTS,
    N_SAMPLES,
    SAMPLE_SEED,
    SCENARIO_TASKS,
    TASK_DISPLAY,
    TOKENIZED_DATASETS_DIR,
)
from ethos_demo.data import (
    build_sample_labels,
    get_patient_demographics,
    get_sample_context_stats,
    get_sample_identity,
    load_dataset,
    sample_common_indices,
)
from ethos_demo.estimator import OutcomeEstimator

_logger = logging.getLogger("ethos_demo")
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
_logger.addHandler(_handler)
_logger.propagate = False

st.set_page_config(page_title="ETHOS Demo", layout="wide")
st.title("ETHOS Demo")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<style>"
        "@keyframes _espin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}"
        ".health-spinner{width:20px;height:20px;border:3px solid #e0e0e0;"
        "border-top:3px solid #666;border-radius:50%;"
        "animation:_espin .8s linear infinite;margin:8px auto 0}"
        "</style>",
        unsafe_allow_html=True,
    )

    @st.fragment(run_every=timedelta(seconds=HEALTH_POLL_SECONDS))
    def _deployment_status():
        hdr_col, btn_col = st.columns([5, 1])
        with hdr_col:
            st.header("ETHOS Status")
        with btn_col:
            btn_ph = st.empty()

        btn_ph.markdown("<div class='health-spinner'></div>", unsafe_allow_html=True)
        healthy = check_health(DEFAULT_BASE_URL)
        btn_ph.button(
            "",
            icon=":material/refresh:",
            key="refresh_health",
            help="Check now",
        )

        if healthy:
            st.success("Deployment healthy", icon=":material/check_circle:")
        else:
            st.error("Deployment unreachable", icon=":material/error:")

        if healthy:
            try:
                available_models = list_models(DEFAULT_BASE_URL)
            except Exception:
                available_models = []
        else:
            available_models = []

        had_models = bool(st.session_state.get("_available_models"))
        st.session_state["_available_models"] = available_models
        if available_models and not had_models:
            st.rerun()

    _deployment_status()

    st.header("Configuration")
    available_models = st.session_state.get("_available_models", [])

    def _model_select(
        label: str,
        key: str,
        default: str | None = None,
        placeholder: str = "Choose model ID",
    ) -> None:
        if available_models:
            idx = (
                available_models.index(default) if default and default in available_models else None
            )
            st.selectbox(
                label,
                options=available_models,
                index=idx,
                placeholder=placeholder,
                key=key,
            )
        else:
            st.selectbox(
                label,
                options=["Could not fetch models"],
                disabled=True,
                key=key,
            )

    _model_select("ETHOS Provider", "ethos_model_id")
    _model_select("LLM Provider", "llm_model_id")

# ── Main area ───────────────────────────────────────────────────────────────
dataset_names = (
    sorted(p.name for p in TOKENIZED_DATASETS_DIR.iterdir() if p.is_dir())
    if TOKENIZED_DATASETS_DIR.is_dir()
    else []
)

col_ds, col_sc = st.columns(2)
with col_ds:
    dataset_name = (
        st.selectbox("Tokenized dataset", options=dataset_names) if dataset_names else None
    )
    if not dataset_names:
        st.warning(f"No datasets found in `{TOKENIZED_DATASETS_DIR}`.")

with col_sc:
    scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIO_TASKS),
        index=None,
        placeholder="Choose scenario",
    )

if dataset_name and scenario:
    tasks = SCENARIO_TASKS[scenario]

    with st.spinner("Loading dataset…"):
        ds = load_dataset(dataset_name, tasks[0])

    indices = sample_common_indices(dataset_name, tasks, N_SAMPLES, SAMPLE_SEED)
    labels = build_sample_labels(ds, indices)

    label_to_idx = {label: idx for idx, label in labels}
    selected_label = st.selectbox("Patient", options=list(label_to_idx))

    if selected_label is not None:
        selected_idx = label_to_idx[selected_label]
        demographics = get_patient_demographics(ds, selected_idx)

        st.subheader("Demographics")
        demo_cols = st.columns(len(demographics))
        for col, (key, value) in zip(demo_cols, demographics.items(), strict=False):
            col.markdown(
                f"<span style='color:gray'>{key}</span><br>"
                f"<span style='font-size:1.3em'>{value}</span>",
                unsafe_allow_html=True,
            )

        context_stats = get_sample_context_stats(ds, selected_idx)
        ctx_cols = st.columns(len(context_stats))
        for col, (key, value) in zip(ctx_cols, context_stats.items(), strict=False):
            col.markdown(
                f"<span style='color:gray'>{key}</span><br>"
                f"<span style='font-size:1.3em'>{value}</span>",
                unsafe_allow_html=True,
            )

        # ── Estimate outcomes ────────────────────────────────────────────
        st.divider()
        estimating = st.session_state.get("_estimating", False)
        ethos_model = st.session_state.get("ethos_model_id")
        has_model = bool(ethos_model and ethos_model != "Could not fetch models")

        btn_col, prog_col = st.columns([1, 3])
        with btn_col:
            run_clicked = st.button(
                "Estimate Outcomes",
                disabled=estimating or not has_model,
            )
        with prog_col:
            progress_ph = st.empty()

        card_cols = st.columns(len(tasks))
        placeholders: dict[str, st.delta_generator.DeltaGenerator] = {}
        for col, t in zip(card_cols, tasks, strict=False):
            info = TASK_DISPLAY[t]
            with col:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<span style='font-size:3em'>{info['icon']}</span><br>"
                    f"<b>{info['title']}</b></div>",
                    unsafe_allow_html=True,
                )
                placeholders[t] = st.empty()
                prob = st.session_state.get(f"prob_{t}")
                if prob is not None:
                    placeholders[t].markdown(
                        f"<div style='text-align:center;font-size:1.5em'>{prob:.0%}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    placeholders[t].markdown(
                        "<div style='text-align:center;font-size:1.5em;color:gray'>??%</div>",
                        unsafe_allow_html=True,
                    )

        if run_clicked:
            st.session_state["_estimating"] = True
            st.rerun()

        if estimating:

            def _on_progress(completed: int, total: int) -> None:
                progress_ph.progress(completed / total, text=f"{completed}/{total}")

            def _on_task_update(task_name: str, prob: float) -> None:
                st.session_state[f"prob_{task_name}"] = prob
                placeholders[task_name].markdown(
                    f"<div style='text-align:center;font-size:1.5em'>{prob:.0%}</div>",
                    unsafe_allow_html=True,
                )

            patient_id, prediction_time = get_sample_identity(ds, selected_idx)
            estimator = OutcomeEstimator(
                dataset_name=dataset_name,
                patient_id=patient_id,
                prediction_time=prediction_time,
                tasks=tasks,
                model_id=ethos_model,
                base_url=DEFAULT_BASE_URL,
                n_requests=N_REQUESTS,
                n_per_request=N_PER_REQUEST,
                on_progress=_on_progress,
                on_task_update=_on_task_update,
            )
            estimator.run()
            progress_ph.empty()
            st.session_state["_estimating"] = False
            st.rerun()


def main():
    """Entrypoint for the `ethos-app` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless=true"]
    st_main()
