"""ETHOS Demo - Streamlit application."""

import logging
import threading
from datetime import timedelta

import streamlit as st

from ethos_demo.client import check_health, list_models
from ethos_demo.config import (
    DEFAULT_BASE_URL,
    DEFAULT_ETHOS_TEMPERATURE,
    HEALTH_POLL_SECONDS,
    N_PER_REQUEST,
    N_REQUESTS,
    N_SAMPLES,
    SAMPLE_SEED,
    SCENARIO_CONTEXT,
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
from ethos_demo.summarizer import SummaryGenerator

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
    _SIDEBAR_CSS = (
        ".status-row{display:flex;align-items:center;gap:8px;height:28px}"
        ".status-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}"
        ".status-dot.ok{background:#4caf50}"
        ".status-dot.err{background:#f44336}"
        ".status-text{font-size:0.85em}"
        ".st-key-health_box [data-testid='stElementContainer']:has(button)"
        "{height:20px;overflow:hidden;margin-bottom:-4px!important}"
        ".st-key-health_box button[disabled]{visibility:hidden}"
        ".st-key-health_box button"
        "{padding:0!important;min-height:0!important}"
        "@keyframes _spin{to{transform:rotate(360deg)}}"
        ".st-key-health_loading span[data-testid='stIconMaterial']"
        "{animation:_spin .7s linear infinite}"
    )
    st.markdown(f"<style>{_SIDEBAR_CSS}</style>", unsafe_allow_html=True)

    def _status_html(dot_cls: str, label: str) -> str:
        return (
            f"<div class='status-row'>"
            f"<span style='color:gray;font-size:0.8em'>ETHOS</span>"
            f"<span class='status-dot {dot_cls}'></span>"
            f"<span class='status-text'>{label}</span>"
            f"</div>"
        )

    def _on_retry():
        st.session_state["_health_loading"] = True

    @st.fragment(run_every=timedelta(seconds=HEALTH_POLL_SECONDS))
    def _deployment_status():
        loading = st.session_state.pop("_health_loading", False)
        was_healthy = st.session_state.get("health_result", None)

        with st.container(key="health_box"):
            if loading:
                with st.container(key="health_loading"):
                    st.button(
                        "",
                        icon=":material/progress_activity:",
                        key="refresh_health",
                        type="tertiary",
                    )
                st.markdown(
                    _status_html("err", "Checking…"),
                    unsafe_allow_html=True,
                )
                return

            st.button(
                "",
                icon=":material/refresh:",
                key="refresh_health",
                type="tertiary",
                disabled=bool(was_healthy),
                on_click=_on_retry,
            )
            healthy = check_health(DEFAULT_BASE_URL)
            st.session_state["health_result"] = healthy

            dot_cls = "ok" if healthy else "err"
            label = "Connected" if healthy else "Unreachable"
            st.markdown(
                _status_html(dot_cls, label),
                unsafe_allow_html=True,
            )

        if healthy:
            try:
                available_models = list_models(DEFAULT_BASE_URL)
            except Exception:
                available_models = []
        else:
            available_models = []

        had_models = bool(st.session_state.get("_available_models"))
        st.session_state["_available_models"] = available_models
        if bool(available_models) != had_models:
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
        saved_key = f"_saved_{key}"
        current = st.session_state.get(key)
        if current and current != "Could not fetch models":
            st.session_state[saved_key] = current

        if available_models:
            prev = st.session_state.get(saved_key, default)
            idx = available_models.index(prev) if prev and prev in available_models else None
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
    st.slider(
        "Temp.",
        min_value=0.0,
        max_value=2.0,
        value=DEFAULT_ETHOS_TEMPERATURE,
        step=0.1,
        format="%.1f",
        key="ethos_temperature",
    )
    _model_select("LLM Provider", "llm_model_id")

# ── Main area ───────────────────────────────────────────────────────────────
dataset_names = (
    sorted(p.name for p in TOKENIZED_DATASETS_DIR.iterdir() if p.is_dir())
    if TOKENIZED_DATASETS_DIR.is_dir()
    else []
)

col_ds, _col_ds_r = st.columns(2)
with col_ds:
    dataset_name = (
        st.selectbox("Tokenized dataset", options=dataset_names) if dataset_names else None
    )
    if not dataset_names:
        st.warning(f"No datasets found in `{TOKENIZED_DATASETS_DIR}`.")

col_sc, col_pt = st.columns(2)
with col_sc:
    scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIO_TASKS),
        index=None,
        placeholder="Choose scenario",
    )

selected_label = None
if dataset_name and scenario:
    tasks = SCENARIO_TASKS[scenario]

    with st.spinner("Loading dataset…"):
        ds = load_dataset(dataset_name, tasks[0])

    indices = sample_common_indices(dataset_name, tasks, N_SAMPLES, SAMPLE_SEED)
    labels = build_sample_labels(ds, indices)

    label_to_idx = {label: idx for idx, label in labels}
    with col_pt:
        selected_label = st.selectbox("Patient", options=list(label_to_idx))

    if selected_label is not None:
        selected_idx = label_to_idx[selected_label]

        current_key = f"{scenario}:{selected_label}"
        if st.session_state.get("_last_selection") != current_key:
            for t in tasks:
                st.session_state.pop(f"prob_{t}", None)
            st.session_state.pop("_ehr_summary", None)
            st.session_state["_last_selection"] = current_key

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

        connected = bool(st.session_state.get("health_result"))
        llm_model = st.session_state.get("llm_model_id")
        has_llm = connected and bool(llm_model and llm_model != "Could not fetch models")

        ehr_hdr_col, ehr_btn_col = st.columns([10, 1])
        with ehr_hdr_col:
            st.subheader("EHR History")
        with ehr_btn_col:
            summary_clicked = st.button(
                "",
                icon=":material/edit_note:",
                disabled=not has_llm,
                help="Summarize last 24h with LLM" if has_llm else "Select an LLM Provider first",
            )

        ehr_cols = st.columns(len(demographics))
        for col, (key, value) in zip(ehr_cols, context_stats.items(), strict=False):
            col.markdown(
                f"<span style='color:gray'>{key}</span><br>"
                f"<span style='font-size:1.3em'>{value}</span>",
                unsafe_allow_html=True,
            )

        # ── EHR Summary (streaming LLM) ─────────────────────────────────
        summary_ph = st.empty()

        if st.session_state.get("_ehr_summary"):
            summary_ph.markdown(
                f"<span style='color:gray'>EHR Summary</span><br>"
                f"<span style='font-size:1.1em'>{st.session_state['_ehr_summary']}</span>",
                unsafe_allow_html=True,
            )

        if summary_clicked:
            st.session_state.pop("_ehr_summary", None)

            _summary_html = (
                "<span style='color:gray'>EHR Summary</span><br>"
                "<span style='font-size:1.1em'>{msg}</span>"
            )

            with summary_ph.container():
                text_area = st.empty()
                summarizer = SummaryGenerator(
                    dataset=ds,
                    selected_idx=selected_idx,
                    scenario_context=SCENARIO_CONTEXT.get(scenario, ""),
                    model_id=llm_model,
                    base_url=DEFAULT_BASE_URL,
                    on_status=lambda msg: text_area.markdown(
                        _summary_html.format(msg=msg), unsafe_allow_html=True
                    ),
                )
                visible_text = ""
                for visible_text in summarizer.run():
                    text_area.markdown(
                        _summary_html.format(msg=visible_text),
                        unsafe_allow_html=True,
                    )

            st.session_state["_ehr_summary"] = visible_text

        # ── Estimate outcomes ────────────────────────────────────────────
        st.divider()
        estimating = st.session_state.get("_estimating", False)
        ethos_model = st.session_state.get("ethos_model_id")
        has_model = connected and bool(ethos_model and ethos_model != "Could not fetch models")

        btn_col, spinner_col, _btn_spacer = st.columns([1, 0.2, 2.8])
        with btn_col:
            if estimating:
                cancel_clicked = st.button("Cancel", use_container_width=True)
            else:
                cancel_clicked = False
                run_clicked = st.button(
                    "Estimate Outcomes",
                    disabled=not has_model,
                    use_container_width=True,
                )
        with spinner_col:
            if estimating:
                st.markdown(
                    "<div class='health-spinner' style='margin-top:8px'></div>",
                    unsafe_allow_html=True,
                )

        card_cols = st.columns(len(tasks))
        placeholders: dict[str, st.delta_generator.DeltaGenerator] = {}
        progress_phs: dict[str, st.delta_generator.DeltaGenerator] = {}
        for col, t in zip(card_cols, tasks, strict=False):
            info = TASK_DISPLAY[t]
            with col:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<span style='font-size:5em'>{info['icon']}</span><br>"
                    f"<b style='font-size:1.3em'>{info['title']}</b></div>",
                    unsafe_allow_html=True,
                )
                placeholders[t] = st.empty()
                prob = st.session_state.get(f"prob_{t}")
                if prob is not None:
                    placeholders[t].markdown(
                        f"<div style='text-align:center;font-size:2em'>{prob:.0%}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    placeholders[t].markdown(
                        "<div style='text-align:center;font-size:2em;color:gray'>???%</div>",
                        unsafe_allow_html=True,
                    )
                progress_phs[t] = st.empty()

        if not estimating and run_clicked:
            for t in tasks:
                st.session_state.pop(f"prob_{t}", None)
            st.session_state["_estimating"] = True
            st.session_state["_cancel_event"] = threading.Event()
            st.rerun()

        if estimating and cancel_clicked:
            cancel_ev = st.session_state.get("_cancel_event")
            if cancel_ev:
                cancel_ev.set()
            st.session_state["_estimating"] = False
            st.rerun()

        if estimating:

            def _on_progress(task_name: str, completed: int, total: int) -> None:
                progress_phs[task_name].progress(completed / total, text=f"{completed}/{total}")

            def _on_task_update(task_name: str, prob: float) -> None:
                st.session_state[f"prob_{task_name}"] = prob
                placeholders[task_name].markdown(
                    f"<div style='text-align:center;font-size:2em'>{prob:.0%}</div>",
                    unsafe_allow_html=True,
                )

            patient_id, prediction_time = get_sample_identity(ds, selected_idx)
            cancel_ev = st.session_state.get("_cancel_event")
            estimator = OutcomeEstimator(
                dataset_name=dataset_name,
                patient_id=patient_id,
                prediction_time=prediction_time,
                tasks=tasks,
                model_id=ethos_model,
                base_url=DEFAULT_BASE_URL,
                temperature=st.session_state.get("ethos_temperature", DEFAULT_ETHOS_TEMPERATURE),
                n_requests=N_REQUESTS,
                n_per_request=N_PER_REQUEST,
                on_progress=_on_progress,
                on_task_update=_on_task_update,
                cancel_event=cancel_ev,
            )
            estimator.run()
            for ph in progress_phs.values():
                ph.empty()
            st.session_state["_estimating"] = False
            st.rerun()


def main():
    """Entrypoint for the `ethos-app` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless=true"]
    st_main()
