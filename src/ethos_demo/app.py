"""ETHOS Demo - Streamlit application."""

import logging
import threading
from datetime import timedelta

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from ethos_demo.backend import BackendEvent, BackendMonitor
from ethos_demo.config import (
    DEFAULT_BASE_URL,
    DEFAULT_ETHOS_TEMPERATURE,
    HEALTH_POLL_SECONDS,
    N_SAMPLES,
    SAMPLE_SEED,
    TOKENIZED_DATASETS_DIR,
)
from ethos_demo.data import (
    build_sample_labels,
    get_patient_demographics,
    get_sample_context_stats,
    load_dataset,
    sample_indices,
)
from ethos_demo.estimator import OutcomeEstimator
from ethos_demo.scenarios import SCENARIOS, Scenario
from ethos_demo.summarizer import SummaryGenerator

_logger = logging.getLogger("ethos_demo")
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    _logger.addHandler(_handler)
_logger.propagate = False

st.set_page_config(page_title="ETHOS Demo", layout="wide")

_GLOBAL_CSS = (
    "@keyframes _spin{to{transform:rotate(360deg)}}"
    ".status-row{display:flex;align-items:center;gap:8px;height:28px}"
    ".status-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}"
    ".status-dot.ok{background:#4caf50}"
    ".status-dot.err{background:#f44336}"
    ".status-text{font-size:0.85em}"
    ".st-key-sidebar_bottom{position:fixed;bottom:1rem;width:inherit}"
    ".st-key-sidebar_bottom button"
    "{padding:0!important;min-height:0!important}"
    ".st-key-ai_working_bar span[data-testid='stIconMaterial']"
    "{animation:_spin .7s linear infinite}"
    ".st-key-health_loading span[data-testid='stIconMaterial']"
    "{animation:_spin .7s linear infinite}"
)
st.markdown(f"<style>{_GLOBAL_CSS}</style>", unsafe_allow_html=True)

st.title("ETHOS Demo")

# ── Helpers ────────────────────────────────────────────────────────────────


def _cancel_ai_work() -> None:
    """Signal any in-flight background AI work to stop."""
    ev: threading.Event | None = st.session_state.get("_cancel_event")
    if ev is not None:
        ev.set()
    st.session_state.pop("_cancel_event", None)
    st.session_state.pop("_ai_working_summary", None)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    backend = BackendMonitor(DEFAULT_BASE_URL)

    def _status_html(dot_cls: str, label: str) -> str:
        return (
            f"<div class='status-row'>"
            f"<span style='color:gray;font-size:0.8em'>ETHOS</span>"
            f"<span class='status-dot {dot_cls}'></span>"
            f"<span class='status-text'>{label}</span>"
            f"</div>"
        )

    # ── AI working indicator (polls every 1s) ─────────────────
    @st.fragment(run_every=timedelta(seconds=1))
    def _ai_working_fragment():
        if not st.session_state.get("_ai_working"):
            return
        with st.container(key="ai_working_bar"):
            st.button(
                "AI is working…",
                icon=":material/progress_activity:",
                type="tertiary",
                disabled=True,
                key="_ai_working_label",
            )

    # ── ETHOS health status (polls every HEALTH_POLL_SECONDS) ─
    @st.fragment(run_every=timedelta(seconds=HEALTH_POLL_SECONDS))
    def _health_status_fragment():
        event = backend.poll()

        if event is BackendEvent.CHECKING:
            status_col, btn_col = st.columns([3, 1])
            with status_col:
                st.markdown(_status_html("err", "Checking…"), unsafe_allow_html=True)
            with btn_col, st.container(key="health_loading"):
                st.button(
                    "",
                    icon=":material/progress_activity:",
                    key="refresh_health",
                    type="tertiary",
                )
            return

        dot_cls = "ok" if backend.healthy else "err"
        label = "Connected" if backend.healthy else "Unreachable"

        if not backend.healthy:
            status_col, btn_col = st.columns([3, 1])
            with status_col:
                st.markdown(_status_html(dot_cls, label), unsafe_allow_html=True)
            with btn_col:
                st.button(
                    "",
                    icon=":material/refresh:",
                    key="refresh_health",
                    type="tertiary",
                    on_click=backend.request_check,
                )
        else:
            st.markdown(_status_html(dot_cls, label), unsafe_allow_html=True)

    with st.container(key="sidebar_bottom"):
        _ai_working_fragment()
        _health_status_fragment()

    # ── Configuration widgets (outside fragment so selection triggers
    #    a full rerun, updating button states in the main area) ─────
    st.header("Configuration")

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

        models = backend.models
        if models:
            prev = st.session_state.get(saved_key, default)
            idx = models.index(prev) if prev and prev in models else None
            if idx is not None:
                st.session_state.pop(key, None)
            st.selectbox(
                label,
                options=models,
                index=idx,
                placeholder=placeholder,
                key=key,
            )
        else:
            st.session_state.pop(key, None)
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


def _start_summary(
    *,
    ds,
    selected_idx: int,
    scenario: Scenario,
    sc,
) -> None:
    """Launch EHR summary generation in a background thread."""
    _cancel_ai_work()

    st.session_state.pop("_ehr_summary", None)
    st.session_state.pop("_summary_status", None)

    cancel_event = threading.Event()
    st.session_state["_cancel_event"] = cancel_event
    gen = st.session_state.get("_ai_gen", 0) + 1
    st.session_state["_ai_gen"] = gen
    st.session_state["_ai_working"] = True
    st.session_state["_ai_working_summary"] = True

    summarizer = SummaryGenerator(
        dataset=ds,
        selected_idx=selected_idx,
        scenario=scenario,
        scenario_context=sc.context,
        model_id=backend.llm_model,
        on_status=lambda msg: st.session_state.__setitem__("_summary_status", msg),
        on_chunk=lambda text: st.session_state.__setitem__("_ehr_summary", text),
        cancel_event=cancel_event,
    )

    my_gen = gen
    ctx = get_script_run_ctx()

    def _bg() -> None:
        summarizer.run()
        st.session_state["_ai_working_summary"] = False
        if st.session_state.get("_ai_gen") == my_gen:
            if not st.session_state.get("_estimating"):
                st.session_state["_ai_working"] = False
            st.session_state.pop("_summary_status", None)

    thread = threading.Thread(target=_bg, daemon=True)
    add_script_run_ctx(thread, ctx)
    thread.start()


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
        options=list(Scenario),
        format_func=lambda s: SCENARIOS[s].description,
        index=None,
        placeholder="Choose scenario",
    )

selected_label = None
if dataset_name and scenario:
    sc = SCENARIOS[scenario]
    tasks = sc.task_names
    ds_task = sc.dataset

    with st.spinner("Loading dataset…"):
        ds = load_dataset(dataset_name, ds_task)

    indices = sample_indices(ds, N_SAMPLES, SAMPLE_SEED)
    labels = build_sample_labels(ds, indices)

    label_to_idx = {label: idx for idx, label in labels}
    with col_pt:
        selected_label = st.selectbox("Patient", options=list(label_to_idx))

    if selected_label is not None:
        selected_idx = label_to_idx[selected_label]

        # ── Auto-trigger on selection change ──────────────────────
        current_key = f"{scenario}:{selected_label}"
        selection_changed = st.session_state.get("_last_selection") != current_key
        if selection_changed:
            st.session_state["_last_selection"] = current_key
            # Cancel any in-flight estimation
            est_ev: threading.Event | None = st.session_state.get("_est_cancel_event")
            if est_ev is not None:
                est_ev.set()
            st.session_state["_estimating"] = False
            st.session_state.pop("_est_progress", None)
            for t in tasks:
                st.session_state.pop(f"prob_{t}", None)
            # Auto-fire summary if LLM model available
            if backend.has_llm_model:
                _start_summary(
                    ds=ds,
                    selected_idx=selected_idx,
                    scenario=scenario,
                    sc=sc,
                )

        # Deferred auto-fire: model selected after patient was already chosen
        if (
            not selection_changed
            and backend.has_llm_model
            and not st.session_state.get("_ehr_summary")
            and not st.session_state.get("_ai_working")
        ):
            _start_summary(
                ds=ds,
                selected_idx=selected_idx,
                scenario=scenario,
                sc=sc,
            )

        # ── Demographics ──────────────────────────────────────────
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

        st.subheader("EHR History")
        ehr_cols = st.columns(len(demographics))
        for col, (key, value) in zip(ehr_cols, context_stats.items(), strict=False):
            col.markdown(
                f"<span style='color:gray'>{key}</span><br>"
                f"<span style='font-size:1.3em'>{value}</span>",
                unsafe_allow_html=True,
            )

        # ── EHR Summary (polls every 0.5s for streaming text) ─────
        _summary_box = (
            "<div style='min-height:120px'>"
            "<span style='color:gray'>EHR Summary</span><br>"
            "<span style='font-size:1.1em'>{msg}</span>"
            "</div>"
        )
        _summary_unavailable = (
            "<div style='min-height:120px'>"
            "<span style='color:gray'>EHR Summary</span><br>"
            "<span style='font-size:1.1em;color:gray'>Summary not available — "
            "select an LLM Provider</span></div>"
        )

        @st.fragment(run_every=timedelta(milliseconds=500))
        def _summary_fragment():
            summary_status = st.session_state.get("_summary_status")
            summary_text = st.session_state.get("_ehr_summary")
            if summary_text or summary_status:
                msg = summary_text or summary_status
                st.markdown(_summary_box.format(msg=msg), unsafe_allow_html=True)
            elif not backend.has_llm_model:
                st.markdown(_summary_unavailable, unsafe_allow_html=True)
            else:
                st.markdown(_summary_box.format(msg=""), unsafe_allow_html=True)

        _summary_fragment()

        # ── Outcome estimation ────────────────────────────────────
        st.divider()

        @st.fragment(run_every=timedelta(seconds=1))
        def _outcomes_fragment():
            estimating = st.session_state.get("_estimating", False)
            has_model = backend.has_ethos_model

            # ── Button row ────────────────────────────────────
            btn_col, _spacer = st.columns([1, 3])
            with btn_col:
                if estimating:
                    cancel_clicked = st.button("Cancel", use_container_width=True)
                    run_clicked = False
                else:
                    cancel_clicked = False
                    run_clicked = st.button(
                        "Estimate Outcomes",
                        disabled=not has_model,
                        use_container_width=True,
                    )

            # ── Outcome cards ─────────────────────────────────
            card_cols = st.columns(len(sc.outcomes))
            for col, rule in zip(card_cols, sc.outcomes, strict=False):
                with col:
                    st.markdown(
                        f"<div style='text-align:center'>"
                        f"<span style='font-size:5em'>{rule.icon}</span><br>"
                        f"<b style='font-size:1.3em'>{rule.title}</b></div>",
                        unsafe_allow_html=True,
                    )
                    prob = st.session_state.get(f"prob_{rule.name}")
                    if prob is not None:
                        st.markdown(
                            f"<div style='text-align:center;font-size:2em'>{prob:.0%}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div style='text-align:center;font-size:2em;color:gray'>???</div>",
                            unsafe_allow_html=True,
                        )

            # ── Progress bar ──────────────────────────────────
            if estimating:
                prog = st.session_state.get("_est_progress")
                if prog is not None:
                    completed, total = prog
                    st.progress(completed / total, text=f"{completed}/{total}")

            # ── Start estimation ──────────────────────────────
            if not estimating and run_clicked:
                for t_name in tasks:
                    st.session_state.pop(f"prob_{t_name}", None)
                st.session_state.pop("_est_progress", None)
                st.session_state["_estimating"] = True
                st.session_state["_ai_working"] = True

                est_cancel = threading.Event()
                st.session_state["_est_cancel_event"] = est_cancel

                estimator = OutcomeEstimator(
                    dataset=ds,
                    sample_idx=selected_idx,
                    scenario=scenario,
                    model_id=backend.ethos_model,
                    temperature=st.session_state.get(
                        "ethos_temperature", DEFAULT_ETHOS_TEMPERATURE
                    ),
                    on_progress=lambda c, tot: st.session_state.__setitem__(
                        "_est_progress", (c, tot)
                    ),
                    on_outcome_update=lambda name, p: st.session_state.__setitem__(
                        f"prob_{name}", p
                    ),
                    cancel_event=est_cancel,
                )

                def _est_bg() -> None:
                    try:
                        estimator.run()
                    except Exception:
                        _logger.exception("Estimation failed")
                    finally:
                        st.session_state["_estimating"] = False
                        st.session_state.pop("_est_progress", None)
                        if not st.session_state.get("_ai_working_summary"):
                            st.session_state["_ai_working"] = False

                thread = threading.Thread(target=_est_bg, daemon=True)
                add_script_run_ctx(thread, get_script_run_ctx())
                thread.start()
                st.rerun(scope="fragment")

            # ── Cancel estimation ─────────────────────────────
            if estimating and cancel_clicked:
                ev = st.session_state.get("_est_cancel_event")
                if ev:
                    ev.set()
                st.session_state["_estimating"] = False
                st.session_state.pop("_est_progress", None)
                if not st.session_state.get("_ai_working_summary"):
                    st.session_state["_ai_working"] = False
                st.rerun(scope="fragment")

        _outcomes_fragment()


def main():
    """Entrypoint for the `ethos-app` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless=true"]
    st_main()
