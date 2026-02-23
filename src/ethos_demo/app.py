"""ETHOS Demo - Streamlit application."""

import logging
from datetime import timedelta

import streamlit as st

from ethos_demo.backend import BackendEvent, BackendMonitor
from ethos_demo.config import (
    DEFAULT_BASE_URL,
    DEFAULT_ETHOS_TEMPERATURE,
    HEALTH_RETRY_SECONDS,
    N_SAMPLES,
    SAMPLE_SEED,
    TOKENIZED_DATASETS_DIR,
)
from ethos_demo.data import (
    build_sample_labels,
    get_allowed_token_ids,
    get_patient_bmi_group,
    get_patient_demographics,
    get_sample_context_stats,
    load_dataset,
    sample_indices,
)
from ethos_demo.estimator import OutcomeEstimator
from ethos_demo.scenarios import SCENARIOS, Scenario
from ethos_demo.summarizer import SummaryGenerator
from ethos_demo.utils import wilson_margin

_logger = logging.getLogger("ethos_demo")
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    _logger.addHandler(_handler)
_logger.propagate = False


def _ai_working() -> bool:
    """True when any background AI task is running."""
    return any(
        o is not None and o.running
        for o in [
            st.session_state.get("_summarizer"),
            st.session_state.get("_estimator"),
        ]
    )


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
)
st.markdown(f"<style>{_GLOBAL_CSS}</style>", unsafe_allow_html=True)

st.title("ETHOS Demo")


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
        slot = st.empty()
        if _ai_working():
            with slot.container(key="ai_working_bar"):
                st.button(
                    "AI is working…",
                    icon=":material/progress_activity:",
                    type="tertiary",
                    disabled=True,
                    key="_ai_working_label",
                )
        else:
            slot.empty()

    # ── ETHOS health status ───────────────────────────────────
    @st.fragment(run_every=timedelta(seconds=HEALTH_RETRY_SECONDS))
    def _health_status_fragment():
        event = backend.poll()
        dot_cls = "ok" if backend.healthy else "err"
        label = "Connected" if backend.healthy else "Unreachable"
        st.markdown(_status_html(dot_cls, label), unsafe_allow_html=True)
        if event != BackendEvent.UNCHANGED:
            st.rerun()

    with st.container(key="sidebar_bottom"):
        _ai_working_fragment()
        _health_status_fragment()

    # ── Configuration widgets (outside fragment so selection triggers
    #    a full rerun, updating button states in the main area) ─────
    st.header("Configuration")

    def _strip_prefix(model_id: str) -> str:
        for prefix in ("ethos/", "llm/"):
            if model_id.startswith(prefix):
                return model_id[len(prefix) :]
        return model_id

    def _model_select(
        label: str,
        key: str,
        models: list[str],
        default: str | None = None,
    ) -> None:
        saved_key = f"_saved_{key}"
        current = st.session_state.get(key)
        if current and current != "Could not fetch models":
            st.session_state[saved_key] = current

        if models:
            prev = st.session_state.get(saved_key, default)
            idx = models.index(prev) if prev and prev in models else 0
            st.session_state.pop(key, None)
            st.selectbox(
                label,
                options=models,
                index=idx,
                format_func=_strip_prefix,
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

    _model_select("ETHOS Provider", "ethos_model_id", backend.ethos_models)
    st.slider(
        "Temp.",
        min_value=0.0,
        max_value=2.0,
        value=DEFAULT_ETHOS_TEMPERATURE,
        step=0.1,
        format="%.1f",
        key="ethos_temperature",
    )
    _model_select("LLM Provider", "llm_model_id", backend.llm_models)


def _start_summary(
    *,
    ds,
    selected_idx: int,
    scenario: Scenario,
    sc,
    dataset_name: str,
) -> None:
    """Launch EHR summary generation in a background thread."""
    old: SummaryGenerator | None = st.session_state.pop("_summarizer", None)
    if old is not None:
        old.cancel()

    summarizer = SummaryGenerator(
        dataset=ds,
        selected_idx=selected_idx,
        scenario=scenario,
        model=backend.llm_model,
        dataset_name=dataset_name,
    )
    st.session_state["_summarizer"] = summarizer
    summarizer.start()


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
            est: OutcomeEstimator | None = st.session_state.pop("_estimator", None)
            if est is not None:
                est.cancel()
            # Auto-fire summary if LLM model available
            if backend.has_llm_model:
                _start_summary(
                    ds=ds,
                    selected_idx=selected_idx,
                    scenario=scenario,
                    sc=sc,
                    dataset_name=dataset_name,
                )

        # Deferred auto-fire: model selected after patient was already chosen
        summarizer = st.session_state.get("_summarizer")
        has_summary = summarizer is not None and summarizer.text is not None
        if (
            not selection_changed
            and backend.has_llm_model
            and not has_summary
            and not _ai_working()
        ):
            _start_summary(
                ds=ds,
                selected_idx=selected_idx,
                scenario=scenario,
                sc=sc,
                dataset_name=dataset_name,
            )

        # ── Demographics ──────────────────────────────────────────
        raw_demo = get_patient_demographics(ds, selected_idx)
        raw_demo["BMI"] = get_patient_bmi_group(ds, selected_idx, dataset_name)
        _DEMO_ORDER = ["Gender", "Race", "Age", "BMI", "Marital Status"]
        demographics = {k: raw_demo.get(k, "???") for k in _DEMO_ORDER}

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
            "<span style='font-size:1.1em;color:gray'>Summary not available.</span></div>"
        )

        @st.fragment(run_every=timedelta(milliseconds=500))
        def _summary_fragment():
            gen: SummaryGenerator | None = st.session_state.get("_summarizer")
            if gen is not None and (gen.text or gen.status):
                msg = gen.text or gen.status
                st.markdown(_summary_box.format(msg=msg), unsafe_allow_html=True)
            elif not backend.has_llm_model:
                st.markdown(_summary_unavailable, unsafe_allow_html=True)
            else:
                st.markdown(_summary_box.format(msg=""), unsafe_allow_html=True)

        with st.container(key="ehr_summary_slot"):
            _summary_fragment()

        # ── Outcome estimation ────────────────────────────────────
        st.divider()

        @st.fragment(run_every=timedelta(seconds=1))
        def _outcomes_fragment():
            estimator: OutcomeEstimator | None = st.session_state.get("_estimator")
            estimating = estimator is not None and estimator.running
            has_model = backend.has_ethos_model

            # ── Button row ────────────────────────────────────
            prog = estimator.progress if estimator else None
            btn_col, _spacer = st.columns([1, 3])
            with btn_col, st.container(key="est_btn"):
                if estimating:
                    pct = int(100 * prog[0] / prog[1]) if prog else 0
                    cancel_clicked = st.button(
                        f"Cancel · {pct}%",
                        key="est_action_btn",
                        use_container_width=True,
                    )
                    run_clicked = False
                else:
                    pct = 0
                    cancel_clicked = False
                    run_clicked = st.button(
                        "Estimate Outcomes",
                        key="est_action_btn",
                        disabled=not has_model,
                        use_container_width=True,
                    )
                _fill = (
                    f"background-image:linear-gradient(to right,"
                    f"rgba(255,255,255,0.10) {pct}%,"
                    f"transparent {pct}%)!important;"
                    if pct
                    else ""
                )
                st.markdown(
                    f"<style>.st-key-est_btn button{{"
                    f"background-color:rgba(255,255,255,0.06)!important;"
                    f"{_fill}}}</style>",
                    unsafe_allow_html=True,
                )

            # ── Outcome cards ─────────────────────────────────
            probs = estimator.probabilities if estimator else {}
            card_cols = st.columns(len(sc.outcomes))
            for col, rule in zip(card_cols, sc.outcomes, strict=False):
                with col:
                    st.markdown(
                        f"<div style='text-align:center'>"
                        f"<span style='font-size:5em'>{rule.icon}</span><br>"
                        f"<b style='font-size:1.3em'>{rule.title}</b></div>",
                        unsafe_allow_html=True,
                    )
                    entry = probs.get(rule.name)
                    if entry is not None:
                        prob, k, n = entry
                        margin = wilson_margin(k, n)
                        st.markdown(
                            f"<div style='text-align:center;font-size:2em'>"
                            f"{prob:.0%}"
                            f"<span style='display:inline-block;width:0;"
                            f"overflow:visible;vertical-align:baseline'>"
                            f"<span style='font-size:0.42em;color:gray;"
                            f"margin-left:4px;white-space:nowrap'>"
                            f"±{margin * 100:.1f}%</span></span></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div style='text-align:center;font-size:2em;color:gray'>???</div>",
                            unsafe_allow_html=True,
                        )

            if estimating and prog:
                _logger.debug(
                    "estimation %d/%d — %s",
                    prog[0],
                    prog[1],
                    estimator.log_summary,
                )

            # ── Start estimation ──────────────────────────────
            if not estimating and run_clicked:
                new_est = OutcomeEstimator(
                    dataset=ds,
                    sample_idx=selected_idx,
                    scenario=scenario,
                    model=backend.ethos_model,
                    max_model_len=backend.ethos_max_model_len,
                    temperature=st.session_state.get(
                        "ethos_temperature", DEFAULT_ETHOS_TEMPERATURE
                    ),
                    allowed_token_ids=get_allowed_token_ids(dataset_name, ds_task),
                )
                st.session_state["_estimator"] = new_est
                st.rerun(scope="fragment")

            # ── Launch estimation (deferred from button click) ──
            if estimator is not None and not estimator.running and estimator.progress is None:
                estimator.start()

            # ── Cancel estimation ─────────────────────────────
            if estimating and cancel_clicked:
                estimator.cancel()
                st.rerun(scope="fragment")

        with st.container(key="outcomes_slot"):
            _outcomes_fragment()


def main():
    """Entrypoint for the `ethos-app` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless=true"]
    st_main()
