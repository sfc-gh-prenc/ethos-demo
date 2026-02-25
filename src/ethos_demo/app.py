"""ETHOS Demo - Streamlit application."""

import logging
from datetime import timedelta

import streamlit as st

from ethos_demo.backend import BackendEvent, BackendMonitor
from ethos_demo.config import (
    DEFAULT_BASE_URL,
    DEFAULT_ETHOS_TEMPERATURE,
    HEALTH_RETRY_SECONDS,
    N_EXPLANATION_TRAJECTORIES,
    N_SAMPLES,
    SAMPLE_SEED,
    TOKENIZED_DATASETS_DIR,
)
from ethos_demo.data import (
    build_sample_labels,
    format_timedelta,
    get_allowed_token_ids,
    get_patient_bmi_group,
    get_patient_demographics,
    load_dataset,
    sample_indices,
)
from ethos_demo.estimator import OutcomeEstimator
from ethos_demo.explainer import TrajectoryExplainer
from ethos_demo.scenarios import SCENARIOS, OutcomeRule, Scenario
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
    if any(
        o is not None and o.running
        for o in [
            st.session_state.get("_summarizer"),
            st.session_state.get("_estimator"),
        ]
    ):
        return True
    return any(
        v is not None and v.running
        for k, v in st.session_state.items()
        if k.startswith("_explainer_")
    )


st.set_page_config(page_title="ETHOS Demo", layout="wide")

_GLOBAL_CSS = (
    "[data-testid='stMainBlockContainer']{max-width:60rem!important;"
    "margin:0 auto!important}"
    "@keyframes _spin{to{transform:rotate(360deg)}}"
    "@keyframes _pulse{0%,100%{opacity:1}50%{opacity:.3}}"
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
    ".st-key-sel_ds label p,"
    ".st-key-sel_sc label p,"
    ".st-key-sel_pt label p,"
    ".st-key-sel_outcome label p"
    "{font-size:0.95rem}"
    ".outcome-card{border:1px solid rgba(255,255,255,0.1);"
    "border-radius:12px;padding:1.2em 0.6em;"
    "text-align:center}"
    "div[class*='st-key-card_']{position:relative}"
    ".st-key-refresh_btn{"
    "position:absolute!important;top:10px;right:10px;"
    "z-index:10;width:auto!important}"
    ".st-key-refresh_btn button{"
    "padding:4px 6px!important;min-height:0!important;"
    "background:transparent!important;border:none!important;"
    "box-shadow:none!important;opacity:0.5}"
    ".st-key-refresh_btn button span[data-testid='stIconMaterial']{"
    "font-size:1.4em}"
    ".st-key-refresh_btn button:hover{opacity:1}"
    ".st-key-stop_expl_btn button{"
    "padding:2px 8px!important;min-height:0!important;opacity:0.5}"
    ".st-key-stop_expl_btn button:hover{opacity:1}"
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


def _trigger_explanation(rule: OutcomeRule, estimator, backend) -> None:
    """Ensure an explanation is running or cached for *rule*.

    Called automatically when the outcome dropdown selection changes.
    """
    expl_key = f"_explainer_{rule.name}"
    existing: TrajectoryExplainer | None = st.session_state.get(expl_key)

    st.session_state["_active_explanation"] = rule.name

    # Already has a final overview — just display it
    if existing is not None and existing.text:
        return

    # Currently running — let it finish
    if existing is not None and existing.running:
        return

    # Stopped early with unsummarized trajectories — resume
    if existing is not None and existing.can_resume:
        existing.resume()
        return

    # Has cached summaries but no final text (and not running) — resume to
    # generate the final overview
    if existing is not None and existing.n_summarized > 0:
        existing.resume()
        return

    # Nothing cached — start from scratch
    if not backend.has_llm_model:
        return

    probs = estimator.probabilities
    entry = probs.get(rule.name)
    if entry is None:
        return
    prob, k, n = entry
    margin = wilson_margin(k, n)

    summarizer: SummaryGenerator | None = st.session_state.get("_summarizer")
    past_summary = (
        summarizer.text
        if summarizer and summarizer.text
        else "Past history summary is not available."
    )
    present_summary = (
        summarizer.text
        if summarizer and summarizer.text
        else "Present encounter summary is not available."
    )
    demographics = get_patient_demographics(estimator.dataset, estimator.sample_idx)
    demo_ctx = {
        "marital_status": demographics.get("Marital Status", "unknown"),
        "race": demographics.get("Race", "unknown"),
        "gender": demographics.get("Gender", "unknown"),
        "age": demographics.get("Age", "unknown"),
    }

    explainer = TrajectoryExplainer(
        outcome_rule=rule,
        probability=prob,
        margin=margin,
        trajectories=estimator.trajectories,
        past_summary=past_summary,
        present_summary=present_summary,
        demographics_context=demo_ctx,
        scenario=estimator.scenario,
        dataset_name=st.session_state["sel_ds"],
        model=backend.llm_model,
    )
    st.session_state[expl_key] = explainer
    explainer.start()


# ── Main area ───────────────────────────────────────────────────────────────
dataset_names = (
    sorted(p.name for p in TOKENIZED_DATASETS_DIR.iterdir() if p.is_dir())
    if TOKENIZED_DATASETS_DIR.is_dir()
    else []
)

col_ds, _col_ds_r = st.columns(2)
with col_ds:
    dataset_name = (
        st.selectbox("Tokenized datasets", options=dataset_names, key="sel_ds")
        if dataset_names
        else None
    )
    if not dataset_names:
        st.warning(f"No datasets found in `{TOKENIZED_DATASETS_DIR}`.")

col_sc, col_pt = st.columns(2)
with col_sc:
    scenario = st.selectbox(
        "Clinical Scenarios",
        options=list(Scenario),
        format_func=lambda s: SCENARIOS[s].description,
        index=None,
        placeholder="Choose scenario",
        key="sel_sc",
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
        selected_label = st.selectbox("Cases", options=list(label_to_idx), key="sel_pt")

    if selected_label is not None:
        selected_idx = label_to_idx[selected_label]

        # ── Auto-trigger on selection change ──────────────────────
        current_key = f"{scenario}:{selected_label}"
        selection_changed = st.session_state.get("_last_selection") != current_key
        if selection_changed:
            st.session_state["_last_selection"] = current_key
            # Cancel any in-flight estimation and explainers
            est: OutcomeEstimator | None = st.session_state.pop("_estimator", None)
            if est is not None:
                est.cancel()
            for _ek in [k for k in st.session_state if k.startswith("_explainer_")]:
                old_expl = st.session_state.pop(_ek, None)
                if old_expl is not None:
                    old_expl.cancel()
            st.session_state.pop("_active_explanation", None)
            st.session_state.pop("_prev_outcome", None)
            # Auto-fire summary if LLM model available
            if backend.has_llm_model:
                _start_summary(
                    ds=ds,
                    selected_idx=selected_idx,
                    scenario=scenario,
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

        split = sc.history_fn(ds, selected_idx)

        st.subheader("EHR")

        _section_hdr = "<span style='font-size:1.1em;font-weight:600'>{title}</span>"
        _stat_html = (
            "<span style='color:gray'>{label}</span><br>"
            "<span style='font-size:1.3em'>{value}</span>"
        )
        col_enc, col_hist = st.columns(2)
        with col_enc:
            st.markdown(_section_hdr.format(title="Current Encounter"), unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            e1.markdown(
                _stat_html.format(
                    label="Time Span", value=format_timedelta(split.present_time_span)
                ),
                unsafe_allow_html=True,
            )
            e2.markdown(
                _stat_html.format(label="Events", value=f"{len(split.present_tokens):,}"),
                unsafe_allow_html=True,
            )
        with col_hist:
            st.markdown(_section_hdr.format(title="History"), unsafe_allow_html=True)
            h1, h2 = st.columns(2)
            h1.markdown(
                _stat_html.format(label="Time Span", value=format_timedelta(split.past_time_span)),
                unsafe_allow_html=True,
            )
            h2.markdown(
                _stat_html.format(label="Events", value=f"{len(split.past_tokens):,}"),
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

            # Detect any running explainer
            running_expl: TrajectoryExplainer | None = None
            for _r in sc.outcomes:
                _e = st.session_state.get(f"_explainer_{_r.name}")
                if _e is not None and _e.running:
                    running_expl = _e
                    break

            # ── Button row ────────────────────────────────────
            prog = estimator.progress if estimator else None
            stop_clicked = False
            probs = estimator.probabilities if estimator else {}
            has_trajectories = (
                estimator is not None and not estimating and len(estimator.trajectories) > 0
            )

            dropdown_col, btn_col, expl_status_col = st.columns([1.5, 1.3, 2.2])

            with dropdown_col:
                outcome_labels = [f"{r.icon}  {r.title}" for r in sc.outcomes]
                sel_idx = st.selectbox(
                    "Outcome",
                    options=range(len(sc.outcomes)),
                    format_func=lambda i: outcome_labels[i],
                    key="sel_outcome",
                    label_visibility="collapsed",
                    disabled=running_expl is not None,
                )

            selected_rule = sc.outcomes[sel_idx]

            with btn_col, st.container(key="est_btn"):
                if estimating:
                    pct = int(100 * prog[0] / prog[1]) if prog else 0
                    cancel_clicked = st.button(
                        f"Cancel \u00b7 {pct}%",
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
                        disabled=not has_model or running_expl is not None,
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

            with expl_status_col:
                if running_expl is not None:
                    ep = running_expl.progress
                    if ep is not None:
                        status_text = running_expl.status or ""
                        scol, xcol = st.columns([5, 1])
                        with scol:
                            st.markdown(
                                f"<div style='padding-top:8px'>"
                                f"<span style='color:gray;font-size:0.85em'>"
                                f"\U0001f4a1 {status_text}</span></div>",
                                unsafe_allow_html=True,
                            )
                        with xcol, st.container(key="stop_expl_btn"):
                            stop_clicked = st.button("\u2715", key="stop_expl_action")

            # ── Selected outcome card (aligned with button) ──
            refresh_clicked = False
            _pad_l, card_col, _pad_r = st.columns([1.5, 2, 1.5])
            with card_col, st.container(key=f"card_{selected_rule.name}"):
                card_html = (
                    "<div class='outcome-card'>"
                    f"<span style='font-size:5em'>{selected_rule.icon}</span><br>"
                    f"<b style='font-size:1.3em'>{selected_rule.title}</b>"
                )
                entry = probs.get(selected_rule.name)
                if entry is not None:
                    prob_val, k, n = entry
                    margin_val = wilson_margin(k, n)
                    card_html += (
                        f"<div style='font-size:2em'>"
                        f"{prob_val:.0%}"
                        f"<span style='display:inline-block;width:0;"
                        f"overflow:visible;vertical-align:baseline'>"
                        f"<span style='font-size:0.42em;color:gray;"
                        f"margin-left:4px;white-space:nowrap'>"
                        f"\u00b1{margin_val * 100:.1f}%</span></span></div>"
                    )
                else:
                    card_html += "<div style='font-size:2em;color:gray'>???</div>"
                card_html += "</div>"
                st.markdown(card_html, unsafe_allow_html=True)

                sel_expl: TrajectoryExplainer | None = st.session_state.get(
                    f"_explainer_{selected_rule.name}"
                )
                sel_running = sel_expl is not None and sel_expl.running
                if has_trajectories and not sel_running:
                    with st.container(key="refresh_btn"):
                        refresh_clicked = st.button(
                            "",
                            icon=":material/refresh:",
                            key="refresh_expl_action",
                        )

            # ── Score Overview display ─────────────────────────
            _expl_box = (
                "<div style='min-height:100px;background:rgba(255,255,255,0.04);"
                "border-radius:10px;padding:1em;margin-top:1em'>"
                "<div style='font-size:1.25em;font-weight:600;"
                "margin-bottom:0.3em'>Score Overview</div>"
                "<div style='color:gray;font-size:0.95em;margin-bottom:0.6em'>"
                "{icon} &ensp; {title}</div>"
                "<div style='font-size:1.05em'>{msg}</div>"
                "{footnote}"
                "</div>"
            )

            active_expl = st.session_state.get("_active_explanation")
            if active_expl:
                active_expl_obj: TrajectoryExplainer | None = st.session_state.get(
                    f"_explainer_{active_expl}"
                )
                if active_expl_obj is not None:
                    in_final_phase = active_expl_obj.text is not None or (
                        active_expl_obj.progress is None
                        and active_expl_obj.running
                        and active_expl_obj.n_summarized > 0
                    )
                    if in_final_phase:
                        active_rule = next(
                            (r for r in sc.outcomes if r.name == active_expl),
                            None,
                        )
                        if active_rule:
                            msg = active_expl_obj.text or active_expl_obj.status or ""
                            n_used = active_expl_obj.n_summarized
                            footnote = ""
                            if n_used > 0 and active_expl_obj.text:
                                footnote = (
                                    "<div style='color:gray;font-size:0.75em;"
                                    f"margin-top:0.5em'>Based on {n_used} "
                                    "ETHOS trajectories</div>"
                                )
                            st.markdown(
                                _expl_box.format(
                                    icon=active_rule.icon,
                                    title=active_rule.title,
                                    msg=msg,
                                    footnote=footnote,
                                ),
                                unsafe_allow_html=True,
                            )

            if estimating and prog:
                _logger.debug(
                    "estimation %d/%d \u2014 %s",
                    prog[0],
                    prog[1],
                    estimator.log_summary,
                )

            # ── Start estimation ──────────────────────────────
            if not estimating and run_clicked:
                for _ek in [k for k in st.session_state if k.startswith("_explainer_")]:
                    old_expl = st.session_state.pop(_ek, None)
                    if old_expl is not None:
                        old_expl.cancel()
                st.session_state.pop("_active_explanation", None)
                st.session_state.pop("_prev_outcome", None)

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
                new_est.start()
                st.rerun(scope="fragment")

            # ── Cancel estimation ─────────────────────────────
            if estimating and cancel_clicked:
                estimator.cancel()
                st.rerun(scope="fragment")

            # ── Auto-trigger explanation on dropdown switch ───
            if has_trajectories and not estimating:
                _prev_outcome = st.session_state.get("_prev_outcome")
                if _prev_outcome != selected_rule.name:
                    st.session_state["_prev_outcome"] = selected_rule.name
                    _trigger_explanation(selected_rule, estimator, backend)
                    st.rerun(scope="fragment")

            # ── Handle refresh button ─────────────────────────
            if refresh_clicked:
                sel_expl_obj: TrajectoryExplainer | None = st.session_state.get(
                    f"_explainer_{selected_rule.name}"
                )
                if sel_expl_obj is not None:
                    if sel_expl_obj.n_summarized >= N_EXPLANATION_TRAJECTORIES:
                        sel_expl_obj.restart()
                    else:
                        sel_expl_obj.resume()
                    st.session_state["_active_explanation"] = selected_rule.name
                else:
                    _trigger_explanation(selected_rule, estimator, backend)
                st.rerun(scope="fragment")

            # ── Handle stop (early finalize) ──────────────────
            if stop_clicked and running_expl is not None:
                running_expl.stop()
                st.rerun(scope="fragment")

        with st.container(key="outcomes_slot"):
            _outcomes_fragment()


def main():
    """Entrypoint for the `ethos-app` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless=true"]
    st_main()
