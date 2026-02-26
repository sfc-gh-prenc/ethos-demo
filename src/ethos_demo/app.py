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
    format_timedelta,
    get_allowed_token_ids,
    get_sample_context,
)
from ethos_demo.estimator import OutcomeEstimator
from ethos_demo.explainer import TrajectoryExplainer
from ethos_demo.scenarios import SCENARIOS, OutcomeRule, Scenario
from ethos_demo.summarizer import SummaryGenerator
from ethos_demo.ui import (
    inject_styles,
    render_demographics,
    render_ehr_stats,
    render_estimate_button_fill,
    render_explainer_status,
    render_outcome_card,
    render_score_overview,
    render_status_indicator,
    render_summary,
)
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
inject_styles()

st.title("ETHOS Demo")


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    backend = BackendMonitor(DEFAULT_BASE_URL)

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
        render_status_indicator(backend.healthy)
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


def _trigger_explanation(rule: OutcomeRule, estimator, backend, ctx) -> None:
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
    demographics = ctx.demographics(estimator.sample_idx)
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


@st.cache_data(ttl=60)
def _list_dataset_names() -> list[str]:
    if not TOKENIZED_DATASETS_DIR.is_dir():
        return []
    return sorted(p.name for p in TOKENIZED_DATASETS_DIR.iterdir() if p.is_dir())


dataset_names = _list_dataset_names()

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

    with st.spinner("Loading dataset…"):
        ctx = get_sample_context(dataset_name, sc.dataset, N_SAMPLES, SAMPLE_SEED)

    with col_pt:
        selected_label = st.selectbox("Cases", options=list(ctx.label_to_idx), key="sel_pt")

    if selected_label is not None:
        selected_idx = ctx.label_to_idx[selected_label]

        # ── Auto-trigger on selection change ──────────────────────
        current_key = f"{scenario}:{selected_label}"
        selection_changed = st.session_state.get("_last_selection") != current_key
        if selection_changed:
            st.session_state["_last_selection"] = current_key
            est: OutcomeEstimator | None = st.session_state.pop("_estimator", None)
            if est is not None:
                est.cancel()
            for _ek in [k for k in st.session_state if k.startswith("_explainer_")]:
                old_expl = st.session_state.pop(_ek, None)
                if old_expl is not None:
                    old_expl.cancel()
            st.session_state.pop("_active_explanation", None)
            st.session_state.pop("_prev_outcome", None)
            if backend.has_llm_model:
                _start_summary(
                    ds=ctx.dataset,
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
                ds=ctx.dataset,
                selected_idx=selected_idx,
                scenario=scenario,
                dataset_name=dataset_name,
            )

        # ── Demographics ──────────────────────────────────────────
        render_demographics(ctx.demographics(selected_idx))

        split = ctx.history_split(selected_idx, scenario)
        render_ehr_stats(
            present_time_span=format_timedelta(split.present_time_span),
            present_count=len(split.present_tokens),
            past_time_span=format_timedelta(split.past_time_span),
            past_count=len(split.past_tokens),
        )

        # ── EHR Summary (polls every 0.5s for streaming text) ─────
        @st.fragment(run_every=timedelta(milliseconds=500))
        def _summary_fragment():
            gen: SummaryGenerator | None = st.session_state.get("_summarizer")
            msg = (gen.text or gen.status) if gen else None
            render_summary(msg, available=backend.has_llm_model)

        with st.container(key="ehr_summary_slot"):
            _summary_fragment()

        # ── Outcome estimation ────────────────────────────────────
        st.divider()

        @st.fragment(run_every=timedelta(seconds=1))
        def _outcomes_fragment():
            estimator: OutcomeEstimator | None = st.session_state.get("_estimator")
            estimating = estimator is not None and estimator.running
            has_model = backend.has_ethos_model
            has_trajectories = (
                estimator is not None and not estimating and len(estimator.trajectories) > 0
            )

            # ── Pre-render: handle dropdown change before any widgets ──
            _outcome_map = {r.name: r for r in sc.outcomes}
            _sel_name = st.session_state.get("sel_outcome", sc.outcomes[0].name)
            if _sel_name not in _outcome_map:
                _sel_name = sc.outcomes[0].name
                st.session_state["sel_outcome"] = _sel_name
            _selected = _outcome_map[_sel_name]

            _prev = st.session_state.get("_prev_outcome")
            if _prev != _selected.name:
                st.session_state["_prev_outcome"] = _selected.name
                if has_trajectories:
                    _trigger_explanation(_selected, estimator, backend, ctx)
                else:
                    st.session_state["_active_explanation"] = _selected.name

            # Auto-trigger explanation when estimation just finished/cancelled
            if has_trajectories and not st.session_state.get(f"_explainer_{_selected.name}"):
                _trigger_explanation(_selected, estimator, backend, ctx)

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

            dropdown_col, btn_col, expl_status_col = st.columns([1.5, 1.3, 2.2])

            with dropdown_col:
                outcome_names = [r.name for r in sc.outcomes]
                sel_name = st.selectbox(
                    "Outcome",
                    options=outcome_names,
                    format_func=lambda n: f"{_outcome_map[n].icon}  {_outcome_map[n].title}",
                    key="sel_outcome",
                    label_visibility="collapsed",
                    disabled=running_expl is not None,
                )

            selected_rule = _outcome_map[sel_name]

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
                render_estimate_button_fill(pct)

            with expl_status_col:
                if running_expl is not None:
                    ep = running_expl.progress
                    if ep is not None:
                        status_text = running_expl.status or ""
                        scol, xcol = st.columns([5, 1])
                        with scol:
                            render_explainer_status(status_text)
                        with xcol, st.container(key="stop_expl_btn"):
                            stop_clicked = st.button("\u2715", key="stop_expl_action")

            # ── Selected outcome card (aligned with button) ──
            refresh_clicked = False
            _pad_l, card_col, _pad_r = st.columns([1.5, 2, 1.5])
            with card_col, st.container(key="card_selected"):
                entry = probs.get(selected_rule.name)
                if entry is not None:
                    prob_val, k, n = entry
                    margin_val = wilson_margin(k, n)
                else:
                    prob_val, margin_val = None, None
                render_outcome_card(
                    selected_rule.icon,
                    selected_rule.title,
                    prob_val,
                    margin_val,
                )

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
            active_expl = st.session_state.get("_active_explanation")
            if active_expl and active_expl == selected_rule.name:
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
                        msg = active_expl_obj.text or active_expl_obj.status or ""
                        n_used = active_expl_obj.n_summarized
                        render_score_overview(
                            selected_rule.icon,
                            selected_rule.title,
                            msg,
                            n_used if active_expl_obj.text else None,
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
                    dataset=ctx.dataset,
                    sample_idx=selected_idx,
                    scenario=scenario,
                    model=backend.ethos_model,
                    max_model_len=backend.ethos_max_model_len,
                    temperature=st.session_state.get(
                        "ethos_temperature", DEFAULT_ETHOS_TEMPERATURE
                    ),
                    allowed_token_ids=get_allowed_token_ids(dataset_name, sc.dataset),
                )
                st.session_state["_estimator"] = new_est
                new_est.start()
                st.rerun(scope="fragment")

            # ── Cancel estimation ─────────────────────────────
            if estimating and cancel_clicked:
                estimator.cancel()
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
                    _trigger_explanation(selected_rule, estimator, backend, ctx)
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
