"""Self-contained Streamlit rendering functions.

Each function owns its HTML and ``st.*`` calls so callers only pass data.
"""

import streamlit as st

# ── Sidebar ──────────────────────────────────────────────────────────────────


def render_status_indicator(healthy: bool) -> None:
    """Render the backend health status dot in the sidebar."""
    dot_cls = "ok" if healthy else "err"
    label = "Connected" if healthy else "Unreachable"
    st.markdown(
        f"<div class='status-row'>"
        f"<span style='color:gray;font-size:0.8em'>ETHOS</span>"
        f"<span class='status-dot {dot_cls}'></span>"
        f"<span class='status-text'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Demographics ─────────────────────────────────────────────────────────────


def render_demographics(demographics: dict[str, str]) -> None:
    """Render the demographics header and value columns."""
    st.subheader("Demographics")
    cols = st.columns(len(demographics))
    for col, (key, value) in zip(cols, demographics.items(), strict=False):
        col.markdown(
            f"<span style='color:gray'>{key}</span><br>"
            f"<span style='font-size:1.3em'>{value}</span>",
            unsafe_allow_html=True,
        )


# ── EHR stats ────────────────────────────────────────────────────────────────

_SECTION_HDR = "<span style='font-size:1.1em;font-weight:600'>{title}</span>"
_STAT_HTML = (
    "<span style='color:gray'>{label}</span><br><span style='font-size:1.3em'>{value}</span>"
)


def render_ehr_stats(
    present_time_span: str,
    present_count: int,
    past_time_span: str,
    past_count: int,
) -> None:
    """Render the EHR section with Current Encounter / History stats."""
    st.subheader("EHR")
    col_enc, col_hist = st.columns(2)
    with col_enc:
        st.markdown(_SECTION_HDR.format(title="Current Encounter"), unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        e1.markdown(
            _STAT_HTML.format(label="Time Span", value=present_time_span),
            unsafe_allow_html=True,
        )
        e2.markdown(
            _STAT_HTML.format(label="Events", value=f"{present_count:,}"),
            unsafe_allow_html=True,
        )
    with col_hist:
        st.markdown(_SECTION_HDR.format(title="History"), unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        h1.markdown(
            _STAT_HTML.format(label="Time Span", value=past_time_span),
            unsafe_allow_html=True,
        )
        h2.markdown(
            _STAT_HTML.format(label="Events", value=f"{past_count:,}"),
            unsafe_allow_html=True,
        )


# ── EHR Summary ──────────────────────────────────────────────────────────────

_SUMMARY_BOX = (
    "<div style='min-height:120px'>"
    "<span style='color:gray'>EHR Summary</span><br>"
    "<span style='font-size:1.1em'>{msg}</span>"
    "</div>"
)
_SUMMARY_UNAVAILABLE = (
    "<div style='min-height:120px'>"
    "<span style='color:gray'>EHR Summary</span><br>"
    "<span style='font-size:1.1em;color:gray'>Summary not available.</span></div>"
)


def render_summary(msg: str | None, available: bool) -> None:
    """Render the EHR summary box.

    *msg* is the current text (or ``None``).  *available* indicates whether
    an LLM model is configured — when ``False`` and *msg* is empty the
    "unavailable" placeholder is shown.
    """
    if msg:
        st.markdown(_SUMMARY_BOX.format(msg=msg), unsafe_allow_html=True)
    elif not available:
        st.markdown(_SUMMARY_UNAVAILABLE, unsafe_allow_html=True)
    else:
        st.markdown(_SUMMARY_BOX.format(msg=""), unsafe_allow_html=True)


# ── Outcome estimation ───────────────────────────────────────────────────────


def render_estimate_button_fill(pct: int) -> None:
    """Inject dynamic gradient CSS for the estimate/cancel button."""
    fill = (
        f"background-image:linear-gradient(to right,"
        f"rgba(255,255,255,0.10) {pct}%,"
        f"transparent {pct}%)!important;"
        if pct
        else ""
    )
    st.markdown(
        f"<style>.st-key-est_btn button{{"
        f"background-color:rgba(255,255,255,0.06)!important;"
        f"{fill}}}</style>",
        unsafe_allow_html=True,
    )


def render_explainer_status(status_text: str) -> None:
    """Render the 'Analyzing trajectories…' progress text."""
    st.markdown(
        f"<div style='padding-top:8px'>"
        f"<span style='color:gray;font-size:0.85em'>"
        f"\U0001f4a1 {status_text}</span></div>",
        unsafe_allow_html=True,
    )


def render_outcome_card(
    icon: str,
    title: str,
    prob: float | None = None,
    margin: float | None = None,
) -> None:
    """Render the centered outcome probability card."""
    html = (
        "<div class='outcome-card'>"
        f"<span style='font-size:5em'>{icon}</span><br>"
        f"<b style='font-size:1.3em'>{title}</b>"
    )
    if prob is not None and margin is not None:
        html += (
            f"<div style='font-size:2em'>"
            f"{prob:.0%}"
            f"<span style='display:inline-block;width:0;"
            f"overflow:visible;vertical-align:baseline'>"
            f"<span style='font-size:0.42em;color:gray;"
            f"margin-left:4px;white-space:nowrap'>"
            f"\u00b1{margin * 100:.1f}%</span></span></div>"
        )
    else:
        html += "<div style='font-size:2em;color:gray'>???</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_score_overview(
    icon: str,
    title: str,
    msg: str,
    n_trajectories: int | None = None,
) -> None:
    """Render the Score Overview box with an optional footnote."""
    footnote = ""
    if n_trajectories is not None and n_trajectories > 0:
        footnote = (
            "<div style='color:gray;font-size:0.75em;"
            f"margin-top:0.5em'>Based on {n_trajectories} "
            "ETHOS trajectories</div>"
        )
    paragraphs = [p.strip() for p in msg.split("\n\n") if p.strip()] if msg else []
    body = (
        "".join(f"<p style='margin:0 0 0.75em'>{p}</p>" for p in paragraphs) if paragraphs else msg
    )
    st.markdown(
        f"<div style='min-height:100px;background:rgba(255,255,255,0.04);"
        f"border-radius:10px;padding:1em;margin-top:1em'>"
        f"<div style='font-size:1.25em;font-weight:600;"
        f"margin-bottom:0.3em'>Score Overview</div>"
        f"<div style='color:gray;font-size:0.95em;margin-bottom:0.6em'>"
        f"{icon} &ensp; {title}</div>"
        f"<div style='font-size:1.05em'>{body}</div>"
        f"{footnote}"
        f"</div>",
        unsafe_allow_html=True,
    )
