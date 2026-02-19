"""ETHOS Demo - Streamlit application."""

import time

import polars as pl
import streamlit as st

from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, send_completion_request

st.set_page_config(page_title="ETHOS Demo", layout="wide")
st.title("ETHOS Demo")

# ── Sidebar configuration ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    base_url = st.text_input("Model endpoint", value=DEFAULT_BASE_URL)
    model_id = st.text_input("Model ID", value=DEFAULT_MODEL)
    num_requests = st.number_input("Number of requests", min_value=1, max_value=500, value=10)

# ── Main area ───────────────────────────────────────────────────────────────
prompt = st.text_area("Prompt (sent to the model for each request)", value="Hello, who are you?")

if st.button("Run"):
    latencies: list[float] = []
    progress = st.progress(0, text="Sending requests...")

    for i in range(num_requests):
        t0 = time.perf_counter()
        try:
            send_completion_request(
                prompt,
                model=model_id,
                base_url=base_url,
            )
            latencies.append(time.perf_counter() - t0)
        except Exception as exc:
            st.error(f"Request {i + 1} failed: {exc}")
            latencies.append(float("nan"))
        progress.progress((i + 1) / num_requests, text=f"Request {i + 1}/{num_requests}")

    progress.empty()

    df = pl.DataFrame({"request": range(1, len(latencies) + 1), "latency_s": latencies})
    st.subheader("Results")
    st.line_chart(df, x="request", y="latency_s")
    st.dataframe(df)


def main():
    """Entrypoint for the `ethos-app` CLI command."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless=true"]
    st_main()
