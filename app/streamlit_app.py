"""Nyaya-RAG — Streamlit demo front-end.

A thin client over the FastAPI backend (``nyaya.api.app``). It sends the user's
question to ``POST /query`` and renders the verified answer, the citation-
verification summary (ADR-005), and the retrieved statute sections.

Run (after the API server is up on CHAMP / locally)::

    pip install -e ".[rag]"
    streamlit run app/streamlit_app.py

Point it at the API with the sidebar field or the NYAYA_API_URL env var
(default http://localhost:9090).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("NYAYA_API_URL", "http://localhost:9090")
ERA_OPTIONS = {
    "Auto-detect": None,
    "New code (BNS / BNSS / BSA)": "new_code",
    "Old code (IPC / CrPC / IEA)": "old_code",
}

st.set_page_config(page_title="Nyaya-RAG", page_icon="⚖️", layout="wide")


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("⚖️ Nyaya-RAG")
    st.caption(
        "Era-aware, citation-verified RAG over Indian statutes. "
        "Fully local — no external APIs."
    )
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
    era_label = st.radio("Legal era", list(ERA_OPTIONS), index=0)
    top_k = st.slider("Sections to retrieve", min_value=2, max_value=12, value=8)

    st.divider()
    if st.button("Check API health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            r.raise_for_status()
            st.success(r.json())
        except requests.RequestException as exc:
            st.error(f"API unreachable: {exc}")


# --- main -------------------------------------------------------------------

st.title("Ask a question about Indian law")
st.caption(
    "Answers are grounded in retrieved statute sections; every citation is "
    "verified against the corpus before it is shown (ADR-005)."
)

question = st.text_area(
    "Your question",
    placeholder="e.g. What is the punishment for murder under the BNS?",
    height=100,
)

if st.button("Ask", type="primary"):
    if not question.strip() or len(question.strip()) < 5:
        st.warning("Please enter a question (at least 5 characters).")
        st.stop()

    payload: dict = {"question": question.strip(), "top_k": top_k}
    era = ERA_OPTIONS[era_label]
    if era is not None:
        payload["era"] = era

    try:
        with st.spinner("Retrieving sections and generating a verified answer…"):
            resp = requests.post(f"{api_url}/query", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    # --- answer ---
    st.subheader("Answer")
    st.markdown(data["answer"])

    meta = st.columns(3)
    meta[0].metric("Era used", data.get("era_used") or "—")
    meta[1].metric("Model", data.get("model") or "—")
    meta[2].metric("Completion tokens", data.get("completion_tokens", 0))

    # --- verification summary (ADR-005) ---
    v = data.get("verification")
    if v:
        st.subheader("Citation verification")
        cols = st.columns(4)
        cols[0].metric("Citations", v["total"])
        cols[1].metric("Verified", v["verified"])
        cols[2].metric("Stripped", v["ungrounded"] + v["hallucinated"])
        cols[3].metric("Hallucination rate", f"{v['hallucination_rate']:.0%}")
        if v["hallucinated"] or v["ungrounded"]:
            st.info(
                f"{v['hallucinated']} hallucinated and {v['ungrounded']} ungrounded "
                "citation(s) were removed from the answer above."
            )

    if data.get("citations"):
        st.write("**Verified citations:** " + ", ".join(data["citations"]))

    # --- retrieved sources ---
    st.subheader("Retrieved sections")
    for c in data.get("chunks", []):
        section = f" §{c['section']}" if c.get("section") else ""
        with st.expander(f"{c.get('act') or c['id']}{section}  ·  [{c['era']}]"):
            st.write(c["text"])
