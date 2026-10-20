"""Minimal Streamlit demo for span extraction.

    streamlit run app/predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.predict import RESEARCH_WARNING, predict_text

st.set_page_config(page_title="MedNoteClassifier", layout="wide")
st.title("MedNoteClassifier")
st.caption("Clinical feature span extraction · NBME patient notes · research only")
st.warning(RESEARCH_WARNING)

text = st.text_area("Patient note", height=220, placeholder="Paste a de-identified patient history…")
col1, col2 = st.columns(2)
with col1:
    case = st.selectbox("Case (optional)", options=["all"] + list(range(10)))
with col2:
    model = st.selectbox("Model", options=["hybrid", "transformer"])

if st.button("Extract features", type="primary"):
    case_num = None if case == "all" else int(case)
    result = predict_text(text, case_num=case_num, model_name=model)
    st.json(result)
    if result.get("entities"):
        st.subheader("Spans")
        st.dataframe(result["entities"], use_container_width=True)
