"""BiasLens demo: score a resume against a JD, then swap the name / college and see the score move.
Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from biaslens import COLLEGES, MODELS, NAME_GROUPS, PREFIX

st.set_page_config(page_title="BiasLens", layout="wide")
st.title("BiasLens: resume-JD match + bias check")


@st.cache_resource
def load(key):
    return SentenceTransformer(MODELS[key])


def score(model, key, resume, jd):
    pre = PREFIX.get(key, "")
    e = model.encode([pre + resume, pre + jd], normalize_embeddings=True)
    return float(e[0] @ e[1])


key = st.sidebar.selectbox("Embedding model", list(MODELS), index=1)
model = load(key)

c1, c2 = st.columns(2)
resume = c1.text_area("Resume text", height=300)
jd = c2.text_area("Job description", height=300)
name = c1.text_input("Candidate name as written in the resume")
college = c2.text_input("College name as written in the resume")

if st.button("Run match + bias check") and resume and jd:
    base = score(model, key, resume, jd)
    st.metric("Match score (cosine)", f"{base:.4f}")

    rows = []
    if name and name in resume:
        for g, names in NAME_GROUPS.items():
            s = score(model, key, resume.replace(name, names[0]), jd)
            rows.append(dict(swap=f"name: {g}", score=s, delta=s - base))
    elif name:
        st.warning("Name not found in the resume text, skipping name swaps.")
    if college and college in resume:
        for tier, cols in COLLEGES.items():
            s = score(model, key, resume.replace(college, cols[0]), jd)
            rows.append(dict(swap=f"college: {tier}", score=s, delta=s - base))
    elif college:
        st.warning("College not found in the resume text, skipping college swaps.")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.round(4), use_container_width=True)
        st.bar_chart(df.set_index("swap")["delta"])
        st.caption("Delta = score after swap minus original. Identical skills, only the attribute changes.")