import streamlit as st
from pathlib import Path

from src.nlp import classify_text, qa_over_readme

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.title("🧠 NLP / LLM Assistant")

tab1, tab2 = st.tabs(["Text Classification", "QA over README (RAG-lite)"])

with tab1:
    st.write("Task: sentiment-style classification (HF if available, fallback otherwise).")
    text = st.text_area("Enter text", "Air quality is improving in Thimphu and the system works well.")
    if st.button("Classify"):
        res = classify_text(text)
        st.write("Mode:", res.detail)
        st.json(res.data)

with tab2:
    st.write("Ask questions about your README.md.")
    q = st.text_input("Question", "How do I run the FastAPI service?")
    if st.button("Answer"):
        res = qa_over_readme(q, PROJECT_ROOT)
        st.write("Mode:", res.detail)
        st.json(res.data)