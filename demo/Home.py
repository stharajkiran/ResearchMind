"""Home page for the configurable ResearchMind literature assistant."""

import streamlit as st

st.set_page_config(page_title="ResearchMind", page_icon="🧠", layout="wide")

st.title("🧠 ResearchMind")
st.subheader("Configurable literature assistant")
st.markdown("""
Use the sidebar to navigate:
- **Search** — hybrid retrieval over the configured corpus
- **Agent Chat** — cited research assistance for the configured corpus
- **Dashboard** — live corpus and feedback metrics

The portfolio release demonstrates these workflows with OOD and anomaly-detection
literature in computer vision.
""")
st.info("Backend: " + __import__('os').environ.get("BACKEND_URL", "http://localhost:8000"))
