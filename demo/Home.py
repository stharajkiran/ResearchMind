import streamlit as st

st.set_page_config(page_title="ResearchMind", page_icon="🧠", layout="wide")

st.title("🧠 ResearchMind")
st.subheader("Self-hosted research intelligence for OOD detection literature")
st.markdown("""
Use the sidebar to navigate:
- **Search** — semantic retrieval over 233 OOD/CV papers
- **Agent Chat** — multi-turn research assistant with citations
""")
st.info("Backend: " + __import__('os').environ.get("BACKEND_URL", "http://localhost:8000"))
