"""Streamlit dashboard: live corpus and feedback metrics from the backend /stats endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pandas as pd
import streamlit as st

from demo.config import BACKEND_URL, HEADERS

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 ResearchMind Dashboard")
st.caption("Live corpus and feedback metrics pulled from the backend.")


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_stats() -> dict:
    """Fetch aggregated stats from the backend. Cached briefly so reruns are cheap."""
    r = httpx.get(f"{BACKEND_URL}/stats", headers=HEADERS, timeout=15.0)
    r.raise_for_status()
    return r.json()


col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        _fetch_stats.clear()

try:
    stats = _fetch_stats()
except httpx.ConnectError:
    st.error(f"Could not connect to backend at {BACKEND_URL}. Is it running?")
    st.stop()
except Exception as e:
    st.error(f"Failed to load stats: {e}")
    st.stop()

# --- TOP-LINE METRICS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Corpus papers", f"{stats['corpus_size']:,}")
c2.metric("Total queries", f"{stats['total_queries']:,}")
avg_rating = stats.get("avg_rating")
c3.metric("Avg rating", f"{avg_rating:.2f} ⭐" if avg_rating is not None else "—")
pass_rate = stats.get("validation_pass_rate")
c4.metric(
    "Validation pass rate",
    f"{pass_rate:.0%}" if pass_rate is not None else "—",
)

st.divider()

# --- CHARTS ROW ---
left, right = st.columns(2)

with left:
    st.subheader("Rating distribution")
    dist = stats.get("rating_distribution") or {}
    if any(dist.values()):
        df = pd.DataFrame(
            {"stars": [f"{k}⭐" for k in dist], "count": list(dist.values())}
        ).set_index("stars")
        st.bar_chart(df, height=280)
    else:
        st.info("No ratings submitted yet.")

with right:
    st.subheader("Query intents")
    intents = stats.get("intent_distribution") or {}
    if intents:
        df = (
            pd.DataFrame({"intent": list(intents), "count": list(intents.values())})
            .set_index("intent")
            .sort_values("count", ascending=False)
        )
        st.bar_chart(df, height=280)
    else:
        st.info("No queries recorded yet.")
