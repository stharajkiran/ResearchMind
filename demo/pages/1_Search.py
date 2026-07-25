"""Streamlit search page for the configured ResearchMind corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import streamlit as st

from demo.config import BACKEND_URL, HEADERS

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Research Search", page_icon="🔍", layout="wide")

# Custom CSS for the "Paper Cards"
st.markdown("""
    <style>
    .paper-card {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .paper-title { font-weight: bold; color: #1E3A8A; font-size: 1.1rem; }
    .paper-title a { color: #1E3A8A; text-decoration: none; }
    .paper-title a:hover { text-decoration: underline; }
    .paper-meta { color: #6B7280; font-size: 0.85rem; margin-bottom: 0.75rem; }
    .excerpt { font-style: italic; color: #374151; margin-bottom: 0.5rem; }
    .section-label { font-weight: 600; color: #4CAF50; font-size: 0.8rem; text-transform: uppercase; }
    .rank-badge {
        float: right;
        background-color: #1E3A8A;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_status" not in st.session_state:
    st.session_state.search_status = "idle"

# --- UI LAYOUT: Sidebar Controls ---
with st.sidebar:
    st.header("Search Parameters")
    retrieval_mode = "standard"
    st.caption(
        "**Supported release mode:** Standard hybrid retrieval. "
        "Rewrite and HyDE remain local experimental modes."
    )
    k_value = st.slider("Top K Results", min_value=1, max_value=20, value=10)
    
    st.divider()
    st.info(f"Backend Target: `{BACKEND_URL}/search`")

# --- MAIN UI ---
st.title("🔍 Research Discovery")
st.caption("Search through indexed arXiv papers using semantic retrieval.")

# Input area
with st.form("search_form"):
    query = st.text_input("Enter your research query", placeholder="e.g., Transformer efficiency in low-resource settings")
    submit_button = st.form_submit_button("Run Search", use_container_width=True)

# --- SEARCH LOGIC ---
if submit_button:
    if not query.strip():
        st.session_state.search_status = "idle"
        st.warning("Please enter a query first.")
    else:
        payload = {
            "query": query,
            "k": k_value,
            "retrieval_mode": retrieval_mode
        }
        
        try:
            with st.spinner(f"Running {retrieval_mode} retrieval..."):
                # Calling your FastAPI backend
                response = httpx.post(f"{BACKEND_URL}/search", json=payload, headers=HEADERS, timeout=15.0)
                response.raise_for_status()
                # search endpoint returns list of chunks
                st.session_state.search_results = response.json()
                st.session_state.search_status = (
                    "results" if st.session_state.search_results else "no_results"
                )
                
        except httpx.ConnectError:
            st.session_state.search_status = "error"
            st.error(f"Could not connect to FastAPI backend. Is it running at {BACKEND_URL}?")
        except Exception as e:
            st.session_state.search_status = "error"
            st.error(f"Search failed: {str(e)}")

# --- DISPLAY RESULTS ---
if st.session_state.search_results:
    results = st.session_state.search_results

    # Group chunks by paper_id, preserving retrieval order of first appearance.
    papers: dict[str, dict] = {}
    for res in results:
        pid = res.get("paper_id", "")
        if pid not in papers:
            papers[pid] = {
                "title": res.get("title") or "Unknown Title",
                "authors": res.get("authors") or [],
                "year": res.get("year"),
                "chunks": [],
            }
        papers[pid]["chunks"].append(res)

    st.subheader(f"Found {len(results)} chunks across {len(papers)} paper(s)")

    for rank, (pid, paper) in enumerate(papers.items(), start=1):
        arxiv_url = f"https://arxiv.org/abs/{pid}" if pid else None

        # authors: first three, then "et al." if more
        authors = paper["authors"]
        if len(authors) > 3:
            author_str = ", ".join(authors[:3]) + " et al."
        else:
            author_str = ", ".join(authors)

        meta_bits = []
        if author_str:
            meta_bits.append(author_str)
        if paper["year"]:
            meta_bits.append(str(paper["year"]))
        if pid:
            meta_bits.append(f"arXiv:{pid}")
        meta_bits.append(f"{len(paper['chunks'])} matching chunk(s)")
        meta_line = " · ".join(meta_bits)

        title = paper["title"]
        title_html = (
            f'<a href="{arxiv_url}" target="_blank">{title}</a>' if arxiv_url else title
        )

        # One excerpt block per matching chunk, labelled by section.
        excerpts_html = ""
        for c in paper["chunks"]:
            section = c.get("section", "General")
            excerpts_html += (
                f'<div class="section-label">&sect; {section}</div>'
                f'<div class="excerpt">"{c.get("text", "")[:400]}..."</div>'
            )

        st.markdown(f"""
            <div class="paper-card">
                <span class="rank-badge">Rank {rank}</span>
                <div class="paper-title">{title_html}</div>
                <div class="paper-meta">{meta_line}</div>
                {excerpts_html}
            </div>
        """, unsafe_allow_html=True)
elif st.session_state.search_status == "no_results":
    st.info(
        "No evidence was returned from the configured corpus. Try a narrower "
        "query that matches its indexed research domain."
    )
elif st.session_state.search_status == "idle":
    st.info("Results will appear here after you trigger a search.")
