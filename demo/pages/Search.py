import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import streamlit as st
import httpx
from demo.config import BACKEND_URL

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
    .paper-meta { color: #6B7280; font-size: 0.85rem; margin-bottom: 0.5rem; }
    .excerpt { font-style: italic; color: #374151; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# --- UI LAYOUT: Sidebar Controls ---
with st.sidebar:
    st.header("Search Parameters")
    retrieval_mode = st.selectbox(
        "Retrieval Mode", 
        options=["standard", "hyde", "rewrite"],
        help="HyDE generates a hypothetical answer to improve embedding search."
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
                response = httpx.post(f"{BACKEND_URL}/search", json=payload, timeout=15.0)
                response.raise_for_status()
                # search endpoint returns list of chunks
                st.session_state.search_results = response.json()
                
        except httpx.ConnectError:
            st.error(f"Could not connect to FastAPI backend. Is it running at {BACKEND_URL}?")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

# --- DISPLAY RESULTS ---
if st.session_state.search_results:
    st.subheader(f"Found {len(st.session_state.search_results)} relevant chunks")
    
    for res in st.session_state.search_results:
        # Using the custom CSS classes defined above
        st.markdown(f"""
            <div class="paper-card">
                <div class="paper-title">{res.get('title', 'Unknown Title')}</div>
                <div class="paper-meta">
                    ID: {res.get('paper_id')} | Section: {res.get('section', 'General')}
                </div>
                <div class="excerpt">"{res.get('text', '')[:400]}..."</div>
            </div>
        """, unsafe_allow_html=True)
else:
    if not submit_button:
        st.info("Results will appear here after you trigger a search.")