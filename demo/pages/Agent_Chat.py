import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import httpx
import uuid
from demo.config import BACKEND_URL

# --- CONFIG ---
st.set_page_config(page_title="Agent Chat", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# RESPONSE RENDERER
# Detects which of the three agent response shapes was returned and renders
# the appropriate UI. Called in both the history loop and the immediate render
# so the logic lives in exactly one place.
# ---------------------------------------------------------------------------
def _detect_type(raw: dict) -> str:
    """Infer response type from unique keys — no type tag needed."""
    if "response" in raw:
        return "rag"
    if "gaps" in raw:
        return "gap"
    if "comparison" in raw:
        return "comparison"
    return "unknown"


@st.cache_data(show_spinner=False)
def _fetch_paper_titles(paper_ids: tuple[str, ...]) -> dict[str, str]:
    """Fetch paper titles from the local backend /paper/{paper_id} endpoint.
    Cached globally — reruns never re-fetch the same IDs.
    Falls back to the raw paper_id string on any error.
    """
    if not paper_ids:
        return {}
    result: dict[str, str] = {}
    for pid in paper_ids:
        try:
            r = httpx.get(f"{BACKEND_URL}/paper/{pid}", timeout=3.0)
            if r.status_code == 200:
                result[pid] = r.json().get("title", pid)
        except Exception:
            pass  # fallback to raw ID handled in _render_sources
    return result


def _all_paper_ids(raw: dict) -> tuple[str, ...]:
    """Collect every unique paper ID from any response type for a single cache key."""
    ids: set[str] = set()
    t = _detect_type(raw)
    if t == "rag":
        ids.update(raw.get("sources", []))
    elif t == "gap":
        for gap in raw.get("gaps", []):
            ids.update(gap.get("supporting_paper_ids", []))
    elif t == "comparison":
        ids.update(raw.get("sources", []))
        for s in raw.get("summaries", []):
            ids.update(s.get("sources", []))
    return tuple(sorted(ids))


def _render_sources(label: str, paper_ids: list[str], titles: dict[str, str]) -> None:
    """Render a labelled vertical bullet list of paper titles linked to arXiv."""
    if not paper_ids:
        return
    st.markdown(f"**{label}**")
    for pid in paper_ids:
        title = titles.get(pid, pid)  # fall back to raw ID if title fetch failed
        st.markdown(f"- [{title}](https://arxiv.org/abs/{pid})")


def render_assistant_content(raw_answer: dict | str) -> None:
    """Render agent answer. raw_answer is either a plain string or a response dict."""
    if not isinstance(raw_answer, dict):
        st.markdown(raw_answer)
        return

    response_type = _detect_type(raw_answer)
    titles = _fetch_paper_titles(_all_paper_ids(raw_answer))

    if response_type == "rag":
        st.markdown(raw_answer["response"])
        _render_sources("Sources", raw_answer.get("sources", []), titles)
        if raw_answer.get("confidence") is not None:
            st.caption(f"Confidence: {raw_answer['confidence']:.0%}")

    elif response_type == "gap":
        st.caption(f"Topic: {raw_answer.get('topic', 'Unknown')}")
        for gap in raw_answer.get("gaps", []):
            with st.expander(gap["description"][:120], expanded=True):
                st.markdown(gap["description"])
                _render_sources(
                    "Supporting papers", gap.get("supporting_paper_ids", []), titles
                )
                st.caption(f"Confidence: {gap['confidence']:.0%}")

    elif response_type == "comparison":
        summaries = raw_answer.get("summaries", [])
        if summaries:
            tabs = st.tabs([s["method"] for s in summaries])
            for tab, summary in zip(tabs, summaries):
                with tab:
                    st.markdown(summary["summary"])
                    _render_sources("Sources", summary.get("sources", []), titles)
        st.markdown("**Cross-method comparison**")
        st.markdown(raw_answer["comparison"])
        _render_sources("All sources", raw_answer.get("sources", []), titles)
        if raw_answer.get("confidence") is not None:
            st.caption(f"Confidence: {raw_answer['confidence']:.0%}")

    else:
        st.markdown(str(raw_answer))


# --- SESSION STATE INIT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": str,
    #                "feedback_id": int|None, "rated": bool, "rating": int|None}
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("Chat Settings")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["standard", "hyde", "rewrite"],
        help="HyDE generates a hypothetical answer to improve embedding search.",
    )
    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

# --- TITLE ---
st.title("🤖 ResearchMind Agent")
st.caption(
    "Ask research questions. The agent retrieves context and generates grounded answers."
)

# --- RENDER CHAT HISTORY ---
STAR_OPTIONS = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_content(msg["content"])
        else:
            st.markdown(msg["content"])

        # Feedback widget — only for assistant messages that have a feedback_id
        if msg["role"] == "assistant" and msg.get("feedback_id") is not None:
            feedback_id = msg["feedback_id"]

            if msg.get("rated"):
                st.caption(f"✅ You rated this {msg['rating']} star(s). Thanks!")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected = st.select_slider(
                        "Rate this answer",
                        options=STAR_OPTIONS,
                        value="⭐⭐⭐",
                        key=f"rating_{feedback_id}",
                        label_visibility="collapsed",
                    )
                with col2:
                    if st.button("Submit", key=f"submit_{feedback_id}"):
                        rating = selected.count("⭐")
                        try:
                            httpx.post(
                                f"{BACKEND_URL}/feedback",
                                json={"feedback_id": feedback_id, "rating": rating},
                                timeout=5.0,
                            )
                            st.session_state.messages[i]["rated"] = True
                            st.session_state.messages[i]["rating"] = rating
                            st.rerun()
                        except Exception as e:
                            st.error(f"Feedback submission failed: {e}")

# --- CHAT INPUT ---
if user_input := st.chat_input("Ask a research question..."):
    # Immediately show the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the agent and show the response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/agent",
                    json={
                        "query": user_input,
                        "session_id": st.session_state.session_id,
                        "retrieval_mode": retrieval_mode,
                    },
                    timeout=60.0,  # agent involves LLM calls — give it time
                )
                response.raise_for_status()
                data = response.json()
                raw_answer = data.get("answer", "No answer returned.")
                feedback_id = data.get("feedback_id")

                render_assistant_content(raw_answer)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": raw_answer,  # store full dict; renderer handles all types
                        "feedback_id": feedback_id,
                        "rated": False,
                        "rating": None,
                    }
                )

            except httpx.ConnectError:
                err = f"Could not connect to backend at {BACKEND_URL}. Is it running?"
                st.error(err)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err,
                        "feedback_id": None,
                        "rated": False,
                        "rating": None,
                    }
                )
            except Exception as e:
                err = f"Agent call failed: {e}"
                st.error(err)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err,
                        "feedback_id": None,
                        "rated": False,
                        "rating": None,
                    }
                )

    # Rerun so the history loop re-renders everything cleanly,
    # including the feedback widget for the new assistant message.
    st.rerun()
