from langsmith import traceable
from researchmind.agent.state import AgentState
from researchmind.utils.llm_client import ResearchMindLLM

intent_labels = {"search", "citation", "compare", "gap_detection", "recent"}

ROUTER_SYSTEM_PROMPT = f"""You are a query classifier for a research intelligence system.

Classify the user's query into exactly one of the following intents:

- search: General questions about papers, methods, concepts, or findings. Use this as the default.
- citation: Questions about what papers cite a given work, or what foundational work a paper depends on or what references a given work. Keywords: cite, cites, citing, references to, builds on
- compare: Questions that explicitly compare two methods, models, architectures, or approaches against each other. Keywords: compare, versus, vs, difference
- gap_detection: Questions about unsolved problems, open challenges, or unexplored areas in a research domain. Keywords: unsolved, open challenges, unexplored, limitations, failure modes, bottleneck, unresolved
- recent: Questions about the latest, newest, or most recent work in an area. Keywords: latest, newest, recent, state-of-the-art, SOTA, newly published

Rules:
- Return only one of: {', '.join(intent_labels)}. No explanation, no punctuation, no extra words.
- If unsure, return search.

Examples:
- "How does DINO work?" → search
- "What papers cite Attention Is All You Need?" → citation
- "What built on this paper?" → citation
- "How does BERT compare to GPT in pre-training?" → compare
- "What problems in anomaly detection remain unsolved?" → gap_detection
- "What are the latest advances in OOD detection?" → recent
- "State-of-the-art results for natural language inference" → recent

"""


@traceable
def route(state: AgentState, llm: ResearchMindLLM) -> dict:
    query = state["query"]
    intent = llm.complete(
        user_prompt=query,
        system_prompt=ROUTER_SYSTEM_PROMPT,
        tier="fast",
        max_tokens=10,
        temperature=0.0,
    )
    intent = intent.strip().lower()
    # default to "search" if the intent is not recognized for some reason
    if intent not in intent_labels:
        intent = "search"
    return {"intent": intent}
