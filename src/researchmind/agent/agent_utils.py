from researchmind.utils.llm_client import ResearchMindLLM

CITATION_DIRECTION_PROMPT = """You are classifying the direction of a citation query.

- inbound: the query asks what papers cite a given work. What built on this? Who referenced this?
- outbound: the query asks what papers a given work cites. What does this depend on? What are its foundations?
- both: the query asks about the full citation context in both directions.

Return only one word: inbound, outbound, or both. No explanation.

Examples:
- "What papers cite Attention Is All You Need?" → inbound
- "What work did ResNet build on?" → outbound
- "What is this paper's relationship to the field?" → both
- "What built on DINO?" → inbound
- "What are the foundational works behind this method?" → outbound
"""

COMPARE_METHODOLOGIES_PROMPT = """Extract the methods or approaches being compared in this query.
Return a JSON object with a single key "subjects" containing a list of strings.

Examples:
- "How does BERT compare to GPT?" → {"subjects": ["BERT", "GPT"]}
- "Compare ResNet, VGG, and EfficientNet" → {"subjects": ["ResNet", "VGG", "EfficientNet"]}
- "DINO vs MAE for anomaly detection" → {"subjects": ["DINO", "MAE"]}

Return only the JSON object. No explanation.
"""


def classify_citation_direction(
    query: str, system_prompt: str, llm: ResearchMindLLM
) -> str:
    direction = llm.complete(
        user_prompt=query,
        system_prompt=system_prompt,
        tier="fast",
        max_tokens=10,
        temperature=0.0,
    )
    direction = direction.strip().lower()
    if direction not in {"inbound", "outbound", "both"}:
        direction = "outbound"  # default to outbound if unclear
    return direction
