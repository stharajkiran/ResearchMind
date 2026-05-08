from prometheus_client import Counter, Histogram

cache_hits_total = Counter(
    "researchmind_cache_hits_total",
    "Number of query cache hits",
)

validation_blocks_total = Counter(
    "researchmind_validation_blocks_total",
    "Number of responses blocked by the validator pipeline",
)

tool_calls_total = Counter(
    "researchmind_tool_calls_total",
    "Number of agent tool calls by tool name",
    ["tool_name"],
)

feedback_scores = Histogram(
    "researchmind_feedback_scores",
    "Distribution of user feedback ratings (1–5)",
    buckets=[1, 2, 3, 4, 5],
)
