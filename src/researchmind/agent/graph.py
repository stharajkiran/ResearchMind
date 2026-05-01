

# START → read_session_memory → route → [conditional on intent] → tool → synthesise_answer → END
#                                                               ↘ detect_research_gaps → END


from functools import partial
from langgraph.graph import StateGraph, END
from researchmind.agent.state import AgentState
from researchmind.agent.router import route
from researchmind.agent.tools import read_session_memory, search_corpus, search_recent, synthesise_answer, trace_citation_graph, compare_methodologies, detect_research_gaps

def build_graph(retriever, llm, citation_graph):
    builder = StateGraph(AgentState)
    
    # add nodes with dependencies bound
    builder.add_node("read_session_memory", read_session_memory)
    builder.add_node("route", partial(route, llm=llm))
    builder.add_node("search_corpus", partial(search_corpus, retriever=retriever))
    builder.add_node("search_recent", partial(search_recent, retriever=retriever, recency_decay_rate=0.1))
    builder.add_node("trace_citation_graph", partial(trace_citation_graph, retriever=retriever, llm=llm, graph=citation_graph))
    builder.add_node("compare_methodologies", partial(compare_methodologies, retriever=retriever, llm=llm))
    builder.add_node("detect_research_gaps", partial(detect_research_gaps, retriever=retriever, llm=llm))
    builder.add_node("synthesise_answer", partial(synthesise_answer, llm=llm))

    
    # entry
    builder.set_entry_point("read_session_memory")
    builder.add_edge("read_session_memory", "route")
    
    # conditional routing
    builder.add_conditional_edges("route", lambda state: state["intent"], {
        "search": "search_corpus",
        "recent": "search_recent",
        "citation": "trace_citation_graph",
        "compare": "compare_methodologies",
        "gap_detection": "detect_research_gaps",
    })
    
    # tool → synthesis (except gap_detection)
    for tool in ["search_corpus", "search_recent", "trace_citation_graph", "compare_methodologies"]:
        builder.add_edge(tool, "synthesise_answer")
    
    builder.add_edge("synthesise_answer", END)
    builder.add_edge("detect_research_gaps", END)
    
    return builder.compile()
