from langgraph.graph import StateGraph, START, END
from agents.nodes import (
    intent_classifier,
    multi_retriever,
    generate_answer
)
# ✅ REMOVED: search_pubmed and fetch_whp_data imports
# — these are tool functions, not graph nodes.
# multi_retriever already calls them internally based on intent.


def build_graph():
    graph = StateGraph(dict)

    # ✅ Only 3 nodes needed
    graph.add_node("intent", intent_classifier)
    graph.add_node("multi", multi_retriever)
    graph.add_node("generate", generate_answer)

    # Entry
    graph.set_entry_point("intent")

    # ✅ FIXED: all intents route to multi_retriever
    # multi_retriever reads state["intent"] and calls
    # PubMed, WHO, or RAG-only accordingly
    graph.add_conditional_edges(
        "intent",
        lambda state: state["intent"],
        {
            "medical": "multi",
            "drug":    "multi",
            "pubmed":  "multi",   # ✅ was incorrectly going to raw pubmed tool
            "who":     "multi"    # ✅ was incorrectly going to raw who tool
        }
    )

    # Linear flow after retrieval
    graph.add_edge("multi", "generate")
    graph.set_finish_point("generate")

    return graph.compile()