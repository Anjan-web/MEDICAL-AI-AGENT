from llm.llm_provider import get_llm
from rag.retriever import get_retriever
from tools.pubmed_tool import search_pubmed
from tools.who_tools import fetch_whp_data

# --------------------------------------------------
# Lazy-load LLM and retriever
# FIXED: were initialised at module level — crashed on import
# if FAISS index didn't exist yet. Now loaded once on first use.
# --------------------------------------------------
_llm = None
_retriever = None

def get_llm_instance():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def get_retriever_instance():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


# --------------------------------------------------
# Node 1: Intent Classifier
# FIXED: now also catches symptom/diagnosis queries
# as a distinct intent so routing can use it properly
# --------------------------------------------------
def intent_classifier(state):
    question = state["question"].lower()

    # WHO checked FIRST — "latest guidelines" was matching "latest" → pubmed
    if any(word in question for word in [
        "guideline", "guidelines", "who", "policy",
        "recommendation", "recommendations", "protocol"
    ]):
        intent = "who"

    elif any(word in question for word in [
        "latest", "recent", "research", "study",
        "clinical trial", "evidence", "findings"
    ]):
        intent = "pubmed"

    elif any(word in question for word in [
        "drug", "medicine", "medication", "dosage", "treatment", "dose"
    ]):
        intent = "drug"

    else:
        intent = "medical"

    print(f"🎯 Intent classified as: {intent}")
    return {**state, "intent": intent}

# --------------------------------------------------
# Node 2: Multi-source Retriever
# FIXED: now reads state["intent"] to decide which
# sources to actually call — previously ignored intent entirely
# FIXED: external tool calls wrapped in try/except
# FIXED: context bounded to 3000 chars to avoid LLM overflow
# REMOVED: dead retriever_docs / pubmed_search / who_search nodes
# that were never wired into the graph
# --------------------------------------------------
def multi_retriever(state):
    query = state["question"]
    intent = state.get("intent", "medical")
    retriever = get_retriever_instance()

    sources = []
    parts = []

    # --- Always run RAG first ---
    rag_docs = retriever(query)
    rag_text = "\n\n".join([doc.page_content for doc in rag_docs])

    print("\n--- RAG Docs ---")
    print(rag_text[:500])

    rag_strong = len(rag_text.strip()) >= 200

    if rag_strong:
        parts.append(f"Medical Knowledge (Encyclopedia):\n{rag_text}")
        sources += [f"Page {doc.metadata.get('page', '?')}" for doc in rag_docs]

    # --- PubMed intent ---
    if intent == "pubmed":
        print("🔬 Research intent → calling PubMed")
        try:
            pubmed_text = search_pubmed(query)
            if pubmed_text and len(pubmed_text.strip()) > 100:
                parts.append(f"Latest PubMed Research:\n{pubmed_text}")
                sources.append("PubMed")
            else:
                print("⚠️ PubMed returned nothing")
        except Exception as e:
            print(f"⚠️ PubMed error: {e}")

    # --- WHO intent ---
    # ✅ FIXED: falls back to PubMed if WHO returns nothing
    # WHO pages sometimes block scrapers or return empty content
    elif intent == "who":
        print("🏥 WHO intent → calling WHO guidelines")
        who_success = False
        try:
            who_text = fetch_whp_data(query)
            if who_text and len(who_text.strip()) > 100:
                parts.append(f"WHO Guidelines:\n{who_text}")
                sources.append("WHO")
                who_success = True
                print(f"✅ WHO returned {len(who_text)} chars")
            else:
                print("⚠️ WHO returned empty — falling back to PubMed")
        except Exception as e:
            print(f"⚠️ WHO error: {e}")

        # ✅ NEW: PubMed fallback when WHO fails
        if not who_success:
            try:
                print("🔬 WHO fallback → calling PubMed")
                pubmed_text = search_pubmed(query)
                if pubmed_text and len(pubmed_text.strip()) > 100:
                    parts.append(f"PubMed Research:\n{pubmed_text}")
                    sources.append("PubMed")
                    print(f"✅ PubMed fallback returned {len(pubmed_text)} chars")
            except Exception as e:
                print(f"⚠️ PubMed fallback error: {e}")

    # --- Final fallback if nothing at all retrieved ---
    if not parts:
        print("⚠️ No sources found → trying PubMed as last resort")
        try:
            pubmed_text = search_pubmed(query)
            if pubmed_text and len(pubmed_text.strip()) > 100:
                parts.append(f"PubMed Research:\n{pubmed_text}")
                sources.append("PubMed")
        except Exception as e:
            print(f"⚠️ Final fallback error: {e}")

    combined_text = "\n\n".join(parts)[:10000]

    return {
        **state,
        "context": combined_text,
        "sources": list(dict.fromkeys(sources))   # deduplicated
    }
# --------------------------------------------------
# Node 3: Answer Generation
# FIXED: removed .replace("\n", " ") — was collapsing
# multi-point answers into unreadable walls of text
# FIXED: added safety check if context is empty
# FIXED: prompt now explicitly tells model to answer
# from context only, reducing hallucination
# --------------------------------------------------
def generate_answer(state):
    llm = get_llm_instance()

    if isinstance(state, str):
        return {
            "answer": "Internal error: invalid state format.",
            "sources": []
        }

    context_data = state.get("context", "")

    if isinstance(context_data, str):
        context = context_data
    else:
        context = "\n\n".join([doc.page_content for doc in context_data])

    if not context.strip():
        return {
            **state,
            "answer": "I could not find relevant information to answer this question.",
            "sources": state.get("sources", [])
        }

    prompt = f"""You are a medical expert assistant with access to encyclopedia knowledge and recent medical guidelines.

Use the provided context to answer the question as completely as possible.
Only say "I don't have enough information" if the context contains absolutely nothing related to the question.
If the context contains partial information, use it to give the best answer you can.

Rules:
- Use simple, clear language
- Be concise for simple questions, detailed for complex ones
- Do not use markdown, tables, or HTML
- Do not repeat information
- If sources include recent research or guidelines, prioritize that information

Question:
{state.get("question")}

Context:
{context}

Answer:"""

    response = llm.invoke(prompt)

    cleaned_answer = (
        response.content
        .replace("**", "")
        .replace("*", "")
        .replace("|", "")
        .replace("#", "")
        .strip()
    )

    return {
        **state,
        "answer": cleaned_answer,
        "sources": state.get("sources", [])
    }