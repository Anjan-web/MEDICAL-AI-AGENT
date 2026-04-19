from fastapi import FastAPI
from agents.graph_builder import build_graph
from safety.guardrails import check_safety, add_disclaimer

app = FastAPI(
    title="Medical AI Agent",
    description="Answers medical questions using RAG + PubMed + WHO Guidelines",
    version="1.0.0"
)

graph = build_graph()


@app.get("/")
def root():
    return {
        "message": "Medical AI Agent API",
        "docs": "/docs",
        "ask": "/ask?question=your+question"
    }


@app.get("/ask")
def ask(question: str):

    # ✅ Safety check first — block harmful queries
    safety = check_safety(question)
    if not safety["safe"]:
        return {
            "question": question,
            "answer": safety["answer"],
            "sources": [],
            "confidence": "N/A"
        }

    # ✅ Run the graph
    result = graph.invoke({"question": question})
    answer = result.get("answer", "No answer generated")

    # ✅ Add medical disclaimer to every answer
    answer = add_disclaimer(answer)

    return {
        "question": question,
        "answer": answer,
        "sources": result.get("sources", []),
        "confidence": "High" if result.get("sources") else "Low"
    }


@app.get("/health")
def health():
    return {"status": "ok"}