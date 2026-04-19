from fastapi import FastAPI
from agents.graph_builder import build_graph
app=FastAPI()

graph=build_graph()

@app.get("/ask")
def ask(question: str):
    result = graph.invoke({"question": question})

    return {
        "question": question,
        "answer": result.get("answer", "No answer generated"),  # ✅ SAFE
        "sources": result.get("sources", []),
        "confidence": "High" if result.get("sources") else "Low"
    }

