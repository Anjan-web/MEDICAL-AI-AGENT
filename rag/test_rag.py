from rag.retriever import get_retriever

retriever = get_retriever()

query = "What is malaria?"

# ✅ CORRECT METHOD
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Page: {doc.metadata.get('page')}")
    print(doc.page_content[:300])