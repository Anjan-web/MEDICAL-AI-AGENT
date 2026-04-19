from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
import torch


INDEX_PATH = "faiss_index"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------
# STEP 1: Load all retriever components once
# Removed: duplicate load_vectorstore() and get_embeddings() functions
# Removed: old rerank_documents() keyword-counting function
# --------------------------------------------------
def load_retriever_components():
    print("Loading FAISS Index...")

    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cuda"},
    )

    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Build BM25 over all stored chunks at load time
    all_docs = list(db.docstore._dict.values())
    corpus_tokens = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(corpus_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANKER_MODEL, device=device)

    print(f"FAISS Index loaded | {len(all_docs)} chunks in BM25 | reranker on {device}")
    return db, bm25, all_docs, reranker


# --------------------------------------------------
# STEP 2: Hybrid retrieval — dense + BM25 + CrossEncoder rerank
# Fixed: typo 'dens_docs' → 'dense_docs'
# Fixed: 'dense_docs' was used before assignment in the merge loop
# --------------------------------------------------
def hybrid_retrieve(query, db, bm25, all_docs, reranker, k=15, final_k=3):

    # Dense semantic retrieval
    dense_docs = db.similarity_search(query, k=k)

    # BM25 sparse retrieval
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_docs = [all_docs[i] for i in top_indices]

    # Merge and deduplicate using first 80 chars as fingerprint
    seen = set()
    candidates = []
    for doc in dense_docs + bm25_docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    # Neural reranking with CrossEncoder
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in ranked[:final_k]]


# --------------------------------------------------
# STEP 3: Public retriever — loads once, returns a callable
# Fixed: old get_retriever() called the removed load_vectorstore()
# Fixed: old get_retriever() ran keyword reranking instead of CrossEncoder
# --------------------------------------------------
def get_retriever():
    db, bm25, all_docs, reranker = load_retriever_components()

    def retrieve(query: str):
        return hybrid_retrieve(query, db, bm25, all_docs, reranker)

    return retrieve