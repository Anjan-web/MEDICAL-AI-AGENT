from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
import torch


INDEX_PATH = "faiss_index"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_retriever_components():
    print("Loading FAISS Index...")

    # ✅ FIXED: auto-detect device instead of hardcoding "cuda"
    # HuggingFace Spaces has no GPU — was crashing with RuntimeError
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    all_docs = list(db.docstore._dict.values())
    corpus_tokens = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(corpus_tokens)

    reranker = CrossEncoder(RERANKER_MODEL, device=device)

    print(f"FAISS Index loaded | {len(all_docs)} chunks in BM25 | reranker on {device}")
    return db, bm25, all_docs, reranker


def hybrid_retrieve(query, db, bm25, all_docs, reranker, k=15, final_k=3):

    dense_docs = db.similarity_search(query, k=k)

    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_docs = [all_docs[i] for i in top_indices]

    seen = set()
    candidates = []
    for doc in dense_docs + bm25_docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in ranked[:final_k]]


def get_retriever():
    db, bm25, all_docs, reranker = load_retriever_components()

    def retrieve(query: str):
        return hybrid_retrieve(query, db, bm25, all_docs, reranker)

    return retrieve