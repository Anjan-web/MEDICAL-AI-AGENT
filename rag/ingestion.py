import os
import fitz
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch


PDF_PATH = "data/medical_docs/encyclopedia_of_medicine.pdf"
INDEX_PATH = "faiss_index"


# --------------------------------------------------
# 🔹 STEP 1: Extract PDF
# --------------------------------------------------
def load_pdf():
    documents = []

    docs = fitz.open(PDF_PATH)

    for page_num in range(len(docs)):
        page = docs[page_num]
        text = page.get_text()

        if not text or len(text.strip()) < 50:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "encyclopedia_of_medicine",
                    "page": page_num + 1
                }
            )
        )

    print(f"✅ Loaded {len(documents)} pages")
    return documents


# --------------------------------------------------
# 🔥 STEP 2: STRONG CLEANING (VERY IMPORTANT)
# --------------------------------------------------
def clean_text(text):
    import re

    # ✅ NEW: strip encyclopedia author/fax header lines
    text = re.sub(r"Fax:.*?\.gov\>?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Z][a-z]+\s[A-Z]\.?\s[A-Z][a-z]+,\s?(RN|PhD|MD|DO|MPH)", "", text)  # author names with credentials

    # existing patterns
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"G A L E.*?M E D I C I N E", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)[^a-zA-Z0-9\s](?!\w)", " ", text)
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    cleaned_chunks = []
    seen_hashes = set()   # ✅ NEW: deduplication

    for doc in documents:
        cleaned_text = clean_text(doc.page_content)

        if "." not in cleaned_text:
            continue

        lines = cleaned_text.strip().split("\n")
        heading = lines[0].strip() if lines and len(lines[0]) < 80 else ""

        chunks = splitter.create_documents([cleaned_text], metadatas=[doc.metadata])

        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()

            if len(text) < 100 or len(text.split()) < 20:
                continue

            # ✅ NEW: skip duplicate chunks using content fingerprint
            fingerprint = text[:120].lower()
            if fingerprint in seen_hashes:
                continue
            seen_hashes.add(fingerprint)

            chunk.metadata.update({
                "chunk_index": i,
                "word_count": len(text.split()),
                "section_heading": heading,
                "chunk_id": f"{doc.metadata['page']}_{i}"
            })

            cleaned_chunks.append(chunk)

    print(f"✅ Created {len(cleaned_chunks)} clean chunks")
    return cleaned_chunks

# --------------------------------------------------
# 🔥 STEP 4: BUILD FAISS INDEX
# --------------------------------------------------
def build_index():
    print("📄 Loading PDF...")
    docs = load_pdf()

    print("✂️ Splitting & cleaning...")
    chunks = split_documents(docs)

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True,
                       "batch_size": 128
                       }
    )

    print(f"📦 Building FAISS index for {len(chunks)} chunks")

    BATCH_SIZE = 1000
    db= None

    for i in range(0,len(chunks),BATCH_SIZE):
        batch=chunks[i:i+BATCH_SIZE]
        print(f"  → Indexing chunks {i}–{i+len(batch)} / {len(chunks)}")
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    db.save_local(INDEX_PATH)

    print("🚀 FAISS Index Created Successfully!")


# --------------------------------------------------
# 🔹 MAIN
# --------------------------------------------------
if __name__ == "__main__":
    build_index()