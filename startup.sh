#!/bin/bash
echo "Downloading FAISS index..."
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
import os
os.makedirs('faiss_index', exist_ok=True)
os.makedirs('data/medical_docs', exist_ok=True)
snapshot_download(repo_id='anjandata/medical-faiss-index', repo_type='dataset', local_dir='faiss_index', ignore_patterns=['*.pdf'])
hf_hub_download(repo_id='anjandata/medical-faiss-index', filename='encyclopedia_of_medicine.pdf', repo_type='dataset', local_dir='data/medical_docs')
print('Download complete!')
"
echo "Starting FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 7860