# Backend – Admissions Chatbot

This backend implements the ingestion and retrieval pipeline for the admissions chatbot as described in the project specification. It is organised using an object-oriented design so that the PDF processing, embedding, and vector search layers can be extended easily.

## Features

- **Table-aware ingestion**: Docling is used as the primary parser with fallbacks to PyMuPDF, Camelot, and Tabula.
- **Chunking strategy**: Text is chunked with overlap, while each table (or table slice) becomes an individual chunk.
- **Embedding**: Uses the open-source `BAAI/bge-m3` sentence-transformers model by default.
- **Vector store**: Local FAISS `IndexFlatIP` index with cosine similarity via L2 normalisation.
- **FastAPI service**: Exposes `/ingest` and `/query` endpoints that can be consumed by the Next.js frontend.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> Camelot (lattice mode) requires Ghostscript and Tabula requires Java. Install them if you want to enable the full fallback chain.

## Usage

### Ingest a PDF

```bash
ingest-pdf data/quy_che_2026.pdf
```

This command stores `index.faiss` and `meta.json` under `backend/data/` (configurable in `rag/config.py`).

### Query from the command line

```bash
query-chatbot "Điểm chuẩn ngành Công nghệ thông tin là bao nhiêu?"
```

### Run the API server

```bash
uvicorn server:app --reload
```

The FastAPI service is required when using the Next.js client.
