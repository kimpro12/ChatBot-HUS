# Chatbot Tuyển Sinh – PDF RAG with Table Awareness

This project implements a complete admissions chatbot pipeline following the provided specification. It consists of a Python backend for ingestion/retrieval and a Next.js frontend with streaming responses.

## Project Structure

- `backend/`: FastAPI service and ingestion pipeline implemented with an object-oriented architecture.
- `frontend/`: Next.js + Vercel AI SDK interface for conversational access to the admissions knowledge base.
- `Chatbot_TuyenSinh_RAG_FAISS_Tables.docx`: Original project specification.

## Getting Started

### Backend

1. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. Ingest a PDF admissions document:
   ```bash
   ingest-pdf data/quy_che_2026.pdf --source "Quy chế tuyển sinh" --year 2026
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn server:app --reload
   ```

### Frontend

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Copy `.env.example` to `.env.local` (create the file) and set:
   ```bash
   OPENAI_API_KEY=your_api_key
   BACKEND_URL=http://localhost:8000
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

The chatbot will stream answers using `gpt-4o-mini` via the Vercel AI SDK while retrieving context from the FAISS index built by the backend.

## Key Design Decisions

- **Table-first parsing** with Docling and a fallback chain of PyMuPDF → Camelot → Tabula.
- **Chunking strategy** that treats each table (or table slice) as a dedicated chunk with preserved headers.
- **Open-source embeddings** (`BAAI/bge-m3`) normalised for cosine similarity via FAISS `IndexFlatIP`.
- **Session-scoped memory** handled by the frontend client state.

Feel free to extend either side of the project to support additional PDF sources or deployment environments.
