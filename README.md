# Chatbot Tuyển Sinh – PDF RAG with Table Awareness

This repository delivers an end-to-end admissions chatbot that ingests a single PDF prospectus and exposes the information through a retrieval-augmented interface. The backend is a FastAPI service that performs Docling-first parsing, table-aware chunking, BGE-M3 embeddings, FAISS search, and local response generation via Hugging Face Transformers. The frontend is a Next.js client that proxies chat requests to the backend and renders the responses.

## 1. Requirements

| Component | Minimum version | Notes |
| --- | --- | --- |
| Python | 3.10 | Used by the backend ingestion + API service |
| Node.js | 18 | Required by the Next.js frontend |
| npm | 9 | For installing frontend dependencies |
| Java | 8 | Enables Tabula as a table-extraction fallback |
| Ghostscript | 9.x | Enables Camelot's lattice mode |
| GPU (optional) | ≥12 GB VRAM | Recommended for faster Qwen2.5-7B generation |

> The backend loads Qwen2.5-7B through `transformers` with an automatic CPU/GPU fallback. Installing NVIDIA drivers and enabling bitsandbytes quantisation dramatically improves latency but is not strictly required.

## 2. Project layout

```
ChatBot-HUS/
├── backend/    # FastAPI service, ingestion pipeline, and unit tests
├── frontend/   # Next.js application that consumes the backend API
└── Chatbot_TuyenSinh_RAG_FAISS_Tables.docx  # Original requirements document
```

## 3. Backend quickstart

### 3.1 Create a virtual environment

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install exposes the helper commands `ingest-pdf`, `query-chatbot`, and `uvicorn rag.server:app` for the FastAPI server.

### 3.2 Ingest your admissions PDF

Place the PDF anywhere on your filesystem (for example `data/quy_che_2026.pdf`) and run:

```bash
ingest-pdf data/quy_che_2026.pdf
```

The command performs the following actions:

1. **Parse and normalise the document** – Docling extracts text and tables; PyMuPDF, Camelot, and Tabula are used as fallbacks when a table is missed.
2. **Chunk the content** – Regular prose is chunked at ~1,000 tokens with 200-token overlap. Each table (or table slice) becomes its own chunk with headers repeated for long tables.
3. **Embed the chunks** – `BAAI/bge-m3` generates cosine-normalised vectors.
4. **Persist the index** – A FAISS `IndexFlatIP` index and a JSON metadata file are written under `backend/data/` by default.

You can adjust the storage paths or chunk sizes by editing `backend/src/rag/config.py`.

### 3.3 Run automated tests (optional)

```bash
python -m unittest discover tests
```

The suite covers the ingestion pipeline, FastAPI routes, and service-level helpers so you can verify that the environment is configured correctly.

### 3.4 Start the FastAPI server

```bash
uvicorn rag.server:app --reload
```

The server exposes three endpoints:

- `POST /ingest` – accepts `{ "pdf_path": "/absolute/or/relative/path.pdf" }` and triggers the same pipeline as `ingest-pdf`.
- `POST /query` – accepts `{ "question": "...", "k": 6 }` and returns the formatted retrieval context (useful for debugging the retriever).
- `POST /chat` – accepts `{ "messages": [{ "role": "user" | "assistant", "content": "..." }], "k": 6 }`, performs retrieval, builds a prompt, and generates a response with the local Qwen model.

## 4. Local Qwen generation with Transformers

The FastAPI backend now owns the language-model layer. When the first `/chat` request arrives it lazily loads `Qwen/Qwen2.5-7B-Instruct` via Hugging Face Transformers. If an NVIDIA GPU is detected and `bitsandbytes` is available, the model is loaded in 4-bit NF4 format; otherwise it falls back to full-precision CPU execution.

### 4.1 GPU acceleration (optional)

1. Install the latest CUDA-enabled PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).
2. Ensure the `nvidia-smi` command reports the GPU you plan to use.
3. Keep the default `use_bitsandbytes=True` setting in `backend/src/rag/config.py` to enable 4-bit quantisation.

### 4.2 CPU-only environments

No additional steps are required. The backend automatically sets `device_map={"": "cpu"}` and generates answers in smaller batches. Expect slower responses for large outputs; consider reducing `LLMConfig.max_new_tokens` if latency is a concern.

### 4.3 Verifying the load

Start the FastAPI server and send a sample chat request:

```bash
uvicorn rag.server:app --reload

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Điểm chuẩn CNTT?"}]}'
```

The first call may take ~30–60 seconds while the model weights download and load into memory. Subsequent requests reuse the cached model instance.

## 5. Frontend setup

### 5.1 Install dependencies and configure environment variables

```bash
cd ../frontend
npm install
cp .env.example .env.local
```

Edit `.env.local` to point the frontend at the FastAPI service:

```
BACKEND_URL=http://localhost:8000
```

The frontend now delegates both retrieval and generation to the backend, so no additional LLM variables are required.

### 5.2 Launch the development server

```bash
npm run dev
```

Navigate to `http://localhost:3000` to interact with the chatbot. The UI keeps the conversation in client state, so refreshing the page clears the chat history and forces a new retrieval + generation cycle.

## 6. End-to-end smoke test

Once everything is running:

1. Upload/ingest a PDF (`ingest-pdf data/quy_che_2026.pdf`).
2. Send a sample query via CLI: `query-chatbot "Điểm chuẩn ngành Công nghệ thông tin là bao nhiêu?"`.
3. Confirm the frontend displays the same retrieved snippets and cites the table/page numbers referenced in the PDF.

If the CLI or API indicates that no relevant context was found, ensure the FAISS files exist under `backend/data/` and that the PDF contains searchable text (scanned images require OCR before ingestion).

## 7. Customisation tips

- **Multiple PDFs**: Ingest each file sequentially; the FAISS index aggregates all chunks in insertion order.
- **Chunk sizes**: Tweak `ChunkingConfig` in `backend/src/rag/config.py` if your PDF contains extremely long tables or dense prose.
- **Alternative embeddings**: Swap `BAAI/bge-m3` for another sentence-transformer by updating `EmbeddingConfig.model_name`.
- **Deployment**: Containerise the backend and frontend separately. Provide the FAISS index and metadata files as persistent volumes so you do not need to re-ingest on every restart.

For further details about each backend component, refer to `backend/README.md`.
