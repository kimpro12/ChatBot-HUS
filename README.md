# Chatbot Tuyển Sinh – PDF RAG with Table Awareness

This repository delivers an end-to-end admissions chatbot that ingests a single PDF prospectus and exposes the information through a retrieval-augmented interface. The backend is a FastAPI service that performs Docling-first parsing, table-aware chunking, BGE-M3 embeddings, and FAISS search. The frontend is a Next.js client that streams answers from a locally served Qwen model.

## 1. Requirements

| Component | Minimum version | Notes |
| --- | --- | --- |
| Python | 3.10 | Used by the backend ingestion + API service |
| Node.js | 18 | Required by the Next.js frontend |
| npm | 9 | For installing frontend dependencies |
| Java | 8 | Enables Tabula as a table-extraction fallback |
| Ghostscript | 9.x | Enables Camelot's lattice mode |
| GPU (optional) | ≥12 GB VRAM | Recommended for serving Qwen2.5-7B locally |

> The backend defaults to CPU-safe settings. Installing GPU drivers/accelerate tooling is only necessary when you want to serve the Qwen model with vLLM or transformers in 4-bit quantisation.

## 2. Project layout

```
ChatBot-HUS/
├── backend/    # FastAPI service, ingestion pipeline, and unit tests
├── frontend/   # Next.js application that consumes the backend API
└── Chatbot_TuyenSinh_RAG_FAISS_Tables_OpenSourceLLM.docx  # Original requirements document
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

The server exposes two endpoints:

- `POST /ingest` – accepts `{ "pdf_path": "/absolute/or/relative/path.pdf" }` and triggers the same pipeline as `ingest-pdf`.
- `POST /query` – accepts `{ "question": "...", "k": 6 }` and returns the formatted retrieval context that the frontend forwards to the LLM.

## 4. Serving the Qwen model locally

The frontend expects an OpenAI-compatible endpoint that streams responses from `Qwen/Qwen2.5-7B-Instruct`. Two common options are summarised below.

### Option A – vLLM (recommended for GPUs)

```bash
pip install "vllm>=0.5.0"
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --dtype float16 \
  --max-model-len 8192 \
  --api-key token-abc123 \
  --port 8000
```

This exposes `http://localhost:8000/v1` with token authentication. Update the API key if you need a different value.

### Option B – Transformers + bitsandbytes (CPU/GPU fallback)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    quantization_config=bnb,
)
```

Wrap the model in an OpenAI-compatible server such as [Text Generation Inference](https://github.com/huggingface/text-generation-inference) or [litellm](https://github.com/BerriAI/litellm) and point the frontend to its base URL.

## 5. Frontend setup

### 5.1 Install dependencies and configure environment variables

```bash
cd ../frontend
npm install
cp .env.example .env.local
```

Edit `.env.local` to match your environment:

```
BACKEND_URL=http://localhost:8000
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=token-abc123
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
```

- `BACKEND_URL` must reach the FastAPI server started in section 3.4.
- `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` should line up with the Qwen endpoint you launched in section 4.

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
