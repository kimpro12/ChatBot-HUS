# Backend – Admissions Chatbot

The backend packages the PDF ingestion pipeline, FAISS vector store, Hugging Face-powered language model, and FastAPI server for the admissions chatbot. Everything is implemented with small, composable classes so that you can swap parsers, embeddings, or storage without rewriting the application.

## 1. Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Installing the project in editable mode registers three CLI entrypoints:

- `ingest-pdf` – run the full ingestion pipeline on a single PDF.
- `query-chatbot` – run a retrieval against the stored FAISS index from the command line.
- `uvicorn rag.server:app` – start the FastAPI API (equivalent to `run-server` defined in `pyproject.toml`).

> Optional system dependencies: Camelot's lattice mode requires Ghostscript and Tabula requires a Java runtime. Without them the Docling and PyMuPDF paths still work, but fewer tables may be captured.

## 2. Configuration overview

All configuration lives in [`rag/config.py`](src/rag/config.py):

- `ChunkingConfig` – controls text chunk size, overlap, and the maximum number of table rows per slice.
- `EmbeddingConfig` – selects the sentence-transformers model (`BAAI/bge-m3` by default) and device placement.
- `VectorStoreConfig` – sets the FAISS index and metadata file locations (defaults to `data/index.faiss` and `data/meta.json`).
- `LLMConfig` – defines the Hugging Face causal LM (`Qwen/Qwen2.5-7B-Instruct` by default), generation parameters, and whether bitsandbytes quantisation should be attempted.
- `ChatbotConfig` – bundles the pipeline + LLM settings passed into `ChatbotService`.

Adjust these values before ingestion if you want to save the index elsewhere or experiment with different chunk sizes.

## 3. Ingestion workflow

1. Prepare your PDF (must contain extractable text; run OCR if it is scanned).
2. Execute:
   ```bash
   ingest-pdf path/to/your_document.pdf
   ```
3. The pipeline performs:
   - Docling conversion to Markdown with table exports.
   - Fallback table detection using PyMuPDF, Camelot, and Tabula.
   - Table-aware chunking where each table (or slice) carries its header.
   - BGE-M3 embedding and FAISS `IndexFlatIP` persistence.
4. Output files are written to the paths defined in `VectorStoreConfig`.

To re-ingest, run the command again—the previous index and metadata files are overwritten.

## 4. Querying the index

### 4.1 CLI query helper

```bash
query-chatbot "Điểm chuẩn ngành Công nghệ thông tin là bao nhiêu?"
```

The command prints the top-k chunks along with their metadata so you can inspect the retrieved evidence manually.

### 4.2 FastAPI endpoints

Start the API server with:

```bash
uvicorn rag.server:app --reload
```

Endpoints:

- `POST /ingest`
  ```json
  {
    "pdf_path": "data/quy_che_2026.pdf"
  }
  ```
  Validates that the file exists and then runs the same ingestion pipeline as the CLI.

- `POST /query`
  ```json
  {
    "question": "Các ngành thuộc khối Kỹ thuật là gì?",
    "k": 6
  }
  ```
  Returns the formatted context string used by the generator. HTTP 404 is returned if no vectors match the query.

- `POST /chat`
  ```json
  {
    "messages": [
      { "role": "assistant", "content": "Xin chào!" },
      { "role": "user", "content": "Điểm chuẩn CNTT năm 2024?" }
    ],
    "k": 6
  }
  ```
  Performs retrieval, builds a prompt, and generates a reply with the locally loaded Qwen model. The response payload contains both the answer and the retrieved context for debugging.

The service automatically loads the FAISS files on demand. If they are missing, the request fails with a `500` error indicating that ingestion must be executed first.

## 5. Running tests

```bash
python -m unittest discover tests
```

The test suite validates:

- Chunking logic for both prose and tables (`tests/test_chunking.py`).
- FastAPI routes including error handling (`tests/test_server.py`).
- The high-level service that coordinates ingestion, embedding, and search (`tests/test_service.py`).

Running the tests after installation is the quickest way to confirm that optional dependencies (Docling, PyMuPDF, FAISS) are importable in your environment.

## 6. Troubleshooting

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| `PDF file not found` when calling `/ingest` | Relative path resolved from API process | Pass an absolute path or run the API from the project root |
| `/query` returns 500 with `index.faiss` missing | Ingestion was not run | Execute `ingest-pdf ...` or call `/ingest` first |
| Tables missing from retrieved context | Camelot/Tabula not installed or PDF is scanned | Install Ghostscript + Java, or convert the PDF to text/OCR before ingestion |
| Slow embeddings on CPU | BGE-M3 is large | Install the model once to cache weights, or switch to a smaller embedding model via `EmbeddingConfig` |

With the backend running and indexed, you can point the Next.js frontend (see the root `README.md`) to `http://localhost:8000` for both retrieval and generation.
