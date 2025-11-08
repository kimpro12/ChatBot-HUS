"""FastAPI server exposing ingestion and query endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import DocumentMetadata, PipelineConfig
from rag.service import ChatbotService

app = FastAPI(title="Admissions Chatbot API", version="1.0.0")
service = ChatbotService(PipelineConfig())


class IngestRequest(BaseModel):
    pdf_path: str
    source: str = "Quy chế tuyển sinh"
    year: str | None = None
    faculty: str | None = None


class QueryRequest(BaseModel):
    question: str
    k: int = 6


class QueryResponse(BaseModel):
    answer_context: str


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    metadata = DocumentMetadata(
        source=request.source,
        year=request.year,
        faculty=request.faculty,
    )
    service.ingest_pdf(pdf_path, metadata)
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        service.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = service.search(request.question, k=request.k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found")
    context = service.format_context(results)
    return QueryResponse(answer_context=context)
