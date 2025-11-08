"""High level chatbot service."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import DocumentMetadata, PipelineConfig, SearchResult
from .embedding import BGEEmbeddingModel, EmbeddingModel
from .pipeline import IngestionPipeline
from .vector_store import FaissVectorStore


@dataclass(slots=True)
class ChatbotService:
    """Coordinate ingestion and retrieval for the admissions chatbot."""

    config: PipelineConfig

    def __post_init__(self) -> None:
        self.pipeline = IngestionPipeline(self.config)
        self.embedding_model: EmbeddingModel = BGEEmbeddingModel(self.config.embedding)
        self.vector_store: FaissVectorStore = self.pipeline.vector_store

    def ingest_pdf(self, pdf_path: str | Path, metadata: DocumentMetadata) -> None:
        self.pipeline.ingest(pdf_path=Path(pdf_path), metadata=metadata)

    def load(self) -> None:
        self.vector_store.load()

    def search(self, query: str, k: int = 6) -> List[SearchResult]:
        vector = self.embedding_model.embed([query])
        return self.vector_store.search(vector, k=k)

    def format_context(self, results: Iterable[SearchResult]) -> str:
        sections: List[str] = []
        for result in results:
            meta = result.chunk.metadata
            if meta is None:
                continue
            citation = f"{meta.source} - Trang {meta.page or '?'}"
            if meta.chunk_type == "table" and meta.table_index:
                citation += f" - Bảng #{meta.table_index}"
            sections.append(f"[{citation}]\n{result.chunk.text}\n")
        return "\n".join(sections)
