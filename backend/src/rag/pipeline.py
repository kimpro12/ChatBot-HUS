"""Ingestion pipeline that turns PDFs into FAISS indices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .chunking import ChunkBuilder
from .config import DocumentMetadata, PipelineConfig
from .document_parsers import (
    CamelotParser,
    CompositeParser,
    DoclingParser,
    DocumentParser,
    PyMuPDFParser,
    TabulaParser,
)
from .embedding import BGEEmbeddingModel, EmbeddingModel
from .vector_store import FaissVectorStore


@dataclass(slots=True)
class IngestionPipeline:
    """End-to-end ingestion pipeline for a single PDF file."""

    config: PipelineConfig

    def __post_init__(self) -> None:
        chunk_builder = ChunkBuilder(self.config.chunking)
        self.parsers: List[DocumentParser] = [
            DoclingParser(chunk_builder),
            PyMuPDFParser(chunk_builder),
            CamelotParser(chunk_builder),
            TabulaParser(chunk_builder),
        ]
        self.parser = CompositeParser(parsers=self.parsers)
        self.embedding_model: EmbeddingModel = BGEEmbeddingModel(self.config.embedding)
        self.vector_store = FaissVectorStore(self.config.vector_store)

    def ingest(self, pdf_path: Path, metadata: DocumentMetadata) -> None:
        chunks = self.parser.parse(pdf_path, metadata)
        vectors = self.embedding_model.embed(chunk.text for chunk in chunks)
        self.vector_store.build(vectors, chunks)

    def load_vector_store(self) -> FaissVectorStore:
        self.vector_store.load()
        return self.vector_store
