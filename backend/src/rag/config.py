"""Configuration models for the admissions chatbot RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ChunkingConfig:
    """Chunking parameters used for text and table segmentation."""

    text_chunk_size: int = 1000
    text_chunk_overlap: int = 200
    table_row_group_size: int = 40


@dataclass(slots=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str = "BAAI/bge-m3"
    normalize: bool = True
    device: Optional[str] = None


@dataclass(slots=True)
class VectorStoreConfig:
    """FAISS index configuration."""

    index_path: Path = Path("data/index.faiss")
    metadata_path: Path = Path("data/meta.json")


@dataclass(slots=True)
class PipelineConfig:
    """High level configuration for the ingestion pipeline."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


@dataclass(slots=True)
class LLMConfig:
    """Configuration for the local causal language model."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.1
    use_bitsandbytes: bool = True
    device_map: Optional[str] = "auto"


@dataclass(slots=True)
class ChatbotConfig:
    """Top-level configuration for the chatbot service."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata stored for each chunk within the vector index."""

    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    year: Optional[str] = None
    faculty: Optional[str] = None
    chunk_type: str = "text"
    table_index: Optional[int] = None

    def to_kwargs(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "year": self.year,
            "faculty": self.faculty,
            "chunk_type": self.chunk_type,
            "table_index": self.table_index,
        }

    def to_serializable(self) -> dict:
        data = self.to_kwargs()
        data["type"] = data.pop("chunk_type")
        return data

    def copy_with(self, **updates) -> "DocumentMetadata":
        data = self.to_kwargs()
        data.update(updates)
        return DocumentMetadata(**data)


@dataclass(slots=True)
class Chunk:
    """Representation of a chunk of text or a table snippet."""

    text: str
    metadata: DocumentMetadata

    def to_dict(self) -> dict:
        payload = self.metadata.to_serializable()
        payload["text"] = self.text
        return payload


@dataclass(slots=True)
class SearchResult:
    """Result returned from the vector store search."""

    score: float
    chunk: Chunk
