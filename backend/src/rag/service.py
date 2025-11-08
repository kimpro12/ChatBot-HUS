"""High level chatbot service."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from .config import ChatbotConfig, DocumentMetadata, SearchResult
from .embedding import EmbeddingModel
from .pipeline import IngestionPipeline
from .vector_store import FaissVectorStore
from .llm import LocalCausalLM, format_chat_prompt


@dataclass(slots=True)
class ChatbotService:
    """Coordinate ingestion and retrieval for the admissions chatbot."""

    config: ChatbotConfig
    pipeline: IngestionPipeline = field(init=False)
    embedding_model: EmbeddingModel = field(init=False)
    vector_store: FaissVectorStore = field(init=False)
    llm: LocalCausalLM = field(init=False)

    def __post_init__(self) -> None:
        pipeline_config = self.config.pipeline
        self.pipeline = IngestionPipeline(pipeline_config)
        self.embedding_model = self.pipeline.embedding_model
        self.vector_store = self.pipeline.vector_store
        self.llm = LocalCausalLM(self.config.llm)

    def ingest_pdf(
        self, pdf_path: str | Path, metadata: DocumentMetadata | None = None
    ) -> None:
        pdf_path = Path(pdf_path)
        metadata = metadata or DocumentMetadata(source=pdf_path.stem)
        self.pipeline.ingest(pdf_path=pdf_path, metadata=metadata)

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

    def chat(self, messages: List[dict], k: int = 6) -> tuple[str, str]:
        question = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                question = message.get("content", "")
                break

        if not question:
            raise ValueError("No user question provided")

        results = self.search(question, k=k)
        if not results:
            raise LookupError("No relevant context found")

        context = self.format_context(results)
        prompt = format_chat_prompt(messages, context, question)
        answer = self.llm.generate(prompt)
        return answer, context
