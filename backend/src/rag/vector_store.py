"""FAISS vector store wrapper."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Sequence

import faiss
import numpy as np

from .config import Chunk, DocumentMetadata, SearchResult, VectorStoreConfig


@dataclass(slots=True)
class FaissVectorStore:
    """Wrapper around a FAISS IndexFlatIP with metadata persistence."""

    config: VectorStoreConfig

    def __post_init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: List[dict] = []
        self._texts: List[str] = []

    # region persistence helpers -------------------------------------------------
    def _ensure_storage(self) -> None:
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # endregion -----------------------------------------------------------------

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index is not initialized. Call build() first.")
        return self._index

    def build(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array")
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._metadata = [chunk.metadata.to_serializable() for chunk in chunks]
        self._texts = [chunk.text for chunk in chunks]
        self._ensure_storage()
        faiss.write_index(index, str(self.config.index_path))
        payload = [dict(meta, text=text) for meta, text in zip(self._metadata, self._texts)]
        self.config.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        if not self.config.index_path.exists():
            raise FileNotFoundError("FAISS index file not found. Have you run the ingestion pipeline?")
        self._index = faiss.read_index(str(self.config.index_path))
        payload = json.loads(self.config.metadata_path.read_text(encoding="utf-8"))
        self._metadata = [{k: v for k, v in item.items() if k != "text"} for item in payload]
        self._texts = [item["text"] for item in payload]

    def search(self, query_vector: np.ndarray, k: int = 6) -> List[SearchResult]:
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded")
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = self._index.search(query, k)
        results: List[SearchResult] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            metadata = self._metadata[idx]
            metadata_kwargs = {
                "source": metadata.get("source", ""),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "year": metadata.get("year"),
                "faculty": metadata.get("faculty"),
                "chunk_type": metadata.get("type", "text"),
                "table_index": metadata.get("table_index"),
            }
            chunk = Chunk(
                text=self._texts[idx],
                metadata=DocumentMetadata(**metadata_kwargs),
            )
            results.append(SearchResult(score=float(distance), chunk=chunk))
        return results
