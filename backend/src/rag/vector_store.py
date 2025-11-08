"""FAISS vector store wrapper."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import List, Sequence, Any

try:  # pragma: no cover - import guard for optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - handled lazily
    faiss = None  # type: ignore

try:  # pragma: no cover - import guard for optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - handled lazily
    np = None  # type: ignore

from .config import Chunk, DocumentMetadata, SearchResult, VectorStoreConfig


@dataclass(slots=True)
class FaissVectorStore:
    """Wrapper around a FAISS IndexFlatIP with metadata persistence."""

    config: VectorStoreConfig
    _index: faiss.Index | None = field(init=False, default=None)
    _metadata: List[dict] = field(init=False, default_factory=list)
    _texts: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # Attributes initialized via dataclass defaults above; method kept for compatibility.
        pass

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
        faiss_module = self._require_faiss()
        self._require_numpy()
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array")
        faiss_module.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss_module.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._metadata = [chunk.metadata.to_serializable() for chunk in chunks]
        self._texts = [chunk.text for chunk in chunks]
        self._ensure_storage()
        faiss_module.write_index(index, str(self.config.index_path))
        payload = [dict(meta, text=text) for meta, text in zip(self._metadata, self._texts)]
        self.config.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        faiss_module = self._require_faiss()
        if not self.config.index_path.exists():
            raise FileNotFoundError("FAISS index file not found. Have you run the ingestion pipeline?")
        self._index = faiss_module.read_index(str(self.config.index_path))
        payload = json.loads(self.config.metadata_path.read_text(encoding="utf-8"))
        self._metadata = [{k: v for k, v in item.items() if k != "text"} for item in payload]
        self._texts = [item["text"] for item in payload]

    def search(self, query_vector: np.ndarray, k: int = 6) -> List[SearchResult]:
        faiss_module = self._require_faiss()
        np_module = self._require_numpy()
        if self._index is None:
            raise RuntimeError("FAISS index is not loaded")
        query = np_module.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss_module.normalize_L2(query)
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

    @staticmethod
    def _require_faiss() -> Any:
        if faiss is None:
            raise RuntimeError("faiss is required for vector store operations. Please install faiss-cpu.")
        return faiss

    @staticmethod
    def _require_numpy() -> Any:
        if np is None:
            raise RuntimeError("numpy is required for vector store operations. Please install numpy.")
        return np
