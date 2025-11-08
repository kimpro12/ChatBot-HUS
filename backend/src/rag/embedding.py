"""Embedding providers for the chatbot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .config import EmbeddingConfig


@dataclass(slots=True)
class EmbeddingModel:
    """Base embedding model interface."""

    config: EmbeddingConfig

    def embed(self, texts: Iterable[str]) -> np.ndarray:  # pragma: no cover - base class
        raise NotImplementedError


@dataclass(slots=True)
class BGEEmbeddingModel(EmbeddingModel):
    """Embedding model backed by sentence-transformers BGE-M3."""

    _model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(list(texts), normalize_embeddings=self.config.normalize)
        return np.asarray(vectors, dtype="float32")


@dataclass(slots=True)
class OpenAIEmbeddingModel(EmbeddingModel):
    """Optional OpenAI embedding provider for higher quality."""

    model: str = "text-embedding-3-large"

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAI embeddings") from exc

        client = OpenAI()
        embeddings: List[List[float]] = []
        for text in texts:
            response = client.embeddings.create(model=self.model, input=text)  # type: ignore
            embeddings.append(response.data[0].embedding)
        vectors = np.asarray(embeddings, dtype="float32")
        if self.config.normalize:
            from faiss import normalize_L2  # type: ignore

            normalize_L2(vectors)
        return vectors
