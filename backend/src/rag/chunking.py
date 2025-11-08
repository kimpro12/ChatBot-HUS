"""Chunking helpers with table-aware logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .config import Chunk, ChunkingConfig, DocumentMetadata


@dataclass(slots=True)
class TextChunker:
    """Chunk free text content using a sliding window."""

    config: ChunkingConfig

    def chunk(self, text: str, metadata: DocumentMetadata) -> List[Chunk]:
        if not text:
            return []

        words = text.split()
        size = self.config.text_chunk_size
        overlap = self.config.text_chunk_overlap
        step = max(size - overlap, 1)
        chunks: List[Chunk] = []
        for start in range(0, len(words), step):
            end = start + size
            piece = " ".join(words[start:end])
            if not piece:
                continue
            chunks.append(Chunk(text=piece, metadata=metadata.copy_with()))
        return chunks


@dataclass(slots=True)
class TableChunker:
    """Chunk tabular content row by row while repeating the header."""

    config: ChunkingConfig

    def chunk(self, markdown_table: str, metadata: DocumentMetadata) -> List[Chunk]:
        lines = [line for line in markdown_table.splitlines() if line.strip()]
        if not lines:
            return []

        header = [line for line in lines[:2]]  # header + separator
        rows = lines[2:]
        group_size = max(self.config.table_row_group_size, 1)
        chunks: List[Chunk] = []
        for index in range(0, len(rows), group_size):
            group = rows[index : index + group_size]
            table_section = "\n".join(header + group)
            chunks.append(Chunk(text=table_section, metadata=metadata.copy_with()))
        return chunks


@dataclass(slots=True)
class ChunkBuilder:
    """Compose text and table chunkers into a single helper."""

    config: ChunkingConfig

    def __post_init__(self) -> None:
        self.text_chunker = TextChunker(self.config)
        self.table_chunker = TableChunker(self.config)

    def build_text_chunks(self, text: str, metadata: DocumentMetadata) -> List[Chunk]:
        return self.text_chunker.chunk(text, metadata)

    def build_table_chunks(self, markdown_table: str, metadata: DocumentMetadata) -> List[Chunk]:
        return self.table_chunker.chunk(markdown_table, metadata.copy_with(chunk_type="table"))
