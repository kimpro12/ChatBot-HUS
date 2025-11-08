"""Document parsing and table extraction utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import Chunk, DocumentMetadata
from .chunking import ChunkBuilder


class DocumentParsingError(RuntimeError):
    """Raised when a parser fails to process a document."""


@dataclass(slots=True)
class DocumentParser(ABC):
    """Base class for document parsers."""

    chunk_builder: ChunkBuilder

    @abstractmethod
    def parse(self, path: Path, base_metadata: DocumentMetadata) -> List[Chunk]:
        """Parse a document and return processed chunks."""


@dataclass(slots=True)
class DoclingParser(DocumentParser):
    """Parse documents using Docling with structured table extraction."""

    def parse(self, path: Path, base_metadata: DocumentMetadata) -> List[Chunk]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DocumentParsingError("docling is not installed") from exc

        converter = DocumentConverter()
        conversion = converter.convert(str(path))
        document = conversion.document

        chunks: List[Chunk] = []

        if document is None:
            raise DocumentParsingError("Docling returned an empty document")

        markdown_text = document.export_to_markdown()
        text_chunks = self.chunk_builder.build_text_chunks(
            markdown_text,
            base_metadata.copy_with(),
        )
        chunks.extend(text_chunks)

        for index, table in enumerate(document.tables, start=1):
            metadata = base_metadata.copy_with(chunk_type="table", table_index=index)
            try:
                dataframe = table.export_to_dataframe()
                markdown_table = dataframe.to_markdown(index=False)
            except Exception:  # pragma: no cover - docling internals
                markdown_table = table.export_to_markdown()
            chunks.extend(self.chunk_builder.build_table_chunks(markdown_table, metadata))

        return chunks


@dataclass(slots=True)
class PyMuPDFParser(DocumentParser):
    """Fallback parser using PyMuPDF's built-in table detection."""

    def parse(self, path: Path, base_metadata: DocumentMetadata) -> List[Chunk]:
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DocumentParsingError("PyMuPDF is not installed") from exc

        document = fitz.open(str(path))
        chunks: List[Chunk] = []

        full_markdown: List[str] = []
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("markdown")
            metadata = base_metadata.copy_with(page=page_number)
            full_markdown.append(text)
            tables = page.find_tables()
            if tables:
                for table_index, table in enumerate(tables.tables, start=1):
                    table_metadata = metadata.copy_with(
                        chunk_type="table",
                        table_index=table_index,
                    )
                    markdown_table = table.to_markdown()
                    chunks.extend(
                        self.chunk_builder.build_table_chunks(markdown_table, table_metadata)
                    )
        document.close()

        # Add combined text chunk at the end to avoid duplicates for each page
        text_metadata = base_metadata.copy_with()
        chunks.extend(
            self.chunk_builder.build_text_chunks("\n".join(full_markdown), text_metadata)
        )
        return chunks


@dataclass(slots=True)
class CamelotParser(DocumentParser):
    """Camelot-based parser for table heavy PDFs."""

    flavor_priority: Sequence[str] = ("lattice", "stream")

    def parse(self, path: Path, base_metadata: DocumentMetadata) -> List[Chunk]:
        try:
            import camelot
        except ImportError as exc:  # pragma: no cover
            raise DocumentParsingError("camelot is not installed") from exc

        chunks: List[Chunk] = []
        combined_tables: List[str] = []
        for flavor in self.flavor_priority:
            try:
                tables = camelot.read_pdf(str(path), pages="all", flavor=flavor)
            except Exception:
                continue
            if not tables:
                continue
            for idx, table in enumerate(tables, start=1):
                metadata = base_metadata.copy_with(chunk_type="table", table_index=idx)
                markdown_table = table.df.to_markdown(index=False)
                chunks.extend(self.chunk_builder.build_table_chunks(markdown_table, metadata))
                combined_tables.append(markdown_table)
            break

        if combined_tables:
            text_metadata = base_metadata.copy_with()
            chunks.extend(
                self.chunk_builder.build_text_chunks("\n".join(combined_tables), text_metadata)
            )
        else:
            raise DocumentParsingError("Camelot could not find tables")
        return chunks


@dataclass(slots=True)
class TabulaParser(DocumentParser):
    """Tabula-based table extractor."""

    def parse(self, path: Path, base_metadata: DocumentMetadata) -> List[Chunk]:
        try:
            import tabula
        except ImportError as exc:  # pragma: no cover
            raise DocumentParsingError("tabula-py is not installed") from exc

        dataframes = tabula.read_pdf(str(path), pages="all", multiple_tables=True)
        if not dataframes:
            raise DocumentParsingError("Tabula returned no tables")

        chunks: List[Chunk] = []
        for idx, dataframe in enumerate(dataframes, start=1):
            metadata = base_metadata.copy_with(chunk_type="table", table_index=idx)
            markdown_table = dataframe.to_markdown(index=False)
            chunks.extend(self.chunk_builder.build_table_chunks(markdown_table, metadata))
        return chunks


@dataclass(slots=True)
class CompositeParser(DocumentParser):
    """Try multiple parsers until one succeeds."""

    parsers: Sequence[DocumentParser]

    def parse(self, path: Path, base_metadata: DocumentMetadata) -> List[Chunk]:
        last_error: Optional[Exception] = None
        for parser in self.parsers:
            try:
                return parser.parse(path, base_metadata)
            except DocumentParsingError as exc:
                last_error = exc
                continue
        raise DocumentParsingError(str(last_error) if last_error else "No parser succeeded")
