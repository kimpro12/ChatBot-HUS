"""CLI entry point to ingest a PDF into the FAISS index."""
from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import DocumentMetadata, PipelineConfig
from rag.service import ChatbotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a PDF admissions document")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--source", default="Quy chế tuyển sinh", help="Tên tài liệu")
    parser.add_argument("--year", default=None, help="Năm ban hành")
    parser.add_argument("--faculty", default=None, help="Khoa / chương trình")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = DocumentMetadata(
        source=args.source,
        year=args.year,
        faculty=args.faculty,
    )
    service = ChatbotService(PipelineConfig())
    service.ingest_pdf(args.pdf, metadata)
    print("Ingestion completed. Index stored at", service.vector_store.config.index_path)


if __name__ == "__main__":
    main()
