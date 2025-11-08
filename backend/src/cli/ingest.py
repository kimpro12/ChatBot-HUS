"""CLI entry point to ingest a PDF into the FAISS index."""
from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import PipelineConfig
from rag.service import ChatbotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a PDF admissions document")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ChatbotService(PipelineConfig())
    service.ingest_pdf(args.pdf)
    print("Ingestion completed. Index stored at", service.vector_store.config.index_path)


if __name__ == "__main__":
    main()
