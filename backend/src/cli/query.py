"""CLI entry point to query the chatbot."""
from __future__ import annotations

import argparse

from rag.config import PipelineConfig
from rag.service import ChatbotService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the admissions chatbot")
    parser.add_argument("question", help="Câu hỏi")
    parser.add_argument("--k", type=int, default=6, help="Số chunk truy hồi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ChatbotService(PipelineConfig())
    service.load()
    results = service.search(args.question, k=args.k)
    if not results:
        print("Không tìm thấy thông tin phù hợp.")
        return
    context = service.format_context(results)
    print(context)


if __name__ == "__main__":
    main()
