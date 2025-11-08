from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import ChatbotConfig, DocumentMetadata, SearchResult, Chunk
from rag.service import ChatbotService


class ChatbotServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ChatbotService(ChatbotConfig())

    def test_ingest_pdf_uses_pdf_stem_as_default_source(self):
        captured = {}

        class DummyPipeline:
            def ingest(self, pdf_path, metadata):
                captured["pdf_path"] = pdf_path
                captured["metadata"] = metadata

        self.service.pipeline = DummyPipeline()  # type: ignore
        pdf_path = Path("/tmp/quy_che_2026.pdf")
        self.service.ingest_pdf(pdf_path)

        self.assertEqual(captured["pdf_path"], pdf_path)
        self.assertIsInstance(captured["metadata"], DocumentMetadata)
        self.assertEqual(captured["metadata"].source, "quy_che_2026")

    def test_format_context_includes_table_reference(self):
        chunk = Chunk(
            text="| A | B |\n| 1 | 2 |",
            metadata=DocumentMetadata(
                source="quy_che", page=3, chunk_type="table", table_index=2
            ),
        )
        result = SearchResult(score=0.9, chunk=chunk)
        formatted = self.service.format_context([result])

        self.assertIn("quy_che - Trang 3 - Bảng #2", formatted)
        self.assertIn("| A | B |", formatted)

    def test_chat_returns_local_answer(self):
        class DummyLLM:
            def __init__(self):
                self.prompts: list[str] = []

            def generate(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return "Câu trả lời"

        chunk = Chunk(
            text="Điểm chuẩn ngành CNTT là 26.",
            metadata=DocumentMetadata(source="quy_che", page=2),
        )

        class DummyEmbedding:
            def embed(self, texts):
                return [[0.0]]

        class DummyVectorStore:
            def search(self, vector, k=6):
                return [SearchResult(score=0.9, chunk=chunk)]

        self.service.llm = DummyLLM()  # type: ignore
        self.service.embedding_model = DummyEmbedding()  # type: ignore
        self.service.vector_store = DummyVectorStore()  # type: ignore

        answer, context = self.service.chat([{"role": "user", "content": "Điểm chuẩn?"}])

        self.assertEqual(answer, "Câu trả lời")
        self.assertIn("Điểm chuẩn ngành CNTT", context)
        self.assertTrue(self.service.llm.prompts)  # type: ignore
        self.assertIn("CONTEXT:\n", self.service.llm.prompts[0])  # type: ignore


if __name__ == "__main__":
    unittest.main()
