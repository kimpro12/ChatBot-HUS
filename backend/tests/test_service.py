import sys
from pathlib import Path
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DocumentMetadata, PipelineConfig, SearchResult, Chunk
from rag.service import ChatbotService


class ChatbotServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ChatbotService(PipelineConfig())

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


if __name__ == "__main__":
    unittest.main()
