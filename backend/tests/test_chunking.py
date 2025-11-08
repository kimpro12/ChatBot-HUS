import sys
from pathlib import Path
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import ChunkBuilder
from rag.config import ChunkingConfig, DocumentMetadata


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChunkingConfig(text_chunk_size=5, text_chunk_overlap=2, table_row_group_size=2)
        self.builder = ChunkBuilder(self.config)
        self.base_metadata = DocumentMetadata(source="quy_che", page=1)

    def test_text_chunking_respects_overlap(self):
        text = "một hai ba bốn năm sáu bảy"
        chunks = self.builder.build_text_chunks(text, self.base_metadata)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk.metadata.chunk_type, "text")
            self.assertEqual(chunk.metadata.source, "quy_che")

    def test_table_chunking_repeats_header_and_marks_table_type(self):
        markdown_table = """| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"""
        metadata = self.base_metadata.copy_with(chunk_type="table", table_index=1)
        chunks = self.builder.build_table_chunks(markdown_table, metadata)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertIn("| A | B |", chunk.text)
            self.assertEqual(chunk.metadata.chunk_type, "table")
            self.assertEqual(chunk.metadata.table_index, 1)


if __name__ == "__main__":
    unittest.main()
