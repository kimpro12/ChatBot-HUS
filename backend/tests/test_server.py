from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import HTTPException

from rag import server


class DummyService:
    def __init__(self):
        self.ingest_calls: list[Path] = []
        self.load_calls: int = 0
        self.search_calls: list[tuple[str, int]] = []
        self.chat_payloads: list[tuple[list[dict], int]] = []

    def ingest_pdf(self, pdf_path, metadata=None):
        self.ingest_calls.append(Path(pdf_path))

    def load(self):
        self.load_calls += 1

    def search(self, question: str, k: int = 6):
        self.search_calls.append((question, k))
        return []

    def format_context(self, results):
        return ""

    def chat(self, messages, k=6):
        self.chat_payloads.append((messages, k))
        raise LookupError("No relevant context found")


def with_dummy_service(testcase):
    original_service = server.service
    dummy = DummyService()
    server.service = dummy  # type: ignore
    try:
        testcase(dummy)
    finally:
        server.service = original_service  # type: ignore


class ServerTests(unittest.TestCase):
    def test_ingest_uses_default_metadata(self):
        def run(dummy: DummyService):
            with TemporaryDirectory() as tmp:
                pdf_path = Path(tmp) / "doc.pdf"
                pdf_path.write_bytes(b"fake")

                request = server.IngestRequest(pdf_path=str(pdf_path))
                response = server.ingest(request)

                self.assertEqual(response, {"status": "ok"})
                self.assertEqual(dummy.ingest_calls, [pdf_path])

        with_dummy_service(run)

    def test_query_raises_404_when_no_results(self):
        def run(dummy: DummyService):
            request = server.QueryRequest(question="Hi?", k=2)

            with self.assertRaises(HTTPException) as ctx:
                server.query(request)

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertEqual(dummy.search_calls, [("Hi?", 2)])
            self.assertEqual(dummy.load_calls, 1)

        with_dummy_service(run)

    def test_chat_propagates_lookup_error(self):
        def run(dummy: DummyService):
            request = server.ChatRequest(messages=[server.ChatMessage(role="user", content="Hi")], k=3)

            with self.assertRaises(HTTPException) as ctx:
                server.chat(request)

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertEqual(dummy.chat_payloads[0][1], 3)
            self.assertEqual(dummy.load_calls, 1)

        with_dummy_service(run)


if __name__ == "__main__":
    unittest.main()
