"""Ensure CLI entry points resolve to the packaged modules."""
from importlib import import_module


def test_ingest_cli_importable():
    module = import_module("rag.cli.ingest")
    assert hasattr(module, "main"), "ingest CLI should expose a main() function"


def test_query_cli_importable():
    module = import_module("rag.cli.query")
    assert hasattr(module, "main"), "query CLI should expose a main() function"
