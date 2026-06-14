import importlib
import os
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
RAG_DIR = os.path.join(REPO_ROOT, "rag_system")
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)


class MatchValue:
    def __init__(self, value):
        self.value = value


class FieldCondition:
    def __init__(self, key, match):
        self.key = key
        self.match = match


class Filter:
    def __init__(self, must=None):
        self.must = must or []


models_mod = types.ModuleType("qdrant_client.models")
models_mod.Filter = Filter
models_mod.FieldCondition = FieldCondition
models_mod.MatchValue = MatchValue
qdrant_mod = types.ModuleType("qdrant_client")
qdrant_mod.models = models_mod
sys.modules.setdefault("qdrant_client", qdrant_mod)
sys.modules.setdefault("qdrant_client.models", models_mod)

ai_models_mod = types.ModuleType("ai_models")
ai_models_mod.encode_query = lambda query: [0.1, 0.2]
ai_models_mod.rerank_pairs = lambda pairs: []
sys.modules.setdefault("ai_models", ai_models_mod)

search = importlib.import_module("search")

fitz_mod = types.ModuleType("fitz")
fitz_mod.__version__ = "1.23.0"
pandas_mod = types.ModuleType("pandas")
sys.modules.setdefault("fitz", fitz_mod)
sys.modules.setdefault("pandas", pandas_mod)
extract_chunks = importlib.import_module("extract_chunks")


class DummyDoc:
    def __init__(self, payload, score=0.5, doc_id=1):
        self.payload = payload
        self.score = score
        self.id = doc_id


class SearchUtilsTest(unittest.TestCase):
    def test_extract_sku_variants_keeps_raw_and_normalized(self):
        variants = search.extract_sku_variants("Giá K-8658T-CP và 3722X-0")
        self.assertIn("K-8658T-CP", variants)
        self.assertIn("K8658TCP", variants)
        self.assertIn("3722X-0", variants)
        self.assertIn("3722X0", variants)

    def test_build_qdrant_filter_ignores_all_and_parses_year(self):
        q_filter = search.build_qdrant_filter("2025", "Tất cả", "Price List")
        self.assertEqual(len(q_filter.must), 2)
        self.assertEqual(q_filter.must[0].key, "year")
        self.assertEqual(q_filter.must[0].match.value, 2025)
        self.assertEqual(q_filter.must[1].key, "doc_type")

    def test_format_context_respects_char_budget(self):
        original_budget = search.config.MAX_CONTEXT_CHARS
        search.config.MAX_CONTEXT_CHARS = 80
        try:
            doc = DummyDoc({
                "source_file": "file.xlsx",
                "location": "Sheet A | Row 2",
                "content": "X" * 500,
            })
            context, sources = search.format_context_and_sources([(doc, 0.9)])
            self.assertLessEqual(len(context), 130)
            self.assertIn("đã cắt bớt", context)
            self.assertEqual(sources[0]["content"], "X" * 500)
        finally:
            search.config.MAX_CONTEXT_CHARS = original_budget

    def test_get_year_prefers_filename_over_mtime(self):
        year = extract_chunks.get_year("/tmp/BangGia_Kohler_2024.xlsx", "content 2023")
        self.assertEqual(year, 2024)

    def test_extract_price_from_row_text(self):
        self.assertIn("12.345.000", extract_chunks.extract_price("SKU K-123 Giá 12.345.000 VNĐ"))


if __name__ == "__main__":
    unittest.main()
