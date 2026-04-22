import copy
import json
import unittest
from unittest.mock import patch

from pageindex.client import PageIndexClient
from pageindex.retrieve import search_document, search_document_by_embedding, search_document_hybrid


def _make_documents():
    return {
        "doc-1": {
            "id": "doc-1",
            "type": "pdf",
            "path": "/tmp/doc.pdf",
            "page_count": 5,
            "structure": [
                {
                    "node_id": "0001",
                    "title": "Alpha",
                    "summary": "Alpha summary",
                    "start_index": 1,
                    "end_index": 2,
                    "nodes": [],
                },
                {
                    "node_id": "0002",
                    "title": "Beta",
                    "summary": "Beta summary",
                    "start_index": 3,
                    "end_index": 3,
                    "nodes": [],
                },
                {
                    "node_id": "0003",
                    "title": "Gamma",
                    "summary": "Gamma summary",
                    "start_index": 4,
                    "end_index": 5,
                    "nodes": [],
                },
            ],
            "pages": [
                {"page": 1, "content": "page 1 alpha"},
                {"page": 2, "content": "page 2 alpha"},
                {"page": 3, "content": "page 3 beta"},
                {"page": 4, "content": "page 4 gamma"},
                {"page": 5, "content": "page 5 gamma"},
            ],
        }
    }


def _fake_embedding_completion(model, input_texts):
    if len(input_texts) == 1:
        return [[1.0, 0.0]]
    return [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ]


class RetrieveTests(unittest.TestCase):
    def setUp(self):
        self.documents = copy.deepcopy(_make_documents())

    @patch("pageindex.retrieve.embedding_completion", side_effect=_fake_embedding_completion)
    def test_embedding_search_returns_score_and_section_metadata(self, _mock_embedding):
        result = json.loads(
            search_document_by_embedding(
                self.documents,
                "doc-1",
                "alpha query",
                embedding_model="text-embedding-3-small",
                top_k=2,
            )
        )

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["score"], 1.0)
        self.assertEqual(result[0]["section"]["node_id"], "0001")
        self.assertEqual(result[0]["section"]["title"], "Alpha")
        self.assertEqual(result[0]["section"]["start"], 1)
        self.assertEqual(result[0]["section"]["end"], 2)
        self.assertEqual(result[0]["section"]["context_start"], 1)
        self.assertEqual(result[0]["section"]["context_end"], 2)
        self.assertEqual([item["page"] for item in result[0]["content"]], [1, 2])

    @patch("pageindex.retrieve.llm_completion", return_value="[1]")
    @patch("pageindex.retrieve.embedding_completion", side_effect=_fake_embedding_completion)
    def test_hybrid_search_expands_context_around_selected_hit(self, _mock_embedding, _mock_llm):
        result = json.loads(
            search_document_hybrid(
                self.documents,
                "doc-1",
                "beta query",
                model="gpt-5.4",
                embedding_model="text-embedding-3-small",
                top_k=1,
                candidate_k=3,
                context_window=1,
            )
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section"]["node_id"], "0002")
        self.assertEqual(result[0]["section"]["start"], 3)
        self.assertEqual(result[0]["section"]["end"], 3)
        self.assertEqual(result[0]["section"]["context_start"], 2)
        self.assertEqual(result[0]["section"]["context_end"], 4)
        self.assertEqual([item["page"] for item in result[0]["content"]], [2, 3, 4])
        self.assertAlmostEqual(result[0]["score"], 0.70710678, places=6)

    @patch("pageindex.retrieve.llm_completion", return_value="[1]")
    def test_tree_search_returns_section_metadata(self, _mock_llm):
        result = json.loads(search_document(self.documents, "doc-1", "beta query", model="gpt-5.4"))

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["score"])
        self.assertEqual(result[0]["section"]["node_id"], "0002")
        self.assertEqual([item["page"] for item in result[0]["content"]], [3])

    @patch("pageindex.client.search_document_hybrid", return_value="[]")
    def test_client_uses_default_hybrid_context_window(self, mock_search_hybrid):
        client = PageIndexClient(
            model="gpt-4o-2024-11-20",
            retrieve_model="gpt-5.4",
            embedding_model="text-embedding-3-small",
            hybrid_context_window=2,
        )
        client.documents["doc-1"] = copy.deepcopy(_make_documents()["doc-1"])

        client.search_document("doc-1", "beta query", strategy="hybrid")

        self.assertEqual(mock_search_hybrid.call_args.kwargs["context_window"], 2)


if __name__ == "__main__":
    unittest.main()
