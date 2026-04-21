"""
OCR Integration Tests for PageIndex GLM-OCR Pipeline

These tests cover three layers:
  1. Unit-level – _glm_ocr_recognize() with a mocked HTTP response
  2. Component-level – get_page_tokens(pdf_parser="glm-ocr") with a mocked OCR call
  3. End-to-end integration – full page_index() with pdf_parser="glm-ocr"
     (only runs when GLM_OCR_API_KEY / ZHIPU_API_KEY is set in the environment)

Run all fast (mocked) tests:
    pytest tests/test_ocr_integration.py -v

Run including live API tests (requires a valid GLM-OCR key):
    GLM_OCR_API_KEY=<your-key> pytest tests/test_ocr_integration.py -v
"""

import base64
import io
import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import pymupdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_PDF = _REPO_ROOT / "examples" / "documents" / "2023-annual-report-truncated.pdf"

# Minimal single-page PDF created in-memory so tests have no hard dependency on
# a specific file on disk (the sample PDF is used only in the live tests).
def _make_minimal_pdf_bytes() -> bytes:
    """Return the bytes of a tiny valid single-page PDF via PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Hello OCR Integration Test", fontsize=14)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _fake_ocr_response(text: str) -> dict:
    """Build a minimal GLM-API-like response dict."""
    return {"choices": [{"message": {"content": text}}]}


# ---------------------------------------------------------------------------
# 1. Unit tests: _glm_ocr_recognize
# ---------------------------------------------------------------------------

class TestGlmOcrRecognize(unittest.TestCase):
    """Tests for the low-level GLM-OCR HTTP wrapper."""

    def _import(self):
        from pageindex.utils import _glm_ocr_recognize
        return _glm_ocr_recognize

    def test_returns_text_on_success(self):
        _glm_ocr_recognize = self._import()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_ocr_response("Sample OCR text")
        mock_resp.raise_for_status.return_value = None

        with patch("pageindex.utils.requests.post", return_value=mock_resp) as mock_post:
            result = _glm_ocr_recognize("base64data==", api_key="test-key")

        self.assertEqual(result, "Sample OCR text")
        mock_post.assert_called_once()

    def test_sends_correct_api_key_header(self):
        _glm_ocr_recognize = self._import()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_ocr_response("text")
        mock_resp.raise_for_status.return_value = None

        with patch("pageindex.utils.requests.post", return_value=mock_resp) as mock_post:
            _glm_ocr_recognize("img==", api_key="my-secret-key")

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer my-secret-key")

    def test_sends_base64_image_in_payload(self):
        _glm_ocr_recognize = self._import()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_ocr_response("text")
        mock_resp.raise_for_status.return_value = None

        with patch("pageindex.utils.requests.post", return_value=mock_resp) as mock_post:
            _glm_ocr_recognize("IMAGEDATA==", api_key="k")

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        image_part = payload["messages"][0]["content"][0]
        self.assertIn("IMAGEDATA==", image_part["image_url"]["url"])

    def test_retries_on_failure_and_returns_empty_on_exhaustion(self):
        _glm_ocr_recognize = self._import()

        with patch("pageindex.utils.requests.post", side_effect=Exception("network error")), \
             patch("pageindex.utils.time.sleep"):  # speed up retries
            result = _glm_ocr_recognize("img==", api_key="k", max_retries=3)

        self.assertEqual(result, "")

    def test_succeeds_on_second_attempt_after_transient_failure(self):
        _glm_ocr_recognize = self._import()
        ok_resp = MagicMock()
        ok_resp.json.return_value = _fake_ocr_response("recovered text")
        ok_resp.raise_for_status.return_value = None

        side_effects = [Exception("transient"), ok_resp]
        with patch("pageindex.utils.requests.post", side_effect=side_effects), \
             patch("pageindex.utils.time.sleep"):
            result = _glm_ocr_recognize("img==", api_key="k", max_retries=3)

        self.assertEqual(result, "recovered text")

    def test_uses_glm_4v_ocr_model_name(self):
        _glm_ocr_recognize = self._import()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_ocr_response("text")
        mock_resp.raise_for_status.return_value = None

        with patch("pageindex.utils.requests.post", return_value=mock_resp) as mock_post:
            _glm_ocr_recognize("img==", api_key="k")

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "glm-4v-ocr")

    def test_request_includes_chinese_prompt(self):
        """OCR prompt must ask the model to preserve original layout."""
        _glm_ocr_recognize = self._import()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_ocr_response("text")
        mock_resp.raise_for_status.return_value = None

        with patch("pageindex.utils.requests.post", return_value=mock_resp) as mock_post:
            _glm_ocr_recognize("img==", api_key="k")

        _, kwargs = mock_post.call_args
        text_part = kwargs["json"]["messages"][0]["content"][1]
        self.assertEqual(text_part["type"], "text")
        self.assertIn("识别", text_part["text"])


# ---------------------------------------------------------------------------
# 2. Component tests: get_page_tokens with pdf_parser="glm-ocr"
# ---------------------------------------------------------------------------

class TestGetPageTokensGlmOcr(unittest.TestCase):
    """Tests for get_page_tokens() using the glm-ocr parser."""

    def _import(self):
        from pageindex.utils import get_page_tokens
        return get_page_tokens

    def _make_pdf_bytesio(self) -> BytesIO:
        return BytesIO(_make_minimal_pdf_bytes())

    # -- API-key validation --------------------------------------------------

    def test_raises_if_no_api_key(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        env_without_keys = {k: v for k, v in os.environ.items()
                            if k not in ("GLM_OCR_API_KEY", "ZHIPU_API_KEY")}
        with patch.dict("os.environ", env_without_keys, clear=True):
            with self.assertRaises(ValueError, msg="Expected ValueError when no API key is set but none was raised"):
                get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

    def test_accepts_glm_ocr_api_key_env_var(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "test-key"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="page text"):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")
        self.assertIsInstance(result, list)

    def test_accepts_zhipu_api_key_env_var(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        env = {k: v for k, v in os.environ.items() if k != "GLM_OCR_API_KEY"}
        env["ZHIPU_API_KEY"] = "zhipu-key"
        with patch.dict("os.environ", env, clear=True), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="page text"):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")
        self.assertIsInstance(result, list)

    # -- Return-value shape --------------------------------------------------

    def test_returns_list_of_tuples(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="hello"):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2, "Each tuple should be (text, token_count)")

    def test_tuple_first_element_is_str(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="extracted text"):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

        for text, _ in result:
            self.assertIsInstance(text, str)

    def test_tuple_second_element_is_non_negative_int(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="some text"):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

        for _, token_count in result:
            self.assertIsInstance(token_count, int)
            self.assertGreaterEqual(token_count, 0)

    def test_page_count_matches_pdf(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        expected_pages = pymupdf.open(stream=pdf_bytes, filetype="pdf").__len__()
        pdf_bytes.seek(0)

        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="text"):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

        self.assertEqual(len(result), expected_pages)

    def test_ocr_text_propagated_correctly(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        ocr_text = "Unique recognised text 12345"
        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value=ocr_text):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

        for text, _ in result:
            self.assertEqual(text, ocr_text)

    def test_accepts_pdf_file_path(self):
        get_page_tokens = self._import()
        if not _SAMPLE_PDF.exists():
            self.skipTest("Sample PDF not found")

        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="text"):
            result = get_page_tokens(str(_SAMPLE_PDF), pdf_parser="glm-ocr")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    # -- Concurrency ---------------------------------------------------------

    def test_ocr_called_once_per_page(self):
        get_page_tokens = self._import()
        # build a 3-page PDF
        doc = pymupdf.open()
        for i in range(3):
            pg = doc.new_page()
            pg.insert_text((72, 100), f"Page {i+1}")
        buf = BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)

        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value="t") as mock_ocr:
            result = get_page_tokens(buf, pdf_parser="glm-ocr")

        self.assertEqual(mock_ocr.call_count, 3)
        self.assertEqual(len(result), 3)

    # -- Error handling ------------------------------------------------------

    def test_returns_empty_string_when_ocr_fails(self):
        """If OCR exhausts all retries it returns ''; get_page_tokens should not crash."""
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()

        with patch.dict("os.environ", {"GLM_OCR_API_KEY": "k"}, clear=False), \
             patch("pageindex.utils._glm_ocr_recognize", return_value=""):
            result = get_page_tokens(pdf_bytes, pdf_parser="glm-ocr")

        for text, _ in result:
            self.assertEqual(text, "")

    def test_raises_for_unsupported_parser(self):
        get_page_tokens = self._import()
        pdf_bytes = self._make_pdf_bytesio()
        with self.assertRaises(ValueError):
            get_page_tokens(pdf_bytes, pdf_parser="unknown-ocr-engine")


# ---------------------------------------------------------------------------
# 3. End-to-end integration tests (require a real GLM-OCR API key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.getenv("GLM_OCR_API_KEY") or os.getenv("ZHIPU_API_KEY")),
    reason="Live OCR tests require GLM_OCR_API_KEY or ZHIPU_API_KEY to be set",
)
class TestPageIndexWithGlmOcrLive(unittest.TestCase):
    """
    Live integration tests that call the real GLM-OCR API.
    Skipped automatically unless an API key is present.
    """

    @classmethod
    def setUpClass(cls):
        if not _SAMPLE_PDF.exists():
            raise unittest.SkipTest(f"Sample PDF not found: {_SAMPLE_PDF}")

    def test_page_index_returns_structure(self):
        from pageindex.page_index import page_index

        result = page_index(
            str(_SAMPLE_PDF),
            pdf_parser="glm-ocr",
            if_add_node_id="yes",
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="no",
        )

        self.assertIsInstance(result, dict)
        self.assertIn("structure", result)
        self.assertIsInstance(result["structure"], list)
        self.assertGreater(len(result["structure"]), 0)

    def test_page_index_nodes_have_required_fields(self):
        from pageindex.page_index import page_index
        from pageindex.utils import structure_to_list

        result = page_index(
            str(_SAMPLE_PDF),
            pdf_parser="glm-ocr",
            if_add_node_id="yes",
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="no",
        )

        nodes = structure_to_list(result["structure"])
        self.assertGreater(len(nodes), 0)
        for node in nodes:
            self.assertIn("title", node, f"Node missing 'title': {node}")
            self.assertIn("start_index", node, f"Node missing 'start_index': {node}")
            self.assertIn("end_index", node, f"Node missing 'end_index': {node}")
            self.assertIn("node_id", node, f"Node missing 'node_id': {node}")

    def test_page_index_start_end_indices_are_valid(self):
        from pageindex.page_index import page_index
        from pageindex.utils import structure_to_list
        import PyPDF2

        result = page_index(
            str(_SAMPLE_PDF),
            pdf_parser="glm-ocr",
            if_add_node_id="yes",
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="no",
        )

        with open(_SAMPLE_PDF, "rb") as f:
            total_pages = len(PyPDF2.PdfReader(f).pages)

        nodes = structure_to_list(result["structure"])
        for node in nodes:
            start = node["start_index"]
            end = node["end_index"]
            if start is not None and end is not None:
                self.assertGreaterEqual(start, 1, f"start_index < 1 in {node}")
                self.assertLessEqual(end, total_pages, f"end_index > total_pages in {node}")
                self.assertLessEqual(start, end, f"start_index > end_index in {node}")

    def test_page_index_with_bytesio_input(self):
        """page_index should also accept a BytesIO object with glm-ocr parser."""
        from pageindex.page_index import page_index

        with open(_SAMPLE_PDF, "rb") as f:
            pdf_bytes = BytesIO(f.read())

        result = page_index(
            pdf_bytes,
            pdf_parser="glm-ocr",
            if_add_node_id="yes",
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="no",
        )

        self.assertIn("structure", result)
        self.assertGreater(len(result["structure"]), 0)

    def test_get_page_tokens_live_ocr(self):
        """Smoke-test: live OCR should return a non-empty text for each page."""
        from pageindex.utils import get_page_tokens

        result = get_page_tokens(str(_SAMPLE_PDF), pdf_parser="glm-ocr")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        non_empty = sum(1 for text, _ in result if text.strip())
        self.assertGreater(
            non_empty, 0,
            "At least one page should contain recognized text from a real PDF",
        )
