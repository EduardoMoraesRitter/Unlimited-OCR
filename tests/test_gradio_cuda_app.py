"""Offline tests for the local web UI helpers (no Gradio, model, or CUDA)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples.gradio_cuda_app import (
    MAX_IMAGE_PIXELS,
    PREVIEW_MAX_PIXELS,
    PREVIEW_MAX_SIDE,
    _check_pixel_limit,
    _page_and_dpi,
    preview_document,
    preview_new_upload,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PreviewControlTests(unittest.TestCase):
    def test_accepts_supported_page_and_dpi(self) -> None:
        self.assertEqual(_page_and_dpi(2, 250), (2, 250))

    def test_rejects_invalid_page_and_dpi(self) -> None:
        with self.assertRaisesRegex(ValueError, "página deve ser 1"):
            _page_and_dpi(0, 200)
        with self.assertRaisesRegex(ValueError, "150, 200, 250 ou 300"):
            _page_and_dpi(1, 400)
        with self.assertRaisesRegex(ValueError, "número inteiro"):
            _page_and_dpi(1.5, 200)
        with self.assertRaisesRegex(ValueError, "número inteiro"):
            _page_and_dpi(float("nan"), 200)

    def test_rejects_invalid_or_excessive_pixel_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensões inválidas"):
            _check_pixel_limit(0, 100)
        with self.assertRaisesRegex(ValueError, "25 milhões"):
            _check_pixel_limit(MAX_IMAGE_PIXELS + 1, 1)


class DocumentPreviewTests(unittest.TestCase):
    def test_previews_repository_image(self) -> None:
        preview, note = preview_document(
            REPOSITORY_ROOT / "assets" / "baidu.png",
            1,
            200,
        )
        self.assertIsNotNone(preview)
        self.assertEqual(preview.size, (440, 133))
        self.assertIn("imagem", note)
        self.assertIn("440 × 133", note)
        fake_gradio = SimpleNamespace(update=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"gradio": fake_gradio}):
            _, _, page_update = preview_new_upload(
                REPOSITORY_ROOT / "assets" / "baidu.png",
                200,
            )
        self.assertEqual(page_update["maximum"], 1)
        self.assertFalse(page_update["interactive"])

    def test_previews_selected_pdf_page_with_ocr_dimensions(self) -> None:
        preview, note = preview_document(
            REPOSITORY_ROOT / "Unlimited-OCR.pdf",
            1,
            200,
        )
        self.assertIsNotNone(preview)
        self.assertEqual(preview.size, (993, 1404))
        self.assertLessEqual(max(preview.size), PREVIEW_MAX_SIDE)
        self.assertLessEqual(preview.width * preview.height, PREVIEW_MAX_PIXELS)
        self.assertIn("página **1 de 14**", note)
        self.assertIn("1654 × 2339", note)
        self.assertIn("200 DPI", note)
        self.assertNotIn(str(REPOSITORY_ROOT), note)

        fake_gradio = SimpleNamespace(update=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"gradio": fake_gradio}):
            _, _, page_update = preview_new_upload(
                REPOSITORY_ROOT / "Unlimited-OCR.pdf",
                200,
            )
        self.assertEqual(page_update["maximum"], 14)
        self.assertTrue(page_update["interactive"])

        import fitz

        with tempfile.TemporaryDirectory() as temp:
            large_pdf = Path(temp) / "large-page.pdf"
            with fitz.open() as document:
                document.new_page(width=1000, height=1000)
                document.save(large_pdf)
            bounded_preview, bounded_note = preview_document(large_pdf, 1, 150)
        self.assertIsNotNone(bounded_preview)
        self.assertEqual(bounded_preview.size, (1600, 1600))
        self.assertLessEqual(
            bounded_preview.width * bounded_preview.height,
            PREVIEW_MAX_PIXELS,
        )
        self.assertIn("2084 × 2084", bounded_note)

    def test_reports_pdf_page_out_of_range_without_crashing(self) -> None:
        preview, note = preview_document(
            REPOSITORY_ROOT / "Unlimited-OCR.pdf",
            999,
            200,
        )
        self.assertIsNone(preview)
        self.assertIn("Escolha uma página entre 1 e 14", note)


if __name__ == "__main__":
    unittest.main()
