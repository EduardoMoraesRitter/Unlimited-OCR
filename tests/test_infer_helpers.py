"""Tests for JSON-related batch runner behavior that does not require a GPU."""

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from examples.difficult_pdf_to_json import response_is_suspicious
from infer import build_jobs, collect_stream_silent, run, run_jobs


class BatchRunnerTests(unittest.TestCase):
    def test_different_image_extensions_have_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            (images / "scan.png").write_bytes(b"png")
            (images / "scan.jpg").write_bytes(b"jpg")
            args = Namespace(
                pdf="",
                image_dir=str(images),
                output_dir=str(root / "outputs"),
            )

            jobs = build_jobs(args)
            outputs = [job[1] for job in jobs]

            self.assertEqual(len(outputs), len(set(outputs)))
            self.assertEqual(
                {Path(output).name for output in outputs},
                {"scan.jpg.md", "scan.png.md"},
            )

    def test_stream_writer_creates_nested_output_directories(self):
        class Response:
            @staticmethod
            def iter_lines():
                yield b'data: {"choices":[{"delta":{"content":"ok"}}]}'
                yield b"data: [DONE]"

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nested" / "scan.png.md"

            result = collect_stream_silent(Response(), str(output))

            self.assertEqual(result["text"], "ok")
            self.assertEqual(output.read_text(encoding="utf-8"), "ok")

    def test_pdf_temporary_directory_is_removed(self):
        args = Namespace(
            pdf="document.pdf",
            image_dir="",
            output_dir="outputs",
            json=False,
        )
        observed = []

        def fake_build_jobs(_args, pdf_temp_dir=None):
            self.assertTrue(os.path.isdir(pdf_temp_dir))
            observed.append(pdf_temp_dir)
            return [("page.png", "page.md", "document.pdf#page=1")]

        with (
            patch("infer.build_jobs", side_effect=fake_build_jobs),
            patch("infer.run_jobs", return_value=0),
        ):
            self.assertEqual(run(args), 0)

        self.assertEqual(len(observed), 1)
        self.assertFalse(os.path.exists(observed[0]))

    def test_json_export_error_produces_failure_status(self):
        args = Namespace(
            output_dir="",
            pdf="",
            concurrency=1,
            image_mode="base",
        )
        result = {
            "tokens": 10,
            "decode_time": 0.1,
            "text": "ok",
            "json_error": "disk full",
        }
        jobs = [("page.png", "page.md", "page.png")]

        with (
            patch("infer.infer_one", return_value=result),
            patch("builtins.print"),
        ):
            self.assertEqual(run_jobs(args, jobs), 1)


class DifficultPdfHeuristicTests(unittest.TestCase):
    def test_repeated_lines_trigger_retry(self):
        line = "same substantial OCR output line"
        raw = "\n".join([line] * 3)

        self.assertTrue(response_is_suspicious(raw, 80))

    def test_unbalanced_reference_marker_triggers_retry(self):
        raw = "<|ref|>text<|det|>[1, 2, 3, 4]<|/det|>broken"

        self.assertTrue(response_is_suspicious(raw, 10))


if __name__ == "__main__":
    unittest.main()
