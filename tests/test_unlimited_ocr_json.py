"""Offline tests for the Unlimited-OCR JSON post-processor."""

import json
import tempfile
import unittest
from pathlib import Path

from unlimited_ocr_json import (
    parse_ocr_output,
    parse_ocr_pages,
    write_json,
)


def warning_codes(document):
    """Return warning codes whether warnings live on the document or pages."""
    warnings = list(document.get("warnings", []))
    for page in document.get("pages", []):
        warnings.extend(page.get("warnings", []))

    return {
        warning.get("code") if isinstance(warning, dict) else warning
        for warning in warnings
    }


class ParseOcrOutputTests(unittest.TestCase):
    def test_parses_inline_det_marker(self):
        raw = (
            "<|det|>header [55, 50, 724, 104]<|/det|>"
            "RELATÓRIO DE TESTE"
        )

        document = parse_ocr_output(raw, source="report.png")
        block = document["pages"][0]["blocks"][0]

        self.assertEqual(len(document["pages"]), 1)
        self.assertEqual(document["source"], "report.png")
        self.assertEqual(block["type"], "header")
        self.assertEqual(block["bbox"], [55, 50, 724, 104])
        self.assertEqual(block["content"], "RELATÓRIO DE TESTE")

    def test_parses_ref_then_det_marker(self):
        raw = (
            "<|ref|>title<|/ref|>"
            "<|det|>[[10, 20, 900, 120]]<|/det|>Quarterly results"
        )

        document = parse_ocr_output(raw)
        block = document["pages"][0]["blocks"][0]

        self.assertEqual(block["type"], "title")
        self.assertEqual(block["bbox"], [10, 20, 900, 120])
        self.assertEqual(block["content"], "Quarterly results")

    def test_page_token_and_explicit_pages_create_numbered_pages(self):
        raw = (
            "<|det|>text [1, 2, 3, 4]<|/det|>first"
            "\n<PAGE>\n"
            "<|det|>text [5, 6, 7, 8]<|/det|>second"
        )

        from_token = parse_ocr_output(raw)
        from_list = parse_ocr_pages(["first page", "second page"])

        self.assertEqual(len(from_token["pages"]), 2)
        self.assertEqual(
            [page["page_number"] for page in from_token["pages"]],
            [1, 2],
        )
        self.assertEqual(
            from_token["pages"][1]["blocks"][0]["content"],
            "second",
        )
        self.assertEqual(len(from_list["pages"]), 2)
        self.assertEqual(
            from_list["pages"][0]["blocks"][0]["content"],
            "first page",
        )

    def test_extracts_html_table(self):
        raw = (
            "<|det|>table [0, 0, 999, 999]<|/det|>"
            "<table><tr><th>Item</th><th>Qtd.</th></tr>"
            "<tr><td>Café</td><td>2</td></tr></table>"
        )

        block = parse_ocr_output(raw)["pages"][0]["blocks"][0]

        self.assertEqual(block["type"], "table")
        self.assertEqual(block["table"]["header_rows"], [0])
        self.assertEqual(block["table"]["rows"][0], ["Item", "Qtd."])
        self.assertEqual(block["table"]["rows"][-1], ["Café", "2"])
        self.assertIn("<table>", block["content"])

    def test_converts_normalized_bbox_to_pixels(self):
        raw = "<|det|>text [100, 200, 900, 800]<|/det|>pixel test"

        block = parse_ocr_output(
            raw,
            image_sizes=[(2000, 1000)],
            coordinate_max=1000,
        )["pages"][0]["blocks"][0]

        self.assertEqual(block["bbox"], [100, 200, 900, 800])
        self.assertEqual(block["bbox_pixels"], [200, 200, 1800, 800])

    def test_preserves_multiple_bboxes_and_computes_outer_bbox(self):
        raw = (
            "<|ref|>text<|/ref|>"
            "<|det|>[[10, 20, 100, 80], [120, 25, 300, 90]]"
            "<|/det|>two regions"
        )

        block = parse_ocr_output(
            raw,
            image_sizes=[(1000, 1000)],
        )["pages"][0]["blocks"][0]

        self.assertEqual(block["bbox"], [10, 20, 300, 90])
        self.assertEqual(
            block["bboxes"],
            [[10, 20, 100, 80], [120, 25, 300, 90]],
        )
        self.assertEqual(len(block["bboxes_pixels"]), 2)

    def test_warns_for_invalid_and_out_of_range_bboxes(self):
        invalid = parse_ocr_output(
            "<|det|>text [1, nope, 3, 4]<|/det|>bad"
        )
        out_of_range = parse_ocr_output(
            "<|det|>text [-1, 0, 1000, 999]<|/det|>outside"
        )

        self.assertIn("invalid_bbox", warning_codes(invalid))
        self.assertIn("bbox_out_of_range", warning_codes(out_of_range))

    def test_warns_for_repeated_output(self):
        line = "same substantial OCR output line"
        document = parse_ocr_output(f"{line}\n{line}\n{line}\n")

        self.assertIn("repetition_detected", warning_codes(document))

    def test_keeps_unmarked_text_as_a_text_block(self):
        document = parse_ocr_output("Primeira linha\nSegunda linha")
        block = document["pages"][0]["blocks"][0]

        self.assertEqual(block["type"], "text")
        self.assertIsNone(block["bbox"])
        self.assertEqual(block["content"], "Primeira linha\nSegunda linha")

    def test_include_raw_false_omits_raw_output_everywhere(self):
        document = parse_ocr_output("plain text", include_raw=False)

        self.assertNotIn("raw_output", document)
        for page in document["pages"]:
            self.assertNotIn("raw_output", page)

    def test_bbox_payload_is_data_and_is_never_evaluated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sentinel = Path(temp_dir) / "eval-was-used.txt"
            expression = (
                "__import__('pathlib').Path("
                f"{str(sentinel)!r}).write_text('owned')"
            )
            raw = f"<|det|>text [{expression}]<|/det|>safe text"

            document = parse_ocr_output(raw)

            self.assertFalse(sentinel.exists())
            self.assertEqual(
                document["pages"][0]["blocks"][0]["content"],
                "safe text",
            )
            self.assertIn("invalid_bbox", warning_codes(document))


class WriteJsonTests(unittest.TestCase):
    def test_writes_utf8_json_and_creates_parent_directories(self):
        document = parse_ocr_output("Olá, São Paulo")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "result.json"
            write_json(document, output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                document,
            )
            self.assertIn(
                "Olá, São Paulo",
                output_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
