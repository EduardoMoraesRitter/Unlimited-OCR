"""Convert Unlimited-OCR layout output into deterministic, auditable JSON.

The model emits text with grounding markers instead of a guaranteed JSON
schema.  This module parses those markers locally; it never asks the model to
rewrite its answer and it intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_COORDINATE_MAX = 999
ImageSize = tuple[int, int]
ImageSizes = Sequence[ImageSize] | ImageSize
BBox = list[int | float]

_PAGE_RE = re.compile(r"<PAGE>", re.IGNORECASE)
_MARKER_RE = re.compile(
    r"(?:<\|ref\|>(?P<reference>.*?)<\|/ref\|>\s*)?"
    r"<\|det\|>(?P<detection>.*?)<\|/det\|>",
    re.DOTALL,
)
_WHITESPACE_RE = re.compile(r"\s+")


class _TableParser(HTMLParser):
    """Extract cell text and header-row positions from an HTML table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.header_rows: list[int] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self._row_has_header = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if tag == "tr":
            self._finish_row()
            self._row = []
            self._row_has_header = False
        elif tag in {"td", "th"}:
            if self._row is None:
                self._row = []
            self._finish_cell()
            self._cell_parts = []
            if tag == "th":
                self._row_has_header = True
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def close(self) -> None:
        super().close()
        self._finish_cell()
        self._finish_row()

    def _finish_cell(self) -> None:
        if self._cell_parts is None:
            return
        if self._row is None:
            self._row = []
        text = _WHITESPACE_RE.sub(" ", "".join(self._cell_parts)).strip()
        self._row.append(text)
        self._cell_parts = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row is None:
            return
        if self._row:
            if self._row_has_header:
                self.header_rows.append(len(self.rows))
            self.rows.append(self._row)
        self._row = None
        self._row_has_header = False


def _warning(code: str, message: str, **details: Any) -> dict[str, Any]:
    warning = {"code": code, "message": message}
    warning.update(details)
    return warning


def _normalise_type(value: str | None) -> str:
    if not value or not value.strip():
        return "unknown"
    label = _WHITESPACE_RE.sub("_", value.strip().lower())
    return label.strip("_:") or "unknown"


def _normalise_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    normalised: BBox = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        if isinstance(item, float):
            if not math.isfinite(item):
                return None
            normalised.append(int(item) if item.is_integer() else item)
        else:
            normalised.append(item)
    return normalised


def _bbox_literal(value: str) -> list[BBox] | None:
    """Safely decode one bbox or a list of bboxes."""
    try:
        decoded = ast.literal_eval(value.strip())
    except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
        return None

    single = _normalise_bbox(decoded)
    if single is not None:
        return [single]
    if not isinstance(decoded, (list, tuple)) or not decoded:
        return None
    multiple = [_normalise_bbox(item) for item in decoded]
    if any(item is None for item in multiple):
        return None
    return [item for item in multiple if item is not None]


def _decode_detection(
    reference: str | None,
    detection: str,
) -> tuple[str, list[BBox] | None, str | None]:
    """Return ``(type, bboxes, error)`` for either marker style."""
    body = detection.strip()
    bracket = body.find("[")

    if reference is not None:
        block_type = _normalise_type(reference)
        literal = body[bracket:] if bracket >= 0 else body
    elif bracket >= 0:
        block_type = _normalise_type(body[:bracket])
        literal = body[bracket:]
    else:
        return (
            _normalise_type(body),
            None,
            "Detection marker has no bounding box",
        )

    bboxes = _bbox_literal(literal)
    if bboxes is None:
        return block_type, None, f"Invalid bounding box: {literal!r}"
    return block_type, bboxes, None


def _enclosing_bbox(bboxes: Sequence[BBox]) -> BBox:
    """Return one stable outer box while preserving components separately."""
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def _pixel_bbox(
    bbox: Sequence[int | float],
    image_size: tuple[int, int] | None,
    coordinate_max: int | float,
) -> list[int] | None:
    if image_size is None:
        return None
    width, height = image_size
    return [
        round(bbox[0] * width / coordinate_max),
        round(bbox[1] * height / coordinate_max),
        round(bbox[2] * width / coordinate_max),
        round(bbox[3] * height / coordinate_max),
    ]


def _table_data(content: str) -> dict[str, Any]:
    parser = _TableParser()
    parser.feed(content)
    parser.close()
    return {"rows": parser.rows, "header_rows": parser.header_rows}


def _validate_bbox(
    bbox: Sequence[int | float],
    coordinate_max: int | float,
    block_index: int,
    bbox_index: int | None = None,
) -> list[dict[str, Any]]:
    warnings = []
    details = {"block_index": block_index}
    if bbox_index is not None:
        details["bbox_index"] = bbox_index
    if any(value < 0 or value > coordinate_max for value in bbox):
        warnings.append(
            _warning(
                "bbox_out_of_range",
                f"Bounding box is outside 0..{coordinate_max}",
                **details,
            )
        )
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        warnings.append(
            _warning(
                "bbox_inverted",
                "Bounding box right/bottom precedes left/top",
                **details,
            )
        )
    return warnings


def _repetition_warning(
    blocks: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Detect obvious loops without rejecting legitimate repetition."""
    previous = ""
    run = 0
    for block in blocks:
        content = _WHITESPACE_RE.sub(
            " ",
            str(block.get("content", "")),
        ).strip()
        if len(content) >= 20 and content == previous:
            run += 1
        else:
            previous = content
            run = 1
        if run >= 3:
            return _warning(
                "repetition_detected",
                "The same substantial block appears at least three times "
                "in a row",
                block_index=block.get("index"),
            )

    substantial_lines = [
        _WHITESPACE_RE.sub(" ", line).strip()
        for block in blocks
        for line in str(block.get("content", "")).splitlines()
        if len(_WHITESPACE_RE.sub(" ", line).strip()) >= 8
    ]
    if any(count >= 3 for count in Counter(substantial_lines).values()):
        return _warning(
            "repetition_detected",
            "A substantial output line appears at least three times",
        )
    return None


def _parse_page(
    raw_output: str,
    page_number: int,
    image_size: tuple[int, int] | None,
    include_raw: bool,
    coordinate_max: int | float,
) -> dict[str, Any]:
    matches = list(_MARKER_RE.finditer(raw_output))
    blocks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    det_open = raw_output.count("<|det|>")
    det_close = raw_output.count("<|/det|>")
    ref_open = raw_output.count("<|ref|>")
    ref_close = raw_output.count("<|/ref|>")
    if det_open != det_close or ref_open != ref_close:
        warnings.append(
            _warning(
                "unbalanced_markers",
                "Opening and closing OCR marker counts do not match",
                det_open=det_open,
                det_close=det_close,
                ref_open=ref_open,
                ref_close=ref_close,
            )
        )

    if not raw_output.strip():
        warnings.append(
            _warning("empty_output", "The model response is empty")
        )

    def add_plain_text(value: str) -> None:
        content = value.strip()
        if not content:
            return
        blocks.append(
            {
                "index": len(blocks),
                "type": "text",
                "bbox": None,
                "bbox_pixels": None,
                "content": content,
            }
        )

    if matches:
        add_plain_text(raw_output[: matches[0].start()])
    elif raw_output.strip():
        add_plain_text(raw_output)

    for match_index, match in enumerate(matches):
        content_end = (
            matches[match_index + 1].start()
            if match_index + 1 < len(matches)
            else len(raw_output)
        )
        content = raw_output[match.end() : content_end].strip()
        block_type, bboxes, bbox_error = _decode_detection(
            match.group("reference"),
            match.group("detection"),
        )
        block_index = len(blocks)
        bbox = _enclosing_bbox(bboxes) if bboxes else None
        bbox_warnings: list[dict[str, Any]] = []
        component_pixels: list[list[int] | None] = []
        if bboxes:
            multiple = len(bboxes) > 1
            for bbox_index, component in enumerate(bboxes):
                component_warnings = _validate_bbox(
                    component,
                    coordinate_max,
                    block_index,
                    bbox_index if multiple else None,
                )
                bbox_warnings.extend(component_warnings)
                component_pixels.append(
                    _pixel_bbox(component, image_size, coordinate_max)
                    if not component_warnings
                    else None
                )
        block: dict[str, Any] = {
            "index": block_index,
            "type": block_type,
            "bbox": bbox,
            "bbox_pixels": (
                _pixel_bbox(bbox, image_size, coordinate_max)
                if bbox is not None and not bbox_warnings
                else None
            ),
            "content": content,
        }
        if bboxes and len(bboxes) > 1:
            block["bboxes"] = bboxes
            block["bboxes_pixels"] = component_pixels
        if bbox_error:
            block["detection_raw"] = match.group("detection").strip()
            warnings.append(
                _warning(
                    "invalid_bbox",
                    bbox_error,
                    block_index=block_index,
                )
            )
        elif bboxes is not None:
            warnings.extend(bbox_warnings)

        if block_type == "table" or "<table" in content.lower():
            block["table"] = _table_data(content)
            if not block["table"]["rows"]:
                warnings.append(
                    _warning(
                        "table_without_rows",
                        "A table block contains no parseable rows",
                        block_index=block_index,
                    )
                )
        if not content:
            warnings.append(
                _warning(
                    "empty_block",
                    "A grounded block has no content",
                    block_index=block_index,
                )
            )
        blocks.append(block)

    repetition = _repetition_warning(blocks)
    if repetition:
        warnings.append(repetition)

    page: dict[str, Any] = {
        "page_number": page_number,
        "image_size": (
            {"width": image_size[0], "height": image_size[1]}
            if image_size is not None
            else None
        ),
        "blocks": blocks,
        "warnings": warnings,
    }
    if include_raw:
        page["raw_output"] = raw_output
    return page


def _normalise_image_sizes(
    image_sizes: ImageSizes | None,
    page_count: int,
) -> list[tuple[int, int] | None]:
    if image_sizes is None:
        return [None] * page_count
    if (
        page_count == 1
        and len(image_sizes) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in image_sizes
        )
    ):
        image_sizes = [(image_sizes[0], image_sizes[1])]
    if len(image_sizes) != page_count:
        raise ValueError(
            f"image_sizes has {len(image_sizes)} item(s), "
            f"expected {page_count}"
        )

    normalised: list[tuple[int, int] | None] = []
    for index, size in enumerate(image_sizes, start=1):
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in size
            )
            or size[0] <= 0
            or size[1] <= 0
        ):
            raise ValueError(f"Invalid image size for page {index}: {size!r}")
        normalised.append((size[0], size[1]))
    return normalised


def parse_ocr_pages(
    raw_pages: Iterable[str],
    source: str | None = None,
    image_sizes: ImageSizes | None = None,
    include_raw: bool = True,
    coordinate_max: int | float = DEFAULT_COORDINATE_MAX,
) -> dict[str, Any]:
    """Parse one raw response per page into a single JSON-ready document."""
    if isinstance(raw_pages, (str, bytes)):
        raise TypeError(
            "raw_pages must be an iterable of strings, not one string"
        )
    pages_input = list(raw_pages)
    if not pages_input:
        raise ValueError("raw_pages must contain at least one page")
    if any(not isinstance(page, str) for page in pages_input):
        raise TypeError("Every raw page must be a string")
    coordinate_max_is_valid = (
        isinstance(coordinate_max, int)
        and not isinstance(coordinate_max, bool)
        and coordinate_max > 0
    ) or (
        isinstance(coordinate_max, float)
        and math.isfinite(coordinate_max)
        and coordinate_max > 0
    )
    if not coordinate_max_is_valid:
        raise ValueError(
            "coordinate_max must be a finite number greater than zero"
        )

    sizes = _normalise_image_sizes(image_sizes, len(pages_input))
    pages = [
        _parse_page(raw, index, sizes[index - 1], include_raw, coordinate_max)
        for index, raw in enumerate(pages_input, start=1)
    ]
    warnings = [
        {"page_number": page["page_number"], **warning}
        for page in pages
        for warning in page["warnings"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "page_count": len(pages),
        "coordinate_system": {
            "type": "normalized",
            "minimum": 0,
            "maximum": coordinate_max,
        },
        "pages": pages,
        "warnings": warnings,
    }


def parse_ocr_output(
    raw_output: str,
    source: str | None = None,
    image_sizes: ImageSizes | None = None,
    include_raw: bool = True,
    coordinate_max: int | float = DEFAULT_COORDINATE_MAX,
) -> dict[str, Any]:
    """Parse a raw response, splitting case-insensitive ``<PAGE>`` markers."""
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    raw_pages = _PAGE_RE.split(raw_output)
    if len(raw_pages) > 1:
        while len(raw_pages) > 1 and not raw_pages[0].strip():
            raw_pages.pop(0)
        while len(raw_pages) > 1 and not raw_pages[-1].strip():
            raw_pages.pop()
    raw_pages = [page.strip("\r\n") for page in raw_pages] or [""]
    return parse_ocr_pages(
        raw_pages,
        source=source,
        image_sizes=image_sizes,
        include_raw=include_raw,
        coordinate_max=coordinate_max,
    )


def write_json(document: Any, output_path: str | Path) -> None:
    """Write indented UTF-8 JSON and create missing parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(
            document,
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")


def _image_size_argument(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if not match or int(match.group(1)) <= 0 or int(match.group(2)) <= 0:
        raise argparse.ArgumentTypeError(
            "image size must look like WIDTHxHEIGHT"
        )
    return int(match.group(1)), int(match.group(2))


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw Unlimited-OCR layout output to deterministic JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="UTF-8 raw model-output file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON file; stdout if omitted",
    )
    parser.add_argument("--source", help="Original image or PDF identifier")
    parser.add_argument(
        "--image-size",
        action="append",
        type=_image_size_argument,
        metavar="WIDTHxHEIGHT",
        help="Rendered size for a page; repeat this option for multiple pages",
    )
    parser.add_argument(
        "--coordinate-max",
        type=float,
        default=DEFAULT_COORDINATE_MAX,
        help="Maximum value used by the model's normalized coordinates",
    )
    parser.add_argument(
        "--exclude-raw",
        action="store_true",
        help="Do not embed raw model output in the JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    try:
        raw_output = args.input.read_text(encoding="utf-8-sig")
        document = parse_ocr_output(
            raw_output,
            source=args.source or str(args.input),
            image_sizes=args.image_size,
            include_raw=not args.exclude_raw,
            coordinate_max=args.coordinate_max,
        )
        if args.output:
            write_json(document, args.output)
            print(f"JSON written to: {args.output.resolve()}")
        else:
            json.dump(
                document,
                sys.stdout,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            sys.stdout.write("\n")
    except (OSError, TypeError, ValueError) as error:
        print(f"Could not convert OCR output: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
