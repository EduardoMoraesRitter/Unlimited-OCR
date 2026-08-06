"""Convert a saved Unlimited-OCR response to deterministic JSON offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from unlimited_ocr_json import parse_ocr_output, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    sample = Path(__file__).with_name("raw_output.txt")
    parser = argparse.ArgumentParser(
        description="Convert raw Unlimited-OCR output to JSON without a GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=sample,
        help="UTF-8 text file containing the raw model response",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file (defaults to outputs/<input>.json)",
    )
    return parser


def main() -> int:
    """Parse the input file and write the resulting JSON document."""
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output or Path("outputs") / f"{input_path.stem}.json"
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    raw_output = input_path.read_text(encoding="utf-8")
    document = parse_ocr_output(
        raw_output,
        source=str(input_path),
        include_raw=True,
        coordinate_max=999,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(document, output_path)
    print(f"JSON written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
