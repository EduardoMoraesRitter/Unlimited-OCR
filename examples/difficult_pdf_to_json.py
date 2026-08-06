"""Parse a difficult PDF one page at a time with bounded GPU usage."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from unlimited_ocr_json import parse_ocr_pages, write_json  # noqa: E402

MODE_CONFIG = {
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}
RETRY_WARNING_CODES = {
    "bbox_inverted",
    "bbox_out_of_range",
    "empty_block",
    "empty_output",
    "invalid_bbox",
    "repetition_detected",
    "unbalanced_markers",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Render a PDF at 300 DPI and run Unlimited-OCR sequentially "
            "with a structured JSON result."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, help="Input PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (defaults to outputs/<pdf-name>)",
    )
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision to pin",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device exposed to the model",
    )
    parser.add_argument(
        "--mode",
        choices=("base", "gundam", "auto"),
        default="auto",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--password", help="Password for an encrypted PDF")
    parser.add_argument(
        "--prompt",
        default="<image>document parsing.",
        help="Model prompt; it must contain the <image> token",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help=(
            "Maximum total sequence length, including image and prompt tokens"
        ),
    )
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also embed raw model responses in document.json",
    )
    parser.add_argument(
        "--auto-min-chars",
        type=int,
        default=80,
        help="Retry a short base response with gundam in auto mode",
    )
    return parser


def load_model(model_name: str, revision: str | None) -> tuple[Any, Any, Any]:
    """Load tokenizer and model after CUDA_VISIBLE_DEVICES is configured."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This Transformers implementation requires "
            "an NVIDIA GPU."
        )

    common: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        common["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    model = AutoModel.from_pretrained(
        model_name,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        **common,
    )
    return torch, tokenizer, model.eval().cuda()


def response_is_suspicious(text: str, minimum_characters: int) -> bool:
    """Flag obvious truncation, empty output, or repetitive generation."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < minimum_characters:
        return True
    if compact.count("<|det|>") != compact.count("<|/det|>"):
        return True
    if compact.lower().count("<table") != compact.lower().count("</table>"):
        return True
    parsed = parse_ocr_pages([text], include_raw=False)
    warning_codes = {
        warning["code"]
        for warning in parsed["warnings"]
        if isinstance(warning, dict) and "code" in warning
    }
    if warning_codes & RETRY_WARNING_CODES:
        return True
    if len(compact) >= 240:
        tail = compact[-80:]
        if compact.count(tail) >= 3:
            return True
    return False


def infer_page(
    model: Any,
    tokenizer: Any,
    image_path: Path,
    artifact_dir: Path,
    mode: str,
    args: argparse.Namespace,
) -> str:
    """Run one single-image inference and return its unmodified response."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_output = model.infer(
        tokenizer,
        prompt=args.prompt,
        image_file=str(image_path),
        output_path=str(artifact_dir),
        max_length=args.max_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
        save_results=False,
        eval_mode=True,
        **MODE_CONFIG[mode],
    )
    if not isinstance(raw_output, str):
        raise RuntimeError("model.infer(eval_mode=True) returned no text")
    return raw_output


def attempt_modes(requested_mode: str) -> tuple[str, ...]:
    """Return the sequential inference plan for one page."""
    if requested_mode == "auto":
        return ("base", "gundam")
    return (requested_mode,)


def process_page(
    torch: Any,
    model: Any,
    tokenizer: Any,
    image_path: Path,
    artifact_root: Path,
    page_number: int,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Process one page, retrying suspect base output in auto mode."""
    candidates: list[tuple[str, str, bool]] = []
    attempts: list[dict[str, Any]] = []

    for mode in attempt_modes(args.mode):
        artifact_dir = artifact_root / f"page_{page_number:04d}_{mode}"
        try:
            raw_output = infer_page(
                model,
                tokenizer,
                image_path,
                artifact_dir,
                mode,
                args,
            )
            suspicious = response_is_suspicious(
                raw_output,
                args.auto_min_chars,
            )
            candidates.append((mode, raw_output, suspicious))
            attempts.append(
                {
                    "mode": mode,
                    "status": "suspect" if suspicious else "ok",
                    "characters": len(raw_output),
                }
            )
            if args.mode != "auto" or not suspicious:
                break
        except RuntimeError as error:
            is_oom = "out of memory" in str(error).lower()
            attempts.append(
                {
                    "mode": mode,
                    "status": "cuda_oom" if is_oom else "error",
                    "error": str(error),
                }
            )
            if is_oom:
                torch.cuda.empty_cache()
                # Gundam uses additional crops, so it is not a useful retry
                # after the lower-memory base mode has already exhausted VRAM.
                break
        except Exception as error:  # Remote model code can raise many types.
            attempts.append(
                {"mode": mode, "status": "error", "error": str(error)}
            )

    if candidates:
        selected_mode, selected_raw, suspicious = max(
            candidates,
            key=lambda candidate: (
                not candidate[2],
                len(candidate[1]),
            ),
        )
        status = "suspect" if suspicious else "ok"
    else:
        selected_mode, selected_raw, status = None, "", "failed"

    metadata = {
        "page": page_number,
        "status": status,
        "selected_mode": selected_mode,
        "attempts": attempts,
    }
    return selected_raw, metadata


def open_pdf(pdf_path: Path, password: str | None) -> Any:
    """Open and, when necessary, authenticate a PDF."""
    import fitz

    document = fitz.open(pdf_path)
    if document.needs_pass:
        if not password or document.authenticate(password) <= 0:
            document.close()
            raise ValueError(
                "The PDF is encrypted; provide the correct --password"
            )
    return document


def add_processing_metadata(
    parsed_document: Any,
    pdf_path: Path,
    page_metadata: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Attach reproducibility and failures without changing parsed blocks."""
    processing = {
        "source": pdf_path.name,
        "dpi": args.dpi,
        "requested_mode": args.mode,
        "max_length": args.max_length,
        "concurrency": 1,
        "pages": page_metadata,
    }
    if isinstance(parsed_document, dict):
        parsed_document["processing"] = processing
        return parsed_document
    return {"document": parsed_document, "processing": processing}


def validate_args(args: argparse.Namespace) -> str | None:
    """Return a validation error, or None when arguments are usable."""
    if not args.pdf.expanduser().resolve().is_file():
        return f"PDF not found: {args.pdf.expanduser().resolve()}"
    if "<image>" not in args.prompt:
        return "--prompt must contain the <image> token"
    if args.dpi <= 0:
        return "--dpi must be greater than zero"
    if args.max_length <= 0:
        return "--max-length must be greater than zero"
    if args.auto_min_chars < 0:
        return "--auto-min-chars cannot be negative"
    return None


def main() -> int:
    """Render, infer, parse, and save the aggregate document."""
    args = build_parser().parse_args()
    validation_error = validate_args(args)
    if validation_error:
        print(validation_error, file=sys.stderr)
        return 2

    pdf_path = args.pdf.expanduser().resolve()
    output_dir = args.output_dir or Path("outputs") / pdf_path.stem
    output_dir = output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        document = open_pdf(pdf_path, args.password)
    except Exception as error:
        print(f"Could not open PDF: {error}", file=sys.stderr)
        return 2
    if document.page_count == 0:
        document.close()
        print("The PDF has no pages", file=sys.stderr)
        return 2

    try:
        torch, tokenizer, model = load_model(args.model, args.revision)
    except Exception as error:
        document.close()
        print(f"Could not load model: {error}", file=sys.stderr)
        return 1

    raw_pages: list[str] = []
    image_sizes: list[tuple[int, int]] = []
    page_metadata: list[dict[str, Any]] = []
    failed_pages = 0

    try:
        import fitz

        with tempfile.TemporaryDirectory(prefix="unlimited_ocr_pdf_") as temp:
            temp_dir = Path(temp)
            matrix = fitz.Matrix(args.dpi / 72, args.dpi / 72)
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                image_path = temp_dir / f"page_{page_number:04d}.png"
                image_size = (
                    round(page.rect.width * args.dpi / 72),
                    round(page.rect.height * args.dpi / 72),
                )
                try:
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image_size = (pixmap.width, pixmap.height)
                    pixmap.save(image_path)
                    raw_output, metadata = process_page(
                        torch,
                        model,
                        tokenizer,
                        image_path,
                        temp_dir,
                        page_number,
                        args,
                    )
                except Exception as error:
                    raw_output = ""
                    metadata = {
                        "page": page_number,
                        "status": "render_error",
                        "selected_mode": None,
                        "attempts": [],
                        "error": str(error),
                    }
                finally:
                    image_path.unlink(missing_ok=True)

                raw_path = raw_dir / f"page_{page_number:04d}.txt"
                raw_path.write_text(raw_output, encoding="utf-8")
                metadata["raw_file"] = str(raw_path.relative_to(output_dir))
                raw_pages.append(raw_output)
                image_sizes.append(image_size)
                page_metadata.append(metadata)
                status = metadata["status"]
                if status == "failed" or status.endswith("error"):
                    failed_pages += 1
                print(
                    f"Page {page_number}/{document.page_count}: "
                    f"{metadata['status']} ({metadata['selected_mode']})"
                )
    finally:
        document.close()

    parsed_document = parse_ocr_pages(
        raw_pages,
        source=str(pdf_path),
        image_sizes=image_sizes,
        include_raw=args.include_raw,
        coordinate_max=999,
    )
    aggregate = add_processing_metadata(
        parsed_document,
        pdf_path,
        page_metadata,
        args,
    )
    json_path = output_dir / "document.json"
    write_json(aggregate, json_path)
    print(f"Aggregate JSON written to: {json_path}")

    if failed_pages:
        print(
            f"Completed with {failed_pages} failed page(s); "
            "inspect processing.pages.",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
