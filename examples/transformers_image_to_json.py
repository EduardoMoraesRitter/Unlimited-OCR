"""Run Unlimited-OCR on one image and save raw output plus parsed JSON."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from unlimited_ocr_json import parse_ocr_output, write_json  # noqa: E402

MODE_CONFIG = {
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run local Transformers inference and emit structured JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Input PNG, JPEG, WebP, or BMP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file (defaults to outputs/<image>.json)",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Raw response file (defaults to <output>.raw.txt)",
    )
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision to pin",
    )
    parser.add_argument("--mode", choices=tuple(MODE_CONFIG), default="base")
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device exposed to the model",
    )
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
    return parser


def load_model(model_name: str, revision: str | None) -> tuple[Any, Any, Any]:
    """Load the tokenizer and model after configuring the CUDA device."""
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


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return absolute JSON and raw-output paths."""
    image = args.image.expanduser().resolve()
    output = args.output or Path("outputs") / f"{image.stem}.json"
    output = output.expanduser().resolve()
    raw_output = args.raw_output or output.with_suffix(".raw.txt")
    return output, raw_output.expanduser().resolve()


def main() -> int:
    """Run OCR, preserve the raw response, and convert it to JSON."""
    args = build_parser().parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2
    if "<image>" not in args.prompt:
        print("--prompt must contain the <image> token", file=sys.stderr)
        return 2
    if args.max_length <= 0:
        print("--max-length must be greater than zero", file=sys.stderr)
        return 2

    output_path, raw_path = resolve_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        torch, tokenizer, model = load_model(args.model, args.revision)
        config = MODE_CONFIG[args.mode]
        with tempfile.TemporaryDirectory(prefix="unlimited_ocr_") as temp_dir:
            raw_output = model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=str(image_path),
                output_path=temp_dir,
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window,
                save_results=False,
                eval_mode=True,
                **config,
            )
        if not isinstance(raw_output, str):
            raise RuntimeError("model.infer(eval_mode=True) returned no text")
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            try:
                torch.cuda.empty_cache()
            except (NameError, AttributeError):
                pass
            print(
                "CUDA ran out of memory. Try --mode base, reduce "
                "--max-length, and close other GPU applications.",
                file=sys.stderr,
            )
            return 3
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    # Model downloads and trusted remote code may fail with backend-specific
    # exception classes, so this CLI boundary reports them uniformly.
    except Exception as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1

    from PIL import Image

    with Image.open(image_path) as image:
        image_size = image.size

    raw_path.write_text(raw_output, encoding="utf-8")
    document = parse_ocr_output(
        raw_output,
        source=str(image_path),
        image_sizes=[image_size],
        include_raw=True,
        coordinate_max=999,
    )
    write_json(document, output_path)
    print(f"Raw response written to: {raw_path}")
    print(f"JSON written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
