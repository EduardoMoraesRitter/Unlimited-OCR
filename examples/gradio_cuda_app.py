"""Small local Gradio UI for Unlimited-OCR running on CUDA.

Run this file inside the same environment used by the Transformers example.
The model is loaded lazily on the first OCR request so opening the page does
not immediately reserve almost all VRAM on an 8 GB GPU.
"""

from __future__ import annotations

import ctypes
import gc
import math
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.difficult_pdf_to_json import response_is_suspicious  # noqa: E402
from unlimited_ocr_json import parse_ocr_output, write_json  # noqa: E402

MODEL_NAME = "baidu/Unlimited-OCR"
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
PROMPT = "<image>document parsing."
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "web-ui"
DEFAULT_DPI = 200
DEFAULT_MAX_LENGTH = 4096
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
APP_CSS = """
:root { --cuda: #76b900; --ink: #17211a; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
.hero { padding: 20px 4px 6px; }
.hero h1 { font-family: Bahnschrift, 'Aptos Display', sans-serif; letter-spacing: -0.04em; }
.hero p { max-width: 720px; color: #58645c; }
.cuda-card { border-left: 4px solid var(--cuda) !important; }
.run-button { background: var(--cuda) !important; border-color: var(--cuda) !important; color: #10170d !important; font-weight: 750 !important; }
.result-code textarea, .result-code pre { font-family: 'Cascadia Code', Consolas, monospace !important; }
footer { display: none !important; }
"""

_MODEL: Any | None = None
_TOKENIZER: Any | None = None
_TORCH: Any | None = None
_INFERENCE_LOCK = threading.Lock()


def _trim_linux_memory() -> None:
    """Ask glibc to return unused host memory to WSL when possible."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def cuda_status() -> str:
    """Return a compact status badge for the page header."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "🔴 **CUDA indisponível** — o modelo não pode ser executado."
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
        model_state = "modelo carregado" if _MODEL is not None else "pronto para carregar"
        return (
            "🟢 **CUDA ativo** · "
            f"{torch.cuda.get_device_name(0)} · "
            f"{free_gib:.1f}/{total_gib:.1f} GB livres · {model_state}"
        )
    except Exception as error:  # The status area must not crash the UI.
        return f"🔴 **Falha ao consultar CUDA:** {error}"


def _load_model() -> tuple[Any, Any, Any]:
    """Load the pinned model once, using the same CUDA path as the CLI."""
    global _MODEL, _TOKENIZER, _TORCH
    if _MODEL is not None:
        return _TORCH, _TOKENIZER, _MODEL

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível neste ambiente.")

    common = {
        "trust_remote_code": True,
        "revision": MODEL_REVISION,
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **common)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **common,
    )
    model = model.eval().cuda()
    _TORCH, _TOKENIZER, _MODEL = torch, tokenizer, model
    _trim_linux_memory()
    return torch, tokenizer, model


def _upload_path(value: Any) -> Path:
    """Normalize Gradio's filepath value across recent Gradio versions."""
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, dict) and value.get("path"):
        path = Path(value["path"])
    elif hasattr(value, "name"):
        path = Path(value.name)
    else:
        raise ValueError("Selecione uma imagem ou um PDF.")
    if not path.is_file():
        raise ValueError("O arquivo enviado não foi encontrado.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("O arquivo excede o limite local de 25 MB.")
    return path


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-.")
    return stem[:60] or "documento"


def _prepare_image(
    upload: Path,
    page_number: int,
    dpi: int,
    temp_dir: Path,
) -> tuple[Path, str]:
    """Return an image path and a privacy-safe source label."""
    if upload.suffix.lower() != ".pdf":
        return upload, upload.name

    import fitz

    with fitz.open(upload) as document:
        if document.page_count == 0:
            raise ValueError("O PDF não contém páginas.")
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(
                f"Escolha uma página entre 1 e {document.page_count}."
            )
        page = document.load_page(page_number - 1)
        rendered_width = math.ceil(page.rect.width * dpi / 72)
        rendered_height = math.ceil(page.rect.height * dpi / 72)
        if rendered_width * rendered_height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "A página renderizada excederia o limite de 25 milhões de pixels."
            )
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        image_path = temp_dir / f"page-{page_number:04d}.png"
        page.get_pixmap(matrix=matrix, alpha=False).save(image_path)
    return image_path, f"{upload.name}#page={page_number}"


def run_ocr(
    upload_value: Any,
    page_number: float,
    dpi: float,
    max_length: float,
) -> Iterator[tuple[str, str, str, dict[str, Any] | None, str | None]]:
    """Run one local OCR request and stream status changes to the UI."""
    try:
        upload = _upload_path(upload_value)
        page = int(page_number)
        render_dpi = int(dpi)
        sequence_limit = int(max_length)
    except (TypeError, ValueError) as error:
        yield cuda_status(), f"❌ **{error}**", "", None, None
        return

    if page < 1:
        yield cuda_status(), "❌ **A página deve ser 1 ou maior.**", "", None, None
        return
    if render_dpi not in {150, 200, 250, 300}:
        yield cuda_status(), "❌ **Use 150, 200, 250 ou 300 DPI.**", "", None, None
        return
    if sequence_limit not in {2048, 3072, 4096}:
        yield (
            cuda_status(),
            "❌ **Use limite de sequência entre 2048 e 4096 nesta GPU de 8 GB.**",
            "",
            None,
            None,
        )
        return

    if upload.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        yield cuda_status(), "❌ **Formato não aceito. Use imagem ou PDF.**", "", None, None
        return

    with _INFERENCE_LOCK:
        try:
            if _MODEL is None:
                yield (
                    cuda_status(),
                    "⏳ **Carregando o Unlimited-OCR na GPU…** Na primeira vez isso pode levar alguns minutos.",
                    "",
                    None,
                    None,
                )
            torch, tokenizer, model = _load_model()
            yield (
                cuda_status(),
                "🔎 **Modelo carregado. Executando OCR local na GPU…**",
                "",
                None,
                None,
            )

            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(0)
            with tempfile.TemporaryDirectory(prefix="unlimited_ocr_web_") as temp:
                temp_dir = Path(temp)
                image_path, source_label = _prepare_image(
                    upload,
                    page,
                    render_dpi,
                    temp_dir,
                )
                from PIL import Image

                with Image.open(image_path) as image:
                    image_size = image.size
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError(
                            "A imagem excede o limite de 25 milhões de pixels."
                        )

                previous_sliding_window = getattr(
                    model.config,
                    "sliding_window",
                    None,
                )
                try:
                    with torch.inference_mode():
                        raw_output = model.infer(
                            tokenizer,
                            prompt=PROMPT,
                            image_file=str(image_path),
                            output_path=str(temp_dir / "model-artifacts"),
                            max_length=sequence_limit,
                            no_repeat_ngram_size=35,
                            ngram_window=128,
                            save_results=False,
                            eval_mode=True,
                            base_size=1024,
                            image_size=1024,
                            crop_mode=False,
                        )
                finally:
                    model.config.sliding_window = previous_sliding_window

            if not isinstance(raw_output, str):
                raise RuntimeError("O modelo não retornou texto.")

            document = parse_ocr_output(
                raw_output,
                source=source_label,
                image_sizes=[image_size],
                include_raw=False,
                coordinate_max=999,
            )
            compact_output = re.sub(r"\s+", " ", raw_output).strip()
            suspicious = response_is_suspicious(raw_output, 1)
            short_output = len(compact_output) < 80
            elapsed = time.perf_counter() - started
            peak_mib = torch.cuda.max_memory_allocated(0) / 1024**2

            run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            base_name = _safe_stem(upload)
            raw_path = run_dir / f"{base_name}.raw.txt"
            json_path = run_dir / f"{base_name}.json"
            raw_path.write_text(raw_output, encoding="utf-8")
            write_json(document, json_path)

            torch.cuda.empty_cache()
            if suspicious:
                warning = (
                    "⚠️ **Concluído, mas a estrutura parece repetitiva ou truncada; revise o resultado.**"
                )
            elif short_output:
                warning = (
                    "✅ **OCR concluído. A resposta é curta, como esperado para imagens com pouco texto.**"
                )
            else:
                warning = "✅ **OCR concluído sem alerta automático de estrutura.**"
            status = (
                f"{warning}\n\nTempo: **{elapsed:.0f} s** · pico CUDA: "
                f"**{peak_mib:.0f} MiB** · modo: **base**"
            )
            yield cuda_status(), status, raw_output, document, str(json_path)
        except RuntimeError as error:
            if _TORCH is not None and "out of memory" in str(error).lower():
                _TORCH.cuda.empty_cache()
                message = (
                    "A GPU ficou sem memória. Feche aplicativos que usam a GPU e tente novamente."
                )
            else:
                message = str(error)
            yield cuda_status(), f"❌ **Falha na execução:** {message}", "", None, None
        except Exception as error:
            yield cuda_status(), f"❌ **Falha na execução:** {error}", "", None, None


def unload_model() -> tuple[str, str]:
    """Release model memory without stopping the web page."""
    global _MODEL, _TOKENIZER, _TORCH
    with _INFERENCE_LOCK:
        torch = _TORCH
        _MODEL = None
        _TOKENIZER = None
        if torch is not None:
            torch.cuda.empty_cache()
        _TORCH = None
        _trim_linux_memory()
    return cuda_status(), "GPU liberada. O modelo será carregado novamente no próximo OCR."


def build_demo() -> Any:
    """Create the intentionally small, Hugging Face-style interface."""
    import gradio as gr

    with gr.Blocks(title="Unlimited-OCR · CUDA local") as demo:
        gr.Markdown(
            """
            # Unlimited‑OCR
            Envie uma imagem ou escolha uma página de PDF. O arquivo fica neste computador e o modelo roda na sua NVIDIA via CUDA.
            """,
            elem_classes="hero",
        )
        cuda_box = gr.Markdown(cuda_status(), elem_classes="cuda-card")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                upload = gr.File(
                    label="Imagem ou PDF",
                    file_types=["image", ".pdf"],
                    type="filepath",
                )
                gr.Examples(
                    examples=[
                        [str(REPOSITORY_ROOT / "assets" / "baidu.png")],
                        [str(REPOSITORY_ROOT / "assets" / "Unlimited-OCR.png")],
                    ],
                    inputs=[upload],
                    label="Exemplos rápidos",
                )
                with gr.Accordion("Opções", open=False):
                    page = gr.Number(label="Página do PDF", value=1, precision=0, minimum=1)
                    dpi = gr.Slider(label="DPI do PDF", minimum=150, maximum=300, step=50, value=DEFAULT_DPI)
                    max_length = gr.Slider(
                        label="Limite de sequência",
                        minimum=2048,
                        maximum=4096,
                        step=1024,
                        value=DEFAULT_MAX_LENGTH,
                    )
                run_button = gr.Button("Executar OCR na GPU", variant="primary", elem_classes="run-button")
                unload_button = gr.Button("Liberar memória da GPU", variant="secondary")

            with gr.Column(scale=7):
                status = gr.Markdown("Aguardando um arquivo.")
                with gr.Tabs():
                    with gr.Tab("Texto bruto"):
                        raw = gr.Code(label=None, language=None, lines=22, elem_classes="result-code")
                    with gr.Tab("JSON"):
                        json_output = gr.JSON(label=None, height=480)
                json_download = gr.File(label="Baixar JSON")

        run_button.click(
            fn=run_ocr,
            inputs=[upload, page, dpi, max_length],
            outputs=[cuda_box, status, raw, json_output, json_download],
            concurrency_limit=1,
            show_progress="full",
        )
        unload_button.click(
            fn=unload_model,
            outputs=[cuda_box, status],
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    import gradio as gr

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app = build_demo()
    app.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        max_file_size=MAX_UPLOAD_BYTES,
        allowed_paths=[str(OUTPUT_ROOT)],
        theme=gr.themes.Soft(primary_hue="lime", neutral_hue="slate"),
        css=APP_CSS,
    )
