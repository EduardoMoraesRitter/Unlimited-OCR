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
from dataclasses import dataclass
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
PREVIEW_DPI = 120
PREVIEW_MAX_SIDE = 1600
PREVIEW_MAX_PIXELS = 3_000_000
APP_CSS = """
:root { --cuda: #76b900; --cuda-dark: #263d0c; }
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  font-family: Aptos, 'Segoe UI Variable', 'Trebuchet MS', sans-serif !important;
}
.hero { padding: 24px 4px 8px; }
.hero .eyebrow {
  color: var(--cuda);
  font-family: Bahnschrift, 'Arial Narrow', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.hero h1 {
  margin: 7px 0 5px;
  font-family: Bahnschrift, 'Arial Narrow', sans-serif;
  font-size: clamp(2.2rem, 5vw, 4rem);
  font-weight: 650;
  letter-spacing: -0.055em;
  line-height: 0.95;
}
.hero p { max-width: 700px; margin: 0; opacity: 0.72; }
.cuda-card {
  margin: 8px 0 16px;
  padding: 4px 0 10px !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.cuda-card > div { border: 0 !important; background: transparent !important; }
.cuda-card p { margin: 0 !important; }
.work-panel {
  padding: 16px !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 18px !important;
  background: var(--block-background-fill) !important;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.08);
}
.panel-title h3 {
  margin: 0 0 10px;
  font-family: Bahnschrift, 'Arial Narrow', sans-serif;
  letter-spacing: -0.02em;
}
.preview-frame {
  min-height: 360px;
  border-radius: 14px !important;
  overflow: hidden;
}
.preview-frame img { object-fit: contain !important; }
.preview-note { min-height: 28px; font-size: 0.88rem; opacity: 0.76; }
.run-button {
  background: var(--cuda) !important;
  border-color: var(--cuda) !important;
  color: #10170d !important;
  font-weight: 750 !important;
}
.run-button:hover { filter: brightness(1.08); transform: translateY(-1px); }
.result-code textarea, .result-code pre {
  font-family: 'Cascadia Code', 'Cascadia Mono', Consolas, monospace !important;
}
footer { display: none !important; }
"""

_MODEL: Any | None = None
_TOKENIZER: Any | None = None
_TORCH: Any | None = None
_INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreviewPayload:
    """Rendered preview plus page metadata needed by the Gradio controls."""

    image: Any
    note: str
    page_count: int
    is_pdf: bool


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
    if isinstance(value, (str, os.PathLike)):
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


def _integer_control(value: Any, label: str) -> int:
    """Return an exact finite integer from a UI/API control value."""
    if isinstance(value, bool):
        raise ValueError(f"{label} precisa ser um número inteiro.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} precisa ser um número inteiro.") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} precisa ser um número inteiro.")
    return int(number)


def _page_and_dpi(page_number: float, dpi: float) -> tuple[int, int]:
    """Validate the page-preview controls independently from Gradio."""
    page = _integer_control(page_number, "A página")
    render_dpi = _integer_control(dpi, "O DPI")
    if page < 1:
        raise ValueError("A página deve ser 1 ou maior.")
    if render_dpi not in {150, 200, 250, 300}:
        raise ValueError("Use 150, 200, 250 ou 300 DPI.")
    return page, render_dpi


def _check_pixel_limit(width: int, height: int) -> None:
    """Reject invalid or unexpectedly large raster dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("O documento possui dimensões inválidas.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("A imagem excede o limite de 25 milhões de pixels.")


def _build_preview(
    upload_value: Any,
    page_number: float,
    dpi: float,
) -> PreviewPayload:
    """Render a bounded preview without importing Gradio, Torch, or the model."""
    upload = _upload_path(upload_value)
    page, render_dpi = _page_and_dpi(page_number, dpi)
    extension = upload.suffix.lower()
    if extension not in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }:
        raise ValueError("Formato não aceito. Use imagem ou PDF.")

    from PIL import Image, ImageOps

    if extension != ".pdf":
        with Image.open(upload) as image:
            width, height = image.size
            _check_pixel_limit(width, height)
            preview = ImageOps.exif_transpose(image).convert("RGB")
        preview.thumbnail(
            (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE),
            Image.Resampling.LANCZOS,
        )
        note = (
            "👁️ **Prévia pronta** · imagem · "
            f"original `{width} × {height}` · prévia `{preview.width} × {preview.height}`"
        )
        return PreviewPayload(preview, note, 1, False)

    import fitz

    with fitz.open(upload) as document:
        if document.needs_pass:
            raise ValueError("PDF protegido por senha não é aceito nesta tela.")
        if document.page_count == 0:
            raise ValueError("O PDF não contém páginas.")
        if page > document.page_count:
            raise ValueError(
                f"Escolha uma página entre 1 e {document.page_count}."
            )
        pdf_page = document.load_page(page - 1)
        width_points = float(pdf_page.rect.width)
        height_points = float(pdf_page.rect.height)
        if not all(
            math.isfinite(value) and value > 0
            for value in (width_points, height_points)
        ):
            raise ValueError("A página do PDF possui dimensões inválidas.")
        ocr_width = math.ceil(width_points * render_dpi / 72)
        ocr_height = math.ceil(height_points * render_dpi / 72)
        _check_pixel_limit(ocr_width, ocr_height)

        preview_scale = min(
            PREVIEW_DPI / 72,
            PREVIEW_MAX_SIDE / max(width_points, height_points),
            math.sqrt(PREVIEW_MAX_PIXELS / (width_points * height_points)),
        )
        matrix = fitz.Matrix(preview_scale, preview_scale)
        pixmap = pdf_page.get_pixmap(
            matrix=matrix,
            colorspace=fitz.csRGB,
            alpha=False,
        )
        preview = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        note = (
            f"👁️ **Prévia pronta** · PDF · página **{page} de {document.page_count}** · "
            f"prévia `{preview.width} × {preview.height}` · "
            f"OCR `{ocr_width} × {ocr_height}` a **{render_dpi} DPI**"
        )
        return PreviewPayload(preview, note, document.page_count, True)


def preview_document(
    upload_value: Any,
    page_number: float,
    dpi: float,
) -> tuple[Any | None, str]:
    """Safe Gradio adapter for page/DPI changes."""
    if not upload_value:
        return None, "Selecione uma imagem ou PDF para visualizar."
    try:
        payload = _build_preview(upload_value, page_number, dpi)
        return payload.image, payload.note
    except Exception as error:
        return None, f"❌ **Não foi possível gerar a prévia:** {error}"


def preview_new_upload(
    upload_value: Any,
    dpi: float,
) -> tuple[Any | None, str, Any]:
    """Preview page one and update the PDF page control bounds."""
    import gradio as gr

    if not upload_value:
        return (
            None,
            "Selecione uma imagem ou PDF para visualizar.",
            gr.update(value=1, minimum=1, maximum=1, interactive=False),
        )
    try:
        payload = _build_preview(upload_value, 1, dpi)
        page_update = gr.update(
            value=1,
            minimum=1,
            maximum=payload.page_count,
            interactive=payload.is_pdf,
        )
        return payload.image, payload.note, page_update
    except Exception as error:
        return (
            None,
            f"❌ **Não foi possível gerar a prévia:** {error}",
            gr.update(value=1, minimum=1, maximum=1, interactive=False),
        )


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
        if document.needs_pass:
            raise ValueError("PDF protegido por senha não é aceito nesta tela.")
        if document.page_count == 0:
            raise ValueError("O PDF não contém páginas.")
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(
                f"Escolha uma página entre 1 e {document.page_count}."
            )
        page = document.load_page(page_number - 1)
        rendered_width = math.ceil(page.rect.width * dpi / 72)
        rendered_height = math.ceil(page.rect.height * dpi / 72)
        _check_pixel_limit(rendered_width, rendered_height)
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
        page, render_dpi = _page_and_dpi(page_number, dpi)
        sequence_limit = _integer_control(max_length, "O limite de sequência")
    except (TypeError, ValueError) as error:
        yield cuda_status(), f"❌ **{error}**", "", None, None
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

    safe_name = f"{_safe_stem(upload)}{upload.suffix.lower()}"
    if upload.suffix.lower() == ".pdf":
        request_description = (
            f"`{safe_name}` · página **{page}** · **{render_dpi} DPI**"
        )
    else:
        request_description = f"`{safe_name}` · imagem"

    with _INFERENCE_LOCK:
        try:
            if _MODEL is None:
                yield (
                    cuda_status(),
                    "⏳ **Carregando o Unlimited-OCR na GPU…** Na primeira vez isso pode levar alguns minutos."
                    f"\n\nEntrada fixada para esta execução: {request_description}.",
                    "",
                    None,
                    None,
                )
            torch, tokenizer, model = _load_model()
            yield (
                cuda_status(),
                "🔎 **Modelo carregado. Executando OCR local na GPU…**"
                f"\n\nEntrada fixada para esta execução: {request_description}.",
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
                    _check_pixel_limit(image.width, image.height)

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
                f"\n\nEntrada processada: {request_description}."
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
            yield (
                cuda_status(),
                f"❌ **Falha na execução:** {message}\n\nEntrada: {request_description}.",
                "",
                None,
                None,
            )
        except Exception as error:
            yield (
                cuda_status(),
                f"❌ **Falha na execução:** {error}\n\nEntrada: {request_description}.",
                "",
                None,
                None,
            )


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
            <div class="hero">
              <span class="eyebrow">Local CUDA document workbench</span>
              <h1>Unlimited‑OCR</h1>
              <p>Veja a página antes de processar. O arquivo fica neste computador e o modelo roda na sua NVIDIA.</p>
            </div>
            """,
        )
        cuda_box = gr.Markdown(cuda_status(), elem_classes="cuda-card")

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes="work-panel"):
                gr.Markdown("### Documento", elem_classes="panel-title")
                upload = gr.File(
                    label="Imagem ou PDF",
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"],
                    type="filepath",
                )
                gr.Examples(
                    examples=[
                        [str(REPOSITORY_ROOT / "assets" / "baidu.png")],
                        [str(REPOSITORY_ROOT / "assets" / "Unlimited-OCR.png")],
                        [str(REPOSITORY_ROOT / "Unlimited-OCR.pdf")],
                    ],
                    inputs=[upload],
                    label="Exemplos rápidos",
                )
                with gr.Row():
                    page = gr.Number(
                        label="Página do PDF",
                        value=1,
                        precision=0,
                        minimum=1,
                        maximum=1,
                        interactive=False,
                    )
                    dpi = gr.Dropdown(
                        label="Qualidade do PDF",
                        choices=[150, 200, 250, 300],
                        value=DEFAULT_DPI,
                    )
                preview_note = gr.Markdown(
                    "Selecione uma imagem ou PDF para visualizar.",
                    elem_classes="preview-note",
                )
                preview = gr.Image(
                    label="Prévia da página selecionada",
                    interactive=False,
                    type="pil",
                    height=480,
                    format="png",
                    buttons=["fullscreen"],
                    elem_classes="preview-frame",
                )
                with gr.Accordion("Opções avançadas", open=False):
                    max_length = gr.Slider(
                        label="Limite de sequência",
                        minimum=2048,
                        maximum=4096,
                        step=1024,
                        value=DEFAULT_MAX_LENGTH,
                    )
                with gr.Row():
                    run_button = gr.Button(
                        "Executar OCR na GPU",
                        variant="primary",
                        elem_classes="run-button",
                    )
                    unload_button = gr.Button(
                        "Liberar GPU",
                        variant="secondary",
                    )

            with gr.Column(scale=6, elem_classes="work-panel"):
                gr.Markdown("### Resultado", elem_classes="panel-title")
                status = gr.Markdown("Aguardando um arquivo.")
                with gr.Tabs():
                    with gr.Tab("Texto bruto"):
                        raw = gr.Code(
                            label=None,
                            language=None,
                            lines=24,
                            elem_classes="result-code",
                        )
                    with gr.Tab("JSON"):
                        json_output = gr.JSON(label=None, height=540)
                json_download = gr.File(label="Baixar JSON")

        upload.change(
            fn=preview_new_upload,
            inputs=[upload, dpi],
            outputs=[preview, preview_note, page],
            concurrency_limit=1,
            concurrency_id="preview",
            trigger_mode="always_last",
            show_progress="minimal",
        )
        page.input(
            fn=preview_document,
            inputs=[upload, page, dpi],
            outputs=[preview, preview_note],
            concurrency_limit=1,
            concurrency_id="preview",
            trigger_mode="always_last",
            show_progress="minimal",
        )
        dpi.input(
            fn=preview_document,
            inputs=[upload, page, dpi],
            outputs=[preview, preview_note],
            concurrency_limit=1,
            concurrency_id="preview",
            trigger_mode="always_last",
            show_progress="minimal",
        )

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
        show_error=False,
        max_file_size=MAX_UPLOAD_BYTES,
        allowed_paths=[str(OUTPUT_ROOT)],
        theme=gr.themes.Soft(primary_hue="lime", neutral_hue="slate"),
        css=APP_CSS,
    )
