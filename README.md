<p align="center">
  <img src="assets/baidu.png" width="40%" alt="Baidu Inc." />
</p>

<hr>

<h1 align="center">Unlimited OCR Works</h1>

<div align="center">

<a href="https://trendshift.io/repositories/62053?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-62053" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/62053/daily" alt="baidu%2FUnlimited-OCR | Trendshift" width="250" height="55"/></a>
  
  <a href="https://github.com/baidu/Unlimited-OCR">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" />
  </a>
  <a href="https://huggingface.co/baidu/Unlimited-OCR">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2606.23050">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Unlimited OCR Works-b31b1b?logo=arxiv&logoColor=white" />
  </a>
  <a href="https://x.com/Baidu_Inc" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Baidu Inc.-white?logo=x&logoColor=white" />
  </a>
</div>

<div align="center">
  <a href="https://github.com/EduardoMoraesRitter/Unlimited-OCR/actions/workflows/tests.yml">
    <img alt="Offline tests" src="https://github.com/EduardoMoraesRitter/Unlimited-OCR/actions/workflows/tests.yml/badge.svg" />
  </a>
</div>

<h3 align="center">Welcome the Era of One-shot Long-horizon Parsing.</h3>

<p align="center">
    <img src="assets/Unlimited-OCR.png" width="1000" alt="Unlimited OCR overview" />
</p>

> [!NOTE]
> This community fork extends Baidu's
> [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) with deterministic
> JSON export, production-oriented PDF examples, and offline validation. It is
> not an official Baidu release.

## What this fork adds

- Deterministic conversion of native OCR grounding markers to auditable JSON.
- Page, layout-block, HTML-table, normalized-coordinate, pixel-coordinate, and
  multi-region bounding-box support.
- `infer.py --json` integration for SGLang image and PDF batches while retaining
  the original Markdown response.
- A sequential 300-DPI workflow for difficult PDFs with bounded GPU use,
  suspicious-output detection, guarded `base` to `gundam` retries, raw output
  per page, and aggregate JSON.
- Runnable examples for offline conversion, single-image Transformers inference,
  and difficult PDFs.
- A small localhost-only Gradio interface for image/PDF upload, live CUDA status,
  raw model output, structured JSON, and explicit GPU-memory release.
- Offline unit tests covering parsing, safety, temporary-file cleanup, output
  collisions, retry detection, and JSON-export failures.
- GitHub Actions CI that runs the offline suite for every push and pull request,
  with read-only repository permissions and commit-pinned actions.
- Safer defaults for public development: generated OCR, model weights, local
  secrets, credentials, caches, and logs are excluded from Git.

Quick smoke test without downloading or running the OCR model:

```shell
python examples/parse_output_to_json.py
python -m unittest discover -s tests -v
```

To use the actual model through a local web page after completing the CUDA
setup in [`examples/README.md`](examples/README.md):

```shell
python -m pip install gradio==6.15.1 accelerate==1.14.0 huggingface-hub==0.36.0
python examples/gradio_cuda_app.py
```

Then open <http://127.0.0.1:7860>. The server binds only to localhost and does
not create a public Gradio share link.

See [the complete change log](CHANGELOG.md), the
[structured JSON guide](#structured-json-output), and the
[Portuguese examples guide](examples/README.md).

## Test status and what CUDA means

The repository has two separate layers: Baidu's neural OCR model and this
fork's deterministic post-processing/tooling. A passing offline test suite is
evidence for the second layer only; it is not an OCR-accuracy benchmark.

| Component | Where it runs | CUDA required? | Current validation status |
| --- | --- | --- | --- |
| JSON parser and validator | Local Python | No | 18 offline tests cover markers, pages, tables, coordinates, warnings, safe parsing, file handling, and runner failures. |
| PDF rendering and page orchestration | Local Python | No for rendering; yes for the OCR step | Rendering, retry selection, cleanup, and error paths are covered offline with fixtures and mocks. |
| Unlimited-OCR through the official Hugging Face Space | Hugging Face infrastructure | Not on the user's computer; the provider supplies the GPU | Used only for exploratory inference. This is not presented as a reproducible quality benchmark. |
| Full local Unlimited-OCR inference | Local machine | Yes for the upstream Transformers path used here | Manually smoke-tested on an RTX 4070 Laptop GPU through the local web UI; still excluded from CI and not an accuracy benchmark. |
| Difficult/low-resolution PDF accuracy | Local or hosted GPU | Yes somewhere during model inference | One dense-form smoke test completed technically but produced repetitive/truncated text. Measured CER/WER results have not yet been published. |

### CUDA is an upstream runtime requirement

CUDA was not added by this fork. The upstream Transformers example loads the
model with `model.eval().cuda()`, so actual local OCR uses a compatible NVIDIA
GPU, CUDA-enabled PyTorch, and the model weights. CUDA is not needed to render a
PDF, parse a saved model response, create JSON, or run the offline test suite.

There are three materially different ways to use this repository:

| Workflow | Runs the OCR model? | File leaves the machine? | Practical consequence |
| --- | --- | --- | --- |
| Offline JSON conversion | No | No | Fast and CPU-only, but it requires a previously generated raw OCR response. |
| Local Transformers inference | Yes, on the local NVIDIA GPU | No, after model/code download | Requires a working CUDA PyTorch environment; 8 GB GPUs are marginal. |
| Hosted Hugging Face demo | Yes, on the provider's GPU | Yes | No local CUDA setup, but documents are uploaded to a third party and service limits apply. |

This fork does not claim a validated CPU fallback for the full model. See
[`examples/README.md`](examples/README.md) for a CUDA verification command and
the WSL/Linux setup used by the examples.

### Local CUDA smoke-test evidence

On 2026-08-06, the localhost web UI ran the pinned model revision
`07dea832e22aefee32ad281d4b80551282e1c168` with PyTorch 2.10.0+cu129 on an
NVIDIA GeForce RTX 4070 Laptop GPU. In `base` mode with `max_length=4096`, the
included `assets/baidu.png` example returned `Baidu 百度`, produced valid schema
1.0 JSON, took 92 seconds, and reported 6,835 MiB peak PyTorch CUDA allocation.

A 200-DPI first page of the official
[2025 IRS Form 1040](https://www.irs.gov/pub/irs-pdf/f1040.pdf) also completed
locally in 388 seconds with a 7,039 MiB peak, but its output entered a repeated
phrase loop and ended with an unclosed table. That run proves the CUDA pipeline
executes; it does **not** prove acceptable OCR quality on dense forms. The web
UI shows a review warning when it detects this kind of malformed or repetitive
response.


## Release
- [2026/07/21] 🤝 Thanks to the [ms-swift community](https://github.com/modelscope/ms-swift) for their support, our model now supports training with [ms-swift](https://github.com/modelscope/ms-swift).
- [2026/07/03] 🤝 Thanks to the Baidu Cloud team for their support. Our model is now available on [Baidu Cloud](https://cloud.baidu.com/doc/OCR/s/fmr1p39gb).
- [2026/06/28] 🤝 Thanks to the [vLLM community](https://github.com/vllm-project/vllm) and [Tianyu Guo](https://github.com/gty111) for their support, our model now supports vLLM inference.
- [2026/06/24] 🤝 Thanks to [AK](https://x.com/_akhaliq) for creating a demo for us. It is now available at [Hugging Face Spaces](https://huggingface.co/spaces/baidu/Unlimited-OCR).
- [2026/06/23] 📄 Our paper is now available on [arXiv](https://arxiv.org/abs/2606.23050).
- [2026/06/23] 🤝 Thanks to the [ModelScope community](https://github.com/modelscope) for their support. Our model is now available at [ModelScope](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR).
- [2026/06/22] 🚀 We present [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), aiming to push [Deepseek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) one step further.

## Inference

### Transformers
Inference using Huggingface transformers on NVIDIA GPUs. Requirements tested on python 3.12.3 + CUDA12.9：

> [!IMPORTANT]
> This section runs the actual OCR model and therefore differs from the
> CPU-only JSON smoke test. Confirm that `torch.cuda.is_available()` is `True`
> before downloading the model. The `.cuda()` call below comes from the
> upstream inference path; it was not introduced by this fork.

```
torch==2.10.0
torchvision==0.25.0
transformers==4.57.1
Pillow==12.1.1
matplotlib==3.10.8
einops==0.8.2
addict==2.4.0
easydict==1.13
pymupdf==1.27.2.2
psutil==7.2.2
```

```python
import os
import torch
from transformers import AutoModel, AutoTokenizer

model_name = 'baidu/Unlimited-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()

# ── Single image supports two configs: gundam or base ──
# gundam: base_size=1024, image_size=640, crop_mode=True
# base: base_size=1024, image_size=1024, crop_mode=False
model.infer(
    tokenizer,
    prompt='<image>document parsing.',
    image_file='your_image.jpg',
    output_path='your/output/dir',
    base_size=1024, image_size=640, crop_mode=True,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=128,
    save_results=True,
)

# ── Multi page / PDF only uses base (image_size=1024) ──
model.infer_multi(
    tokenizer,
    prompt='<image>Multi page parsing.',
    image_files=['page1.png', 'page2.png', 'page3.png'],
    output_path='your/output/dir',
    image_size=1024,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024,
    save_results=True,
)

# ── PDF (convert pages to images, then multi-page parsing) ──
import tempfile, fitz  # PyMuPDF

def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f'page_{i+1:04d}.png')
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths

model.infer_multi(
    tokenizer,
    prompt='<image>Multi page parsing.',
    image_files=pdf_to_images('your_doc.pdf', dpi=300),
    output_path='your/output/dir',
    image_size=1024,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024,
    save_results=True,
)
```

### vLLM

Please refer to the official vLLM recipe for deployment details:

**Recipe:** [https://recipes.vllm.ai/baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR)

##### Docker Images
Use the following Docker images depending on your GPU platform:

**Default (CUDA 13.0):**
```bash
docker pull vllm/vllm-openai:unlimited-ocr
```
**For Hopper GPUs (CUDA 12.9)**
```bash
docker pull vllm/vllm-openai:unlimited-ocr-cu129
```

### SGLang

Set up the environment (uv-managed virtualenv). Install the local SGLang wheel first,
then pin `kernels==0.9.0` and install PyMuPDF for PDF-to-image conversion:
```shell
uv venv --python 3.12
source .venv/bin/activate

uv pip install wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
uv pip install kernels==0.11.7
uv pip install pymupdf==1.27.2.2
```

Start the SGLang server:
```shell
python -m sglang.launch_server \
    --model baidu/Unlimited-OCR \
    --served-model-name Unlimited-OCR \
    --attention-backend fa3 \
    --page-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --enable-custom-logit-processor \
    --disable-overlap-schedule \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 10000
```

Send streaming requests to the OpenAI-compatible API:
```python
import base64
import json
import os
import tempfile

import fitz
import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

server_url = "http://127.0.0.1:10000"

session = requests.Session()
session.trust_env = False


def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths = []
    for i, page in enumerate(doc):
        image_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(image_path)
        image_paths.append(image_path)
    doc.close()
    return image_paths


def encode_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(prompt, image_paths):
    return [{"type": "text", "text": prompt}] + [encode_image(path) for path in image_paths]


def generate(prompt, image_paths, image_mode, ngram_window):
    payload = {
        "model": "Unlimited-OCR",
        "messages": [{"role": "user", "content": build_content(prompt, image_paths)}],
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"image_mode": image_mode},
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
        "custom_params": {
            "ngram_size": 35,
            "window_size": ngram_window,
        },
        "stream": True,
    }
    response = session.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=1200,
        stream=True,
    )
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        event = json.loads(data)
        delta = event["choices"][0].get("delta", {}).get("content", "")
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


# Single image supports two configs: gundam or base. Example below uses gundam.
generate("document parsing.", ["your_image.jpg"], image_mode="gundam", ngram_window=128)

# Multi image (base only)
generate("Multi page parsing.", ["page1.png", "page2.png"], image_mode="base", ngram_window=1024)

# PDF (base only)
generate("Multi page parsing.", pdf_to_images("your_doc.pdf", dpi=300), image_mode="base", ngram_window=1024)
```

For batch inference, `infer.py` starts the SGLang server automatically and sends concurrent requests for an image directory or PDF:
```shell
# Image directory
python infer.py \
    --image_dir /path/to/images \
    --output_dir ./outputs \
    --concurrency 8 \
    --image_mode gundam

# PDF pages
python infer.py \
    --pdf ./Unlimited-OCR.pdf \
    --output_dir ./outputs \
    --concurrency 8 \
    --image_mode gundam
```

Useful options:
```shell
--model_dir baidu/Unlimited-OCR   # Local path or Hugging Face model ID
--gpu 0                           # CUDA_VISIBLE_DEVICES value
--server_log ./log/sglang_server.log
```

For OmniDocBench evaluation, you need to perform the following post-processing.
```python
DET_RE = re.compile(r'<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)', re.DOTALL)

def remove_det(raw: str) -> str:
    """
    Strip <|det|>type [bbox]<|/det|> markers, group lines belonging to the
    same block with \\n, and separate different blocks with \\n\\n.
    """
    blocks = []
    cur = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == 'image':
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    text = '\n\n'.join('\n'.join(b) for b in blocks).strip()
    return text
```

## Structured JSON output

The model is optimized for grounded OCR text, not for generating a JSON schema
directly from a prompt. Its raw response contains layout markers, normalized
bounding boxes, Markdown, and HTML tables. `unlimited_ocr_json.py` converts that
response to deterministic JSON without asking the model to rewrite its own
output.

The converter uses only the Python standard library and can be tested without a
GPU or model download:

```shell
python unlimited_ocr_json.py examples/raw_output.txt \
    --output outputs/example.json \
    --source example.png \
    --image-size 1400x1000
```

Use `--json` with the SGLang batch runner to create a structured `.json` file
next to every raw `.md` result:

```shell
python infer.py \
    --pdf ./Unlimited-OCR.pdf \
    --output_dir ./outputs \
    --concurrency 1 \
    --image_mode base \
    --json
```

Each JSON document contains pages, ordered layout blocks, source coordinates,
pixel coordinates when the image size is known, parsed HTML table rows, the raw
model output, and validation warnings. Keeping the raw response makes it
possible to audit or reprocess a result later. When one block has several
regions, `bboxes` preserves every box and `bbox` contains their outer envelope.

### Examples

| Example | Purpose |
| --- | --- |
| [`examples/parse_output_to_json.py`](examples/parse_output_to_json.py) | Convert a saved raw response without loading the model. |
| [`examples/transformers_image_to_json.py`](examples/transformers_image_to_json.py) | Run one image with Transformers and preserve layout in JSON. |
| [`examples/difficult_pdf_to_json.py`](examples/difficult_pdf_to_json.py) | Render and process a difficult PDF page by page with guarded retries. |

See [`examples/README.md`](examples/README.md) for complete commands.

### Difficult PDFs and low-VRAM GPUs

For scanned, dense, rotated, or table-heavy PDFs:

1. Render pages at 300 DPI and process them independently so one bad page does
   not invalidate the whole document.
2. Start with `base` mode. Retry only suspicious pages with `gundam`, which can
   recover smaller text but uses more visual tokens and GPU memory.
3. Treat 8 GB GPUs as marginal: start with `base`, use `--concurrency 1`, and
   keep a bounded generation length. Even then, some pages may not fit. Increase
   concurrency only after measuring free memory.
4. Inspect the JSON warnings for empty output, invalid boxes, or suspicious
   repetition, and retain each page's raw output for troubleshooting.
5. Run sensitive documents locally instead of uploading them to a hosted demo.

The page-by-page PDF example implements this conservative workflow and still
allows `base`, `gundam`, or automatic retry mode.

## Visualization

<img src="assets/long-horizon-ocr.gif" width="100%" alt="Long-horizon OCR demo" />

## Acknowledgement

We would like to thank [Deepseek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [Deepseek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for their valuable models and ideas.

## Citation
```bibtex
@misc{yin2026unlimitedocrworks,
      title={Unlimited OCR Works}, 
      author={Youyang Yin and Huanhuan Liu and YY and Qunyi Xie and Chaorun Liu and Shiqi Yang and Shaohua Wang and Zhanlong Liu and Hao Zou and Jinyue Chen and Shu Wei and Jingjing Wu and Mingxin Huang and Zhen Wu and Guibin Wang and Tengyu Du and Lei Jia},
      year={2026},
      eprint={2606.23050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.23050}, 
}
