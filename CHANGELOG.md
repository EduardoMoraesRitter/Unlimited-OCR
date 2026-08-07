# Change log

This file describes the community additions made on top of
[`baidu/Unlimited-OCR`](https://github.com/baidu/Unlimited-OCR). The upstream
model, paper, assets, and original inference documentation remain credited to
Baidu under the repository's MIT license.

## 2026-08-06 — Structured JSON and difficult-document workflows

### Deterministic JSON output

- Added `unlimited_ocr_json.py`, a standard-library post-processor for native
  Unlimited-OCR output.
- Supports inline `<|det|>` markers, combined `<|ref|>` and `<|det|>` markers,
  `<PAGE>` boundaries, ungrounded text, HTML tables, and multiple bounding boxes.
- Preserves normalized coordinates and calculates pixel coordinates when page
  dimensions are known.
- Retains raw model output by default for auditability, with an option to omit it.
- Reports structured warnings for empty output, malformed or out-of-range boxes,
  inverted boxes, unbalanced markers, empty blocks, and suspicious repetition.
- Uses `ast.literal_eval` rather than `eval` when decoding coordinates.

### SGLang integration

- Added `--json` to `infer.py`, producing a JSON companion for every Markdown
  result.
- JSON conversion failures no longer repeat an otherwise successful and costly
  OCR request; the runner records the error and exits with a failure status.
- PDF renderings now use a managed temporary directory that is removed after
  processing.
- Image output paths retain their source extension and directory hierarchy,
  preventing collisions such as `scan.png` and `scan.jpg` overwriting each other.

### Examples

- `examples/parse_output_to_json.py`: converts a saved raw response without a
  model, network connection, or GPU.
- `examples/transformers_image_to_json.py`: performs local OCR for one image,
  preserves the raw response, and writes structured JSON.
- `examples/difficult_pdf_to_json.py`: renders a PDF at 300 DPI and processes one
  page at a time, with encrypted-PDF support, conservative GPU usage, raw output
  per page, and aggregate document JSON.
- The PDF `auto` mode starts with `base`, retries suspicious responses with
  `gundam`, and avoids the higher-memory retry after a CUDA out-of-memory error.
- Added a Portuguese usage guide and a synthetic Portuguese OCR response with an
  HTML table.

### Validation and public-repository safety

- Added offline unit coverage for marker variants, pages, tables, pixel and
  multi-region boxes, malformed input, repetition, UTF-8 JSON, and avoidance of
  executable coordinate parsing.
- Added runner tests for distinct output names, nested output directories,
  managed PDF temporary files, automatic retry warnings, and JSON failure status.
- The validation suite contains 18 tests and can run without model weights.
- Added GitHub Actions CI for pushes and pull requests using read-only
  permissions and commit-pinned official actions.
- Expanded `.gitignore` coverage for `.env` files, credentials, generated OCR,
  logs, caches, checkpoints, and model weights.

Full CUDA inference is environment-dependent and is intentionally separate from
the offline test suite because it requires the model weights and compatible GPU
dependencies.
