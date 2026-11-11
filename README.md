# COMP5933 Phase 1: Zero-shot Physical Dynamics from Video

This repository is a minimal, reproducible Phase 1 pipeline for classifying IRIS videos with
`Qwen/Qwen2.5-VL-7B-Instruct`. It performs inference only: there is no training or fine-tuning.

The closed label set is fixed as:

| Choice | Label |
|---|---|
| A | Dropping Ball |
| B | Pendulum |
| C | Sliding Cone |
| D | Hitting Cones |

## Repository structure

```text
iris_phase1/              dataset, sampling, prompting, model, and pipeline code
scripts/inspect_iris.py   structure-agnostic recursive discovery + manifest template
scripts/smoke_test.py     exactly one video, at both 1 and 8 frames
scripts/run_zero_shot.py  manifest-based experiment runner
scripts/evaluate.py       accuracy, macro-F1, per-class accuracy, confusion matrix
requirements.txt
```

## 1. GPU environment

The project deliberately does **not** install PyTorch. Reuse the CUDA-enabled PyTorch supplied by
the server administrators so that `pip` cannot silently replace it with a different CUDA or CPU
build. First verify the server environment **before creating the project virtual environment**:

```bash
nvidia-smi
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("built for CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch is unavailable; ask the server administrator which module/environment to load.")
print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU/PyTorch environment does not support BF16.")
PY
```

If that check succeeds, create an isolated environment that can see the verified server PyTorch,
then install only this project's non-PyTorch dependencies:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Pin dependency resolution to the exact CUDA PyTorch already visible here.
# If another dependency is incompatible, pip must fail instead of replacing it.
COMP5933_TORCH_VERSION="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
COMP5933_CONSTRAINT_FILE="$(mktemp)"
printf 'torch==%s\n' "$COMP5933_TORCH_VERSION" > "$COMP5933_CONSTRAINT_FILE"
python -m pip install --constraint "$COMP5933_CONSTRAINT_FILE" -r requirements.txt
python -m pip install --no-deps -e .

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__, torch.__file__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())
PY
```

Do not run a generic `pip install torch` on the company server. If the initial check cannot import
PyTorch, load the server's documented CUDA/PyTorch module or activate its approved Conda environment
first, then repeat the check. An A100 80 GB supports BF16 when paired with a compatible PyTorch/CUDA
build.

Model weights are downloaded by Hugging Face on first use; set `HF_HOME` to a large
non-repository filesystem. If authentication is required, use `huggingface-cli login` or provide
`HF_TOKEN` in the shell environment—never commit it.

Run commands from the repository root so Python can import `iris_phase1`.

## 2. Inspect IRIS and make a manifest

No IRIS directory layout or directory-to-label mapping is assumed. First inspect recursively:

```bash
python scripts/inspect_iris.py \
  --dataset-root /absolute/path/to/IRIS \
  --show 50 \
  --write-manifest /absolute/path/to/experiment/iris_manifest.csv
```

The generated CSV has `video_id,video_path,label`. Paths are relative to `--dataset-root`.
Inspect IRIS's supplied annotations/metadata and fill `label` with exactly one of `Dropping Ball`,
`Pendulum`, `Sliding Cone`, or `Hitting Cones`. Do not infer ground truth from a directory name
without verifying the benchmark documentation. A hand-written manifest may instead use absolute
paths; `video_id` is optional and defaults to the file stem.

Example:

```csv
video_id,video_path,label
example_001,clips/example_001.mp4,Dropping Ball
```

## 3. Exact one-video smoke test

Choose one video whose ground-truth label has been verified, then run:

```bash
mkdir -p /absolute/path/to/experiment
export HF_HOME=/absolute/path/on/large/storage/huggingface
export CUDA_VISIBLE_DEVICES=0
python scripts/smoke_test.py \
  --video /absolute/path/to/IRIS/example.mp4 \
  --label "Dropping Ball" \
  --output /absolute/path/to/experiment/smoke_results.csv
```

This loads Qwen in BF16, runs deterministic greedy generation, and writes two rows: one with one
frame and one with eight ordered frames. It does not claim a successful GPU test until you execute
this command on the server. Inspect `raw_response`, `valid_prediction`, and `error` in the CSV.

## 4. Full manifest inference

```bash
python scripts/run_zero_shot.py \
  --manifest /absolute/path/to/experiment/iris_manifest.csv \
  --dataset-root /absolute/path/to/IRIS \
  --frame-counts 1 8 \
  --output /absolute/path/to/experiment/iris_qwen_results.csv
```

Use `--limit 1` for a manifest-based dry run. Results are flushed after every row and contain video
ID/path, ground truth, parsed choice/label, validity, requested frame count, sampled timestamps,
the exact prompt, model ID, raw response, and errors. An unreadable/corrupt video creates an error
row rather than stopping the complete experiment.

Sampling sequentially decodes the video once to count actually readable frames and a second time
to extract the temporal center for one frame or `round(linspace(0, N-1, 8))` for eight frames.
Order is preserved. Very short videos repeat frames deterministically. Timestamps are frame index
divided by reported FPS; when FPS is unavailable they are stored as JSON `null`, while sampling
still works from frame indices.

## 5. Evaluate

```bash
python scripts/evaluate.py \
  --results /absolute/path/to/experiment/iris_qwen_results.csv \
  --output-json /absolute/path/to/experiment/metrics.json
```

Metrics are reported separately for each frame count. The confusion-matrix row/column order is the
four-label order at the top of this README. Invalid/error predictions count as wrong in accuracy,
macro-F1, and per-class accuracy. Because they have no closed-set prediction column, they do not
appear in the confusion matrix and are also reported as `invalid_rows`.

## Reproducibility and storage

- Decoding and temporal indices are deterministic; generation uses `do_sample=False`.
- Outputs are configurable and should live outside the Git checkout.
- `.gitignore` excludes common datasets, videos, weights, frames, logs, secrets, and outputs.
- Keep the manifest if it contains only paths/labels and is safe to share; do not commit private or
  machine-specific paths.
- `requirements.txt` excludes PyTorch so dependency installation cannot replace the server's
  CUDA build. Other direct dependencies use bounded major-version ranges.
- For a final experiment, archive `python -m pip freeze` and record the GPU/driver shown by
  `nvidia-smi` outside Git.

## Local checks (no model download)

Syntax-check all modules without loading Qwen weights:

```bash
python -m compileall -q iris_phase1 scripts
```

Actual model inference requires the CUDA/BF16 server and is intentionally not faked.
