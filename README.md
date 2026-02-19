# LLM Inference Benchmarks on NVIDIA GPUs
**PyTorch (HF Transformers) vs vLLM (optional) vs TensorRT-LLM (optional)**

This repo is a lightweight, reproducible benchmark harness to measure **latency**, **throughput**, and **GPU memory** for LLM text generation on NVIDIA GPUs.

It’s designed to be honest and practical:
- Runs **PyTorch + Hugging Face** by default
- Runs **vLLM** if installed
- Includes a **TensorRT-LLM placeholder** runner (so you can add it when ready)
- Produces a single JSON report you can paste into a resume / writeup

> **Note:** LLM inference performance depends heavily on GPU type, driver/CUDA versions, batch size, prompt length, and generation length.
> This project focuses on making results **reproducible** and **comparable**.

---

## Quick Start (PyTorch baseline)

### 1) Create env and install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Run a benchmark (GPU recommended)
```bash
python benchmarks/run_benchmarks.py --backend pytorch --model distilgpt2 --device cuda
```

This writes:
- `results/latest.json` (full report)
- `results/latest.md` (human-friendly summary)

---

## Optional: vLLM backend
vLLM is optional because it has platform-specific wheels.

```bash
pip install vllm
python benchmarks/run_benchmarks.py --backend vllm --model mistralai/Mistral-7B-Instruct-v0.2 --device cuda
```

---

## TensorRT-LLM (optional)
TensorRT-LLM requires additional setup and engine build steps.
This repo includes a placeholder runner under `backends/tensorrt_llm_runner.py` so you can wire it in once you have engines.

---

## What this measures
- **Prefill + decode latency** (wall-clock)
- **Tokens/sec** (approx throughput)
- **Peak GPU memory** (if CUDA is available)
- Prompt length / generation length / batch size

---

## Suggested “NVIDIA-core” talking points (for interviews)
- How you chose batch sizes and sequence lengths for fair comparisons
- Why KV-cache and attention often dominate decode time
- How you would profile bottlenecks with Nsight Systems/Compute
- The tradeoffs between eager PyTorch, vLLM’s paged attention, and TensorRT-LLM engines

---

## Repo structure
- `benchmarks/run_benchmarks.py` — main CLI
- `benchmarks/configs/*.json` — benchmark presets
- `backends/` — implementations per runtime
- `utils/metrics.py` — timing + tokens/sec + GPU memory helpers
- `results/` — output artifacts

---

## License
MIT
