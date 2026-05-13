# LLM Inference Benchmarks on NVIDIA GPUs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green)
![Backends](https://img.shields.io/badge/Backends-PyTorch%20%7C%20vLLM%20%7C%20TensorRT--LLM-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A lightweight, reproducible benchmark harness measuring **latency, throughput, and GPU memory usage** for LLM text generation across multiple inference backends on NVIDIA GPUs.

Designed to be honest, practical, and production-aware — not just a toy script.

---

## Why This Exists

Most LLM benchmark comparisons are either:
- Tied to a specific cloud provider's marketing numbers
- Run on hardware most engineers don't have access to
- Not reproducible (undocumented batch sizes, prompt lengths, driver versions)

This repo fixes that: every result includes full hardware context, config, and methodology so you can reproduce or extend it.

---

## Backends Supported

| Backend | Status | Notes |
|---|---|---|
| PyTorch + HuggingFace Transformers | ✅ Default | Eager mode, no extra setup |
| vLLM | ✅ Optional | PagedAttention, continuous batching |
| TensorRT-LLM | 🔧 Placeholder | Engine integration ready, results pending |

---

## Benchmark Results

### Tesla T4 (Google Colab, CUDA 12.x)

| Model | Backend | Batch | Prompt Len | Gen Len | Avg Latency (ms) | P95 Latency (ms) | Throughput (tok/s) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|
| distilgpt2 (82M) | PyTorch HF | 2 | 128 | 128 | 666.10 | 684.84 | 384.32 | 0.18 |
| GPT-2 (124M) | PyTorch HF | 2 | 128 | 128 | — | — | — | — |
| GPT-2 Medium (345M) | PyTorch HF | 4 | 128 | 128 | — | — | — | — |

> **Notes:** Decode time dominated by attention + KV-cache growth. Results collected over 3 runs with 1 warmup run. All timings wall-clock end-to-end (prefill + decode).

### A100 (Planned)

| Model | Backend | Batch | Throughput (tok/s) | Peak VRAM (GB) |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | vLLM | 8 | — | — |
| LLaMA-3-8B | vLLM | 8 | — | — |
| LLaMA-3-8B | TensorRT-LLM | 8 | — | — |
| Mistral-7B | TensorRT-LLM | 8 | — | — |

> A100 results in progress. PRs with results on other hardware welcome.

---

## What This Measures

- **Prefill + decode latency** (wall-clock, ms)
- **Tokens/sec throughput** (approx, generation tokens only)
- **Peak GPU memory** (torch.cuda.max_memory_allocated)
- **P95 latency** across multiple runs
- Full config snapshot: batch size, prompt length, generation length, model, backend, GPU

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Run PyTorch baseline (default)

```bash
python benchmarks/run_benchmarks.py \
  --backend pytorch \
  --model distilgpt2 \
  --device cuda
```

Outputs:
- `results/latest.json` — machine-readable report
- `results/latest.md` — human-readable summary

### 3. vLLM backend (optional)

```bash
pip install vllm
python benchmarks/run_benchmarks.py \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda
```

### 4. TensorRT-LLM (optional)

TensorRT-LLM requires a built engine. Once available, plug it into:

```bash
python benchmarks/run_benchmarks.py \
  --backend tensorrt \
  --engine_path /path/to/engine \
  --device cuda
```

---

## Repo Structure

```
.
├── benchmarks/
│   ├── run_benchmarks.py       # Main CLI entrypoint
│   └── configs/                # Benchmark presets (batch size, seq len, etc.)
├── backends/
│   ├── pytorch_runner.py       # HuggingFace Transformers backend
│   ├── vllm_runner.py          # vLLM PagedAttention backend
│   └── tensorrt_llm_runner.py  # TensorRT-LLM placeholder
├── utils/
│   └── metrics.py              # Latency, throughput, VRAM helpers
└── results/                    # Output artifacts (JSON + Markdown)
```

---

## Performance Analysis: What the Numbers Tell You

### Batch size vs throughput trade-off
Larger batches amortize prefill cost but increase per-request latency. vLLM's continuous batching partially closes this gap via PagedAttention.

### KV-cache and decode bottlenecks
For autoregressive decode, attention over the growing KV-cache dominates. On a T4, this is visible in the P95 vs avg latency spread as generation length increases.

### Backend comparison intuition
| Scenario | Recommended Backend |
|---|---|
| Prototyping / debugging | PyTorch HF (eager) |
| High-throughput serving | vLLM (PagedAttention) |
| Latency-critical production | TensorRT-LLM (compiled engine) |

### Profiling deeper
For kernel-level analysis beyond wall-clock timings:
```bash
nsys profile python benchmarks/run_benchmarks.py --backend pytorch --model gpt2
ncu --target-processes all python benchmarks/run_benchmarks.py --backend pytorch --model gpt2
```

---

## At Production Scale

At multi-hundred-GPU scale (e.g., serving a 70B+ parameter model), the bottlenecks shift:
- **Network collective ops** (all-reduce across tensor-parallel ranks) dominate over compute
- **KV-cache memory pressure** requires paged or offloaded attention
- **Batching strategy** (static vs continuous vs chunked prefill) becomes the key throughput lever

This harness is designed to make those tradeoffs visible at small scale — the patterns transfer directly to production inference systems.

---

## Contributing

Results on hardware not listed above are welcome. Open a PR with:
- GPU model + CUDA/driver version
- Full config used
- `results/latest.json` from your run

---

## License

MIT
