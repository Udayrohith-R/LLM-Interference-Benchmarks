# LLM Inference Benchmarks

**PyTorch (HF Transformers) · vLLM · TensorRT-LLM**

A production-grade, reproducible benchmark harness for measuring LLM inference
latency, throughput, and GPU memory across multiple runtimes.

---

## What this measures — and how

### Timing taxonomy

```
┌─── prefill ──────┐┌── decode 1 ──┐┌── decode 2 ──┐ … ┌── decode N ──┐
│  forward(prompt) ││ forward(t₁)  ││ forward(t₂)  │   │ forward(tN)  │
└──────────────────┘└──────────────┘└──────────────┘   └──────────────┘
│<─── TTFT ────────>│
                    │<─ ITL[1] ──>│<─ ITL[2] ──>│   │<─ ITL[N] ──>│
│<───────────────────────── total latency ──────────────────────────>│

TPOT = mean(ITL[1..N])   — decode throughput per token (excludes prefill)
TPS  = new_tokens / total_latency_s
```

### Why CUDA events, not wall-clock

`time.perf_counter()` around a CUDA kernel call returns the *enqueue* time —
the actual GPU work may not have started yet.  This harness uses
`torch.cuda.Event(enable_timing=True)` so that latency is measured inside the
CUDA stream, giving sub-millisecond accuracy.  Wall-clock is kept only as a
fallback on CPU.

### Per-token timing via LogitsProcessor

We inject a `_CUDATimingLogitsProcessor` into HF's `model.generate()`.  It
records one CUDA Event at the end of every forward pass — step 0 is the
prefill, steps 1..N are decode steps.  This gives us TTFT and per-token ITL
without modifying model internals or forking Transformers.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single run (GPU):
python benchmarks/run_benchmarks.py \
    --backend pytorch \
    --model distilgpt2 \
    --device cuda \
    --dtype float16

# Single run (CPU, no CUDA required):
python benchmarks/run_benchmarks.py \
    --backend pytorch \
    --model distilgpt2 \
    --device cpu \
    --dtype float32

# From a YAML config:
python benchmarks/run_benchmarks.py --config benchmarks/configs/small.yaml

# Batch-size sweep (8 combinations):
python benchmarks/run_benchmarks.py --config benchmarks/configs/sweep_batch_size.yaml
```

Outputs written to `results/`:
- `latest.json` — full result with schema version, hardware info, all percentiles
- `latest.md` — human-readable Markdown summary
- `sweep_<name>.json` / `sweep_<name>.md` — comparison matrix (sweep mode only)

---

## Backends

| Backend | Install | Notes |
|---------|---------|-------|
| `pytorch` | default | HF Transformers greedy decode; CUDA events for timing |
| `vllm` | `pip install vllm` | PagedAttention; per-request TTFT from `RequestMetrics` |
| `tensorrt-llm` | [TRT-LLM docs](https://github.com/NVIDIA/TensorRT-LLM) | Requires pre-built engine; see `backends/tensorrt_llm_runner.py` |

### TensorRT-LLM

TRT-LLM needs an engine directory built with `trtllm-build`:

```bash
trtllm-build \
    --checkpoint_dir ./llama3-8b-hf \
    --output_dir ./engines/llama3-8b-fp16 \
    --dtype float16 \
    --max_batch_size 8 \
    --max_input_len 512 \
    --max_output_len 256

python benchmarks/run_benchmarks.py \
    --backend tensorrt-llm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --trt-engine-dir ./engines/llama3-8b-fp16 \
    --batch-size 4 \
    --prompt-tokens 256 \
    --new-tokens 128
```

---

## Result schema (v2.0)

```json
{
  "schema_version": "2.0",
  "timestamp": "2025-03-01T12:00:00Z",
  "config": { "backend": "pytorch", "model": "gpt2", "batch_size": 4, ... },
  "stats": {
    "mean_ms": 142.3,  "std_ms": 2.1,  "cv": 0.015,
    "p50_ms": 141.8,   "p90_ms": 144.9, "p95_ms": 146.1, "p99_ms": 149.2,
    "tokens_per_sec": 361.2,
    "prefill_tokens_per_sec": 18400.0,
    "token_timing": {
      "ttft_ms": 55.4,
      "tpot_ms": 3.8,
      "itl_p50_ms": 3.7,  "itl_p95_ms": 4.1,  "itl_p99_ms": 4.9
    },
    "peak_gpu_mem_bytes": 1073741824,
    "peak_gpu_mem_gib": 1.000
  },
  "hardware": {
    "device_name": "NVIDIA A100-SXM4-40GB",
    "cuda_version": "12.1",
    "compute_capability": "8.0",
    "total_memory_gib": 40.0,
    "sm_count": 108
  },
  "latencies_ms": [141.2, 142.8, 143.1, ...]
}
```

---

## Regression CI

Two GitHub Actions workflows are included:

- **`bench_cpu.yml`** — runs on every push/PR; CPU sanity check with `distilgpt2`.
  Validates that outputs are well-formed and non-zero.

- **`regression.yml`** — runs on PRs; compares the PR result against the baseline
  stored on `main`.  Fails if latency increases > 15% or throughput drops > 10%.
  Configure thresholds via `--latency-threshold` / `--throughput-threshold`.

To update the baseline after a legitimate performance change:

```bash
python -m utils.regression save results/latest.json results/baseline.json
git add results/baseline.json && git commit -m "chore: update latency baseline"
```


## Repo structure

```
benchmarks/
  run_benchmarks.py       Main CLI — single run and sweep mode
  configs/                YAML benchmark presets
backends/
  base.py                 Abstract InferenceBackend
  pytorch_runner.py       HF Transformers + CUDA event timing + TTFT/TPOT
  vllm_runner.py          vLLM + RequestMetrics TTFT
  tensorrt_llm_runner.py  TRT-LLM ModelRunner (requires pre-built engine)
utils/
  config.py               Pydantic v2 BenchConfig + SweepDimension
  hardware.py             GPU/CPU fingerprinting
  metrics.py              BenchStats, TokenTimingStats, BenchResult
  render.py               Rich tables + Markdown reports
  regression.py           CI regression checker
results/                  Output artifacts (gitignored except baseline.json)
.github/workflows/        bench_cpu.yml + regression.yml
```

---

## License

MIT
