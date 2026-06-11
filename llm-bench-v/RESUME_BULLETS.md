# Resume bullets — LLM Inference Benchmarks

Fill in the bracketed numbers with your actual measured results.

---

**LLM Inference Acceleration — PyTorch vs vLLM vs TensorRT-LLM**

- Built a production-grade inference benchmark harness measuring TTFT, TPOT,
  inter-token latency (ITL P50/P95/P99), and peak GPU memory; replaced
  wall-clock timing with CUDA Events injected via a custom `LogitsProcessor`
  to eliminate Python-scheduling jitter and achieve sub-millisecond timing
  accuracy on NVIDIA [A100 / H100].

- Characterised PyTorch greedy-decode prefill as compute-bound (O(seq²)
  attention) and decode as memory-bandwidth-bound (KV-cache streaming);
  measured prefill throughput of [X] tok/s vs decode TPOT of [Y] ms/tok
  across batch sizes 1–8 at fp16.

- Demonstrated vLLM's PagedAttention reducing peak GPU memory by [Z]% vs
  HF Transformers at batch size 8 (prompt=256, decode=128), with
  [W]% throughput improvement at P95 latency budget of [V] ms.

- Implemented a Pydantic-validated sweep runner enumerating batch × seq-len
  combinations; output versioned JSON artifacts (schema v2.0) with hardware
  fingerprinting (GPU model, CUDA version, SM count, memory bandwidth) for
  cross-machine reproducibility.

- Added CI regression check (GitHub Actions) that fails PRs when mean latency
  increases > 15% or throughput drops > 10% against a stored baseline, blocking
  regressions before merge.
