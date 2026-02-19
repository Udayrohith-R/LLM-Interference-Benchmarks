from __future__ import annotations
from typing import List, Tuple
import numpy as np
from utils.metrics import now_ms, try_cuda_peak_memory, get_cuda_peak_memory_bytes

def run_vllm_generate(
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
    prompt_tokens: int,
    new_tokens: int,
    warmup_runs: int,
    runs: int,
) -> Tuple[List[float], float, int | None, str]:
    """
    vLLM runner (optional). Requires `pip install vllm` and a supported environment.
    """
    notes = []
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError("vLLM is not installed or not usable in this environment. Install with: pip install vllm") from e

    # vLLM typically expects CUDA; keep device for reporting but vLLM will use GPU if available.
    _ = try_cuda_peak_memory()

    # Build synthetic prompt text that yields ~prompt_tokens (approx, tokenizer differs per model).
    base = "Hello NVIDIA performance benchmarking. " * 64
    prompts = [base for _ in range(batch_size)]

    llm = LLM(model=model_name, dtype=dtype, enforce_eager=False)
    params = SamplingParams(temperature=0.0, max_tokens=int(new_tokens))

    # Warmup
    for _ in range(max(0, warmup_runs)):
        _ = llm.generate(prompts, params)

    latencies = []
    for _ in range(runs):
        t0 = now_ms()
        _ = llm.generate(prompts, params)
        t1 = now_ms()
        latencies.append(t1 - t0)

    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    total_tokens = batch_size * new_tokens
    tokens_per_sec = (total_tokens / (avg_ms / 1000.0)) if avg_ms > 0 else 0.0

    peak = get_cuda_peak_memory_bytes()
    return latencies, float(tokens_per_sec), peak, "; ".join(notes)
