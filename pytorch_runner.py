from __future__ import annotations
from typing import List, Tuple
import numpy as np

from utils.metrics import now_ms, try_cuda_peak_memory, get_cuda_peak_memory_bytes

def run_pytorch_generate(
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
    Returns: (latencies_ms, tokens_per_sec, peak_gpu_mem_bytes, notes)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    notes = []
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    # Build synthetic prompt of approx prompt_tokens using repeated tokens.
    # We do this to keep prompts reproducible without external datasets.
    base_text = "Hello NVIDIA performance benchmarking. "
    enc = tokenizer(base_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    # Repeat to reach target tokens
    reps = max(1, int(np.ceil(prompt_tokens / input_ids.shape[1])))
    input_ids = input_ids.repeat(batch_size, reps).to(device)
    input_ids = input_ids[:, :prompt_tokens]

    attn = torch.ones_like(input_ids, device=device)

    # Reset peak mem stats if available
    _ = try_cuda_peak_memory()

    gen_kwargs = dict(
        max_new_tokens=int(new_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Warmup
    with torch.inference_mode():
        for _ in range(max(0, warmup_runs)):
            _ = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    latencies = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = now_ms()
            _ = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = now_ms()
            latencies.append(t1 - t0)

    # tokens/sec approx: total generated tokens / avg latency seconds
    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    total_tokens = batch_size * new_tokens
    tokens_per_sec = (total_tokens / (avg_ms / 1000.0)) if avg_ms > 0 else 0.0

    peak = get_cuda_peak_memory_bytes()
    if device.startswith("cuda") and peak is None:
        notes.append("CUDA peak memory not available; check torch.cuda availability.")
    return latencies, float(tokens_per_sec), peak, "; ".join(notes)
