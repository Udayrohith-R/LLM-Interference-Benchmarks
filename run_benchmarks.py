from __future__ import annotations
import argparse
import json
import os
from statistics import mean
from typing import Any, Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table

from metrics import BenchResult
from render import render_summary_md

console = Console()

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), p))

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_results_dir() -> str:
    outdir = os.path.join(os.path.dirname(__file__), "..", "results")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="Path to a JSON config preset.")
    ap.add_argument("--backend", type=str, default="pytorch", choices=["pytorch","vllm","tensorrt-llm"])
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    backend = cfg.get("backend", args.backend)
    model = cfg.get("model", args.model)
    device = cfg.get("device", args.device)
    dtype = cfg.get("dtype", args.dtype)
    batch_size = int(cfg.get("batch_size", args.batch_size))
    prompt_tokens = int(cfg.get("prompt_tokens", args.prompt_tokens))
    new_tokens = int(cfg.get("new_tokens", args.new_tokens))
    warmup_runs = int(cfg.get("warmup_runs", args.warmup_runs))
    runs = int(cfg.get("runs", args.runs))

    latencies: List[float] = []
    tokens_per_sec = 0.0
    peak_mem = None
    notes = ""

    if backend == "pytorch":
        from backends.pytorch_runner import run_pytorch_generate
        latencies, tokens_per_sec, peak_mem, notes = run_pytorch_generate(
            model, device, dtype, batch_size, prompt_tokens, new_tokens, warmup_runs, runs
        )
    elif backend == "vllm":
        from backends.vllm_runner import run_vllm_generate
        latencies, tokens_per_sec, peak_mem, notes = run_vllm_generate(
            model, device, dtype, batch_size, prompt_tokens, new_tokens, warmup_runs, runs
        )
    else:
        from backends.tensorrt_llm_runner import run_tensorrt_llm_generate
        latencies, tokens_per_sec, peak_mem, notes = run_tensorrt_llm_generate(
            model, device, dtype, batch_size, prompt_tokens, new_tokens, warmup_runs, runs
        )

    avg_ms = float(mean(latencies)) if latencies else 0.0
    p95_ms = percentile(latencies, 95)

    result = BenchResult(
        backend=backend,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        warmup_runs=warmup_runs,
        runs=runs,
        avg_latency_ms=avg_ms,
        p95_latency_ms=p95_ms,
        tokens_per_sec=float(tokens_per_sec),
        peak_gpu_mem_bytes=peak_mem,
        notes=notes,
    )

    report = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "result": result.to_dict(),
        "latencies_ms": latencies,
    }

    outdir = ensure_results_dir()
    json_path = os.path.join(outdir, "latest.json")
    md_path = os.path.join(outdir, "latest.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_summary_md(report))

    table = Table(title="LLM Inference Benchmark")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Backend", backend)
    table.add_row("Model", model)
    table.add_row("Avg latency (ms)", f"{avg_ms:.2f}")
    table.add_row("P95 latency (ms)", f"{p95_ms:.2f}")
    table.add_row("Tokens/sec", f"{tokens_per_sec:.2f}")
    table.add_row("Peak GPU mem", str(peak_mem) if peak_mem is not None else "N/A")
    console.print(table)
    console.print(f"[green]Wrote[/green] {json_path}")
    console.print(f"[green]Wrote[/green] {md_path}")

if __name__ == "__main__":
    main()
