from __future__ import annotations
from typing import Dict, Any

def bytes_to_gb(x: int) -> float:
    return x / (1024**3)

def render_summary_md(report: Dict[str, Any]) -> str:
    r = report["result"]
    peak = r.get("peak_gpu_mem_bytes")
    peak_gb = f"{bytes_to_gb(peak):.2f} GB" if isinstance(peak, int) else "N/A"

    md = f"""# LLM Inference Benchmark Summary

**Backend:** {r['backend']}
**Model:** {r['model']}
**Device:** {r['device']}
**DType:** {r['dtype']}

## Workload
- Batch size: **{r['batch_size']}**
- Prompt tokens: **{r['prompt_tokens']}**
- New tokens: **{r['new_tokens']}**
- Runs: **{r['runs']}** (+ {r['warmup_runs']} warmup)

## Results
- Avg latency: **{r['avg_latency_ms']:.2f} ms**
- P95 latency: **{r['p95_latency_ms']:.2f} ms**
- Throughput: **{r['tokens_per_sec']:.2f} tokens/sec**
- Peak GPU mem: **{peak_gb}**

## Notes
{r.get('notes','') or '—'}
"""
    return md
