"""
Rendering helpers.

render_rich_table()   — prints a Rich table to the terminal for a single run.
render_sweep_table()  — prints a comparison table across a sweep.
render_summary_md()   — returns a Markdown string for a single BenchResult.
render_sweep_md()     — returns a Markdown comparison table for a sweep.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich import box

from utils.metrics import BenchResult, BenchStats

console = Console()


# ---------------------------------------------------------------------------
# Single-run Rich table
# ---------------------------------------------------------------------------

def render_rich_table(result: BenchResult) -> None:
    cfg = result.config
    stats = result.stats

    t = Table(
        title=f"[bold cyan]LLM Inference Benchmark[/bold cyan]  "
              f"[dim]{cfg['backend']} / {cfg['model']}[/dim]",
        box=box.ROUNDED,
        highlight=True,
    )
    t.add_column("Metric", style="bold", min_width=28)
    t.add_column("Value", justify="right", min_width=18)

    def row(label: str, val: Any, unit: str = "") -> None:
        t.add_row(label, f"{val}{unit}")

    # Config
    row("Backend",              cfg["backend"])
    row("Model",                cfg["model"])
    row("Device",               cfg["device"])
    row("DType",                cfg["dtype"])
    row("Batch size",           cfg["batch_size"])
    row("Prompt tokens",        cfg["prompt_tokens"])
    row("New tokens",           cfg["new_tokens"])
    row("Warmup / timed runs",  f"{cfg['warmup_runs']} / {cfg['runs']}")

    t.add_section()

    # Latency
    row("Mean latency",         f"{stats['mean_ms']:.2f}", " ms")
    row("Std latency",          f"{stats['std_ms']:.2f}", " ms")
    row("CV (σ/μ)",             f"{stats['cv']:.3f}")
    row("P50 latency",          f"{stats['p50_ms']:.2f}", " ms")
    row("P90 latency",          f"{stats['p90_ms']:.2f}", " ms")
    row("P95 latency",          f"{stats['p95_ms']:.2f}", " ms")
    row("P99 latency",          f"{stats['p99_ms']:.2f}", " ms")
    row("Min / Max",            f"{stats['min_ms']:.2f} / {stats['max_ms']:.2f}", " ms")

    t.add_section()

    # Throughput
    row("Throughput",           f"{stats['tokens_per_sec']:.1f}", " tok/s")
    if stats.get("prefill_tokens_per_sec", 0):
        row("Prefill throughput", f"{stats['prefill_tokens_per_sec']:.1f}", " tok/s")

    # Token-level timing
    tt = stats.get("token_timing")
    if tt:
        t.add_section()
        row("TTFT (mean)",       f"{tt['ttft_ms']:.2f}", " ms")
        row("TPOT (mean)",       f"{tt['tpot_ms']:.2f}", " ms")
        row("ITL P50",           f"{tt['itl_p50_ms']:.2f}", " ms")
        row("ITL P95",           f"{tt['itl_p95_ms']:.2f}", " ms")
        row("ITL P99",           f"{tt['itl_p99_ms']:.2f}", " ms")

    # Memory
    t.add_section()
    mem_gib = stats.get("peak_gpu_mem_gib")
    row("Peak GPU mem", f"{mem_gib:.3f} GiB" if mem_gib is not None else "N/A")

    console.print(t)


# ---------------------------------------------------------------------------
# Sweep comparison table (multiple runs)
# ---------------------------------------------------------------------------

def render_sweep_table(results: List[BenchResult]) -> None:
    if not results:
        return

    t = Table(
        title="[bold cyan]Benchmark Sweep Results[/bold cyan]",
        box=box.SIMPLE_HEAD,
        highlight=True,
        show_lines=False,
    )
    t.add_column("Backend",   style="cyan",  no_wrap=True)
    t.add_column("BS",        justify="right")
    t.add_column("Prompt",    justify="right")
    t.add_column("New",       justify="right")
    t.add_column("Mean (ms)", justify="right")
    t.add_column("P95 (ms)",  justify="right")
    t.add_column("Tok/s",     justify="right")
    t.add_column("TTFT (ms)", justify="right")
    t.add_column("TPOT (ms)", justify="right")
    t.add_column("Mem (GiB)", justify="right")

    for r in results:
        cfg = r.config
        s = r.stats
        tt = s.get("token_timing") or {}
        t.add_row(
            cfg["backend"],
            str(cfg["batch_size"]),
            str(cfg["prompt_tokens"]),
            str(cfg["new_tokens"]),
            f"{s['mean_ms']:.1f}",
            f"{s['p95_ms']:.1f}",
            f"{s['tokens_per_sec']:.0f}",
            f"{tt['ttft_ms']:.1f}" if tt.get("ttft_ms") else "—",
            f"{tt['tpot_ms']:.1f}" if tt.get("tpot_ms") else "—",
            f"{s['peak_gpu_mem_gib']:.2f}" if s.get("peak_gpu_mem_gib") else "—",
        )

    console.print(t)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def bytes_to_gib(x: int) -> float:
    return x / (1024 ** 3)


def render_summary_md(result: BenchResult) -> str:
    r = result if isinstance(result, dict) else result.to_dict()
    cfg = r["config"]
    s = r["stats"]
    hw = r["hardware"]
    tt = s.get("token_timing") or {}

    mem = (
        f"{s['peak_gpu_mem_gib']:.3f} GiB"
        if s.get("peak_gpu_mem_gib") is not None
        else "N/A"
    )
    gpu_name = hw.get("device_name", hw.get("cpu_model", "unknown"))

    lines = [
        f"# LLM Inference Benchmark — {cfg['backend']} / {cfg['model']}",
        "",
        f"**Timestamp:** {r['timestamp']}  ",
        f"**Schema version:** {r['schema_version']}",
        "",
        "## Hardware",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| GPU   | {gpu_name} |",
        f"| CUDA  | {hw.get('cuda_version','N/A')} |",
        f"| PyTorch | {hw.get('torch_version','N/A')} |",
        "",
        "## Configuration",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Backend | {cfg['backend']} |",
        f"| Model | `{cfg['model']}` |",
        f"| Device | {cfg['device']} |",
        f"| DType | {cfg['dtype']} |",
        f"| Batch size | {cfg['batch_size']} |",
        f"| Prompt tokens | {cfg['prompt_tokens']} |",
        f"| New tokens | {cfg['new_tokens']} |",
        f"| Warmup / runs | {cfg['warmup_runs']} / {cfg['runs']} |",
        "",
        "## Latency",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean | **{s['mean_ms']:.2f} ms** |",
        f"| Std  | {s['std_ms']:.2f} ms |",
        f"| CV   | {s['cv']:.3f} |",
        f"| P50  | {s['p50_ms']:.2f} ms |",
        f"| P90  | {s['p90_ms']:.2f} ms |",
        f"| P95  | {s['p95_ms']:.2f} ms |",
        f"| P99  | {s['p99_ms']:.2f} ms |",
        f"| Min / Max | {s['min_ms']:.2f} / {s['max_ms']:.2f} ms |",
        "",
        "## Throughput",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Decode throughput | **{s['tokens_per_sec']:.1f} tok/s** |",
        f"| Prefill throughput | {s.get('prefill_tokens_per_sec', '—')} tok/s |",
    ]

    if tt.get("ttft_ms") is not None:
        lines += [
            "",
            "## Token-Level Timing",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| TTFT (mean) | {tt['ttft_ms']:.2f} ms |",
            f"| TPOT (mean) | {tt['tpot_ms']:.2f} ms |",
            f"| ITL P50 | {tt.get('itl_p50_ms', '—')} ms |",
            f"| ITL P95 | {tt.get('itl_p95_ms', '—')} ms |",
            f"| ITL P99 | {tt.get('itl_p99_ms', '—')} ms |",
        ]

    lines += [
        "",
        "## Memory",
        f"Peak GPU memory: **{mem}**",
        "",
        f"## Notes",
        r.get("notes") or "—",
    ]

    return "\n".join(lines) + "\n"


def render_sweep_md(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Benchmark Sweep Results",
        "",
        "| Backend | BS | Prompt | New | Mean (ms) | P95 (ms) | Tok/s | TTFT (ms) | TPOT (ms) | Mem (GiB) |",
        "|---------|-----|--------|-----|-----------|----------|-------|-----------|-----------|-----------|",
    ]
    for r in results:
        cfg = r["config"]
        s = r["stats"]
        tt = s.get("token_timing") or {}
        lines.append(
            f"| {cfg['backend']} | {cfg['batch_size']} | {cfg['prompt_tokens']} | "
            f"{cfg['new_tokens']} | {s['mean_ms']:.1f} | {s['p95_ms']:.1f} | "
            f"{s['tokens_per_sec']:.0f} | "
            f"{tt.get('ttft_ms', '—'):.1f} | "
            f"{tt.get('tpot_ms', '—'):.1f} | "
            f"{s['peak_gpu_mem_gib'] or '—'} |"
        )
    return "\n".join(lines) + "\n"
