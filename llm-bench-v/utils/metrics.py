"""
Metrics primitives and result containers.

Design notes
------------
* BenchStats holds every statistic we care about for a single run series.
* TokenTimingStats holds the decode-level breakdown (TTFT / TPOT / ITL).
* BenchResult is the fully-serialisable top-level object written to disk.
  It includes config, stats, hardware, package versions, and schema version
  so future tooling can parse old results without ambiguity.

Timing taxonomy (decoder-only models)
--------------------------------------
  ┌─── prefill ───┐┌── decode step 1 ──┐┌── decode step 2 ──┐ …
  │   forward(prompt)  │   forward(tok_1)   │   forward(tok_2)   │
  └───────────────┘└───────────────────┘└───────────────────┘
  |<── TTFT ──────>|
                   |<─ ITL[1] ─>|<─ ITL[2] ─>|
  |<────────────────── total latency ────────────────────────>|

  TPOT  = mean(ITL[1..N-1])   — excludes the prefill step
  TPS   = new_tokens / (total_latency_s)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import numpy as np

SCHEMA_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_ms() -> float:
    """Wall-clock milliseconds via perf_counter."""
    return time.perf_counter() * 1000.0


def reset_cuda_peak() -> None:
    """Reset CUDA peak memory stats if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_cuda_peak_memory_bytes() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Token-level timing
# ---------------------------------------------------------------------------

@dataclass
class TokenTimingStats:
    """
    Per-token timing breakdown extracted from CUDA events recorded by
    _CUDATimingLogitsProcessor inside pytorch_runner.

    ttft_ms  — Time-to-First-Token: wall-clock from generate() entry to
               the end of the prefill forward pass (step 0).
    tpot_ms  — Mean Time Per Output Token across decode steps (steps 1+).
    itl_ms   — Raw inter-token latencies for all decode steps.
    itl_p50_ms / itl_p95_ms / itl_p99_ms — percentile ITL.
    """
    ttft_ms: float
    tpot_ms: float
    itl_ms: List[float] = field(default_factory=list)
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0

    @classmethod
    def from_cuda_events(
        cls,
        start_evt: "torch.cuda.Event",  # type: ignore[name-defined]
        token_evts: "list[torch.cuda.Event]",  # type: ignore[name-defined]
    ) -> "TokenTimingStats":
        """
        Build from CUDA events recorded during model.generate().

        token_evts[0]  → end of prefill (step 0)
        token_evts[1:] → end of each decode step
        """
        import torch
        torch.cuda.synchronize()

        ttft = start_evt.elapsed_time(token_evts[0]) if token_evts else 0.0

        itl: List[float] = []
        for i in range(1, len(token_evts)):
            itl.append(token_evts[i - 1].elapsed_time(token_evts[i]))

        tpot = float(np.mean(itl)) if itl else 0.0
        p50 = float(np.percentile(itl, 50)) if itl else 0.0
        p95 = float(np.percentile(itl, 95)) if itl else 0.0
        p99 = float(np.percentile(itl, 99)) if itl else 0.0

        return cls(
            ttft_ms=ttft,
            tpot_ms=tpot,
            itl_ms=itl,
            itl_p50_ms=p50,
            itl_p95_ms=p95,
            itl_p99_ms=p99,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Don't bloat the JSON with potentially hundreds of raw ITL values
        # unless the caller wants them.  Keep summary stats only by default.
        d.pop("itl_ms", None)
        return d


# ---------------------------------------------------------------------------
# Aggregate stats for a run series
# ---------------------------------------------------------------------------

@dataclass
class BenchStats:
    """
    Aggregated statistics over `runs` timed iterations.

    Latencies are in milliseconds.  Throughput is tokens/second where
    "tokens" counts only newly generated (output) tokens.
    """
    # ----- raw latencies (wall-clock unless CUDA events were available) -----
    latencies_ms: List[float]

    # ----- distribution -----
    mean_ms: float
    std_ms: float
    cv: float            # coefficient of variation = std / mean
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    # ----- throughput -----
    total_output_tokens: int          # batch_size × new_tokens
    tokens_per_sec: float             # total_output_tokens / mean_latency_s
    prefill_tokens_per_sec: float     # prompt_tokens × batch_size / ttft_s  (0 if unknown)

    # ----- decode breakdown (optional; populated when CUDA events available) -----
    token_timing: Optional[TokenTimingStats] = None

    # ----- memory -----
    peak_gpu_mem_bytes: Optional[int] = None
    peak_gpu_mem_gib: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("latencies_ms", None)   # stored at the top level of BenchResult
        if self.token_timing is not None:
            d["token_timing"] = self.token_timing.to_dict()
        return d


def compute_bench_stats(
    latencies_ms: List[float],
    batch_size: int,
    new_tokens: int,
    prompt_tokens: int,
    token_timing: Optional[TokenTimingStats] = None,
    peak_gpu_mem_bytes: Optional[int] = None,
) -> BenchStats:
    """Derive BenchStats from a list of raw latency measurements."""
    arr = np.array(latencies_ms, dtype=float)
    mn = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    cv = sd / mn if mn > 0 else 0.0
    total_out = batch_size * new_tokens
    tps = total_out / (mn / 1000.0) if mn > 0 else 0.0

    prefill_tps = 0.0
    if token_timing is not None and token_timing.ttft_ms > 0:
        prefill_tps = (batch_size * prompt_tokens) / (token_timing.ttft_ms / 1000.0)

    mem_gib: Optional[float] = None
    if peak_gpu_mem_bytes is not None:
        mem_gib = round(peak_gpu_mem_bytes / (1024 ** 3), 3)

    return BenchStats(
        latencies_ms=latencies_ms,
        mean_ms=mn,
        std_ms=sd,
        cv=round(cv, 4),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        total_output_tokens=total_out,
        tokens_per_sec=round(tps, 2),
        prefill_tokens_per_sec=round(prefill_tps, 2),
        token_timing=token_timing,
        peak_gpu_mem_bytes=peak_gpu_mem_bytes,
        peak_gpu_mem_gib=mem_gib,
    )


# ---------------------------------------------------------------------------
# Top-level result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """
    Fully self-describing benchmark result.

    Written as a versioned JSON artifact.  The `schema_version` field allows
    future tooling to handle old results gracefully.
    """
    schema_version: str
    timestamp: str
    config: Dict[str, Any]
    stats: BenchStats
    hardware: Dict[str, Any]
    latencies_ms: List[float]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "config": self.config,
            "stats": self.stats.to_dict(),
            "hardware": self.hardware,
            "latencies_ms": self.latencies_ms,
            "notes": self.notes,
        }
