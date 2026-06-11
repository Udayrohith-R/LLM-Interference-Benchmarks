"""
Abstract base for inference backends.

Each backend must implement `load()` and `run_single()`.  The base class
provides the full benchmark loop (warmup + timed runs + stat collection) so
there is no logic duplication across backends.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from utils.config import SingleRunConfig
from utils.metrics import (
    BenchStats,
    TokenTimingStats,
    compute_bench_stats,
    get_cuda_peak_memory_bytes,
    reset_cuda_peak,
)

logger = logging.getLogger(__name__)


class InferenceBackend(ABC):
    """
    Contract for all inference backends.

    Subclasses implement:
      load()       — one-time model/engine initialisation.
      run_single() — one timed forward+generate pass.
                     Returns (latency_ms, token_timing | None).

    The base class owns the benchmark loop so all backends share identical
    warmup / timing / stat-collection semantics.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, config: SingleRunConfig) -> None:
        """Load model / engine.  Called once before any runs."""

    @abstractmethod
    def run_single(
        self, config: SingleRunConfig
    ) -> Tuple[float, Optional[TokenTimingStats]]:
        """
        Execute one generate() call.

        Returns
        -------
        latency_ms : float
            End-to-end generation latency for this call.
        token_timing : TokenTimingStats | None
            Per-token breakdown if available (CUDA events), else None.
        """

    # ------------------------------------------------------------------
    # Benchmark loop (shared by all backends)
    # ------------------------------------------------------------------

    def benchmark(self, config: SingleRunConfig) -> BenchStats:
        """
        Run warmup + timed iterations and return aggregated BenchStats.

        Warmup runs are not timed.  CUDA peak memory is reset once after
        all warmup iterations so that model weights pre-loaded into VRAM
        during warmup are counted in peak_gpu_mem_bytes (that is the true
        working memory footprint, not just the activation overhead).
        """
        logger.info(
            "Warming up for %d run(s) — backend=%s model=%s bs=%d",
            config.warmup_runs,
            config.backend,
            config.model,
            config.batch_size,
        )
        for _ in range(config.warmup_runs):
            self.run_single(config)

        # Reset peak memory AFTER warmup so the weights are already resident.
        reset_cuda_peak()

        logger.info("Timing %d run(s)…", config.runs)
        latencies: list[float] = []
        token_timings: list[TokenTimingStats] = []

        for i in range(config.runs):
            lat, tt = self.run_single(config)
            latencies.append(lat)
            if tt is not None:
                token_timings.append(tt)
            logger.debug("  run %d/%d  %.2f ms", i + 1, config.runs, lat)

        # Aggregate token timing across runs: average TTFT and TPOT.
        agg_tt: Optional[TokenTimingStats] = None
        if token_timings:
            import numpy as np
            agg_tt = TokenTimingStats(
                ttft_ms=float(np.mean([t.ttft_ms for t in token_timings])),
                tpot_ms=float(np.mean([t.tpot_ms for t in token_timings])),
                itl_p50_ms=float(np.mean([t.itl_p50_ms for t in token_timings])),
                itl_p95_ms=float(np.mean([t.itl_p95_ms for t in token_timings])),
                itl_p99_ms=float(np.mean([t.itl_p99_ms for t in token_timings])),
            )

        peak = get_cuda_peak_memory_bytes()
        return compute_bench_stats(
            latencies_ms=latencies,
            batch_size=config.batch_size,
            new_tokens=config.new_tokens,
            prompt_tokens=config.prompt_tokens,
            token_timing=agg_tt,
            peak_gpu_mem_bytes=peak,
        )
