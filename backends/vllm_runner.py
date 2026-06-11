"""
vLLM backend.

vLLM's LLM.generate() batches requests internally and returns a
list of RequestOutput objects.  We extract per-request metrics from
request_output.metrics (RequestMetrics), which vLLM populates when
available:

  metrics.first_token_time  — absolute timestamp of first token output
  metrics.finished_time     — absolute timestamp when generation finished
  metrics.first_scheduled_time — when the request was first scheduled

TTFT = first_token_time - first_scheduled_time
Total latency = finished_time - first_scheduled_time

Notes
-----
* vLLM manages its own CUDA graph / PagedAttention internals; we do not
  inject CUDA events.  Latencies are therefore wall-clock, but vLLM's
  scheduler is tight enough that these are accurate to within ~1 ms.
* The `enforce_eager=False` default lets vLLM use CUDA graphs for decode,
  which is the production configuration.  Set enforce_eager=True to
  profile the un-graphed path.
* dtype is passed directly to vLLM's engine so it must match vLLM's
  accepted strings ("float16", "bfloat16", "auto").
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from backends.base import InferenceBackend
from utils.config import SingleRunConfig
from utils.metrics import (
    BenchStats,
    TokenTimingStats,
    compute_bench_stats,
    get_cuda_peak_memory_bytes,
    reset_cuda_peak,
)

logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """vLLM offline-inference backend."""

    def __init__(self) -> None:
        self._llm = None
        self._sampling_params = None
        self._prompts: List[str] = []

    # ------------------------------------------------------------------
    # InferenceBackend.load
    # ------------------------------------------------------------------

    def load(self, config: SingleRunConfig) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed.  Run: pip install vllm"
            ) from exc

        logger.info("Initialising vLLM engine for model '%s'", config.model)
        self._llm = LLM(
            model=config.model,
            dtype=config.dtype,
            enforce_eager=False,        # CUDA graphs on by default
            max_model_len=config.prompt_tokens + config.new_tokens + 64,
        )
        self._sampling_params = SamplingParams(
            temperature=0.0,            # greedy
            max_tokens=config.new_tokens,
        )
        # Synthetic prompt — same tiling strategy as PyTorchBackend.
        base = "Benchmark prompt for LLM inference latency measurement. "
        # Repeat until we have roughly prompt_tokens worth of text.
        target_chars = config.prompt_tokens * 5  # rough chars-per-token estimate
        prompt = (base * (target_chars // len(base) + 1))[: target_chars]
        self._prompts = [prompt] * config.batch_size
        logger.info("vLLM engine ready")

    # ------------------------------------------------------------------
    # InferenceBackend.run_single
    # ------------------------------------------------------------------

    def run_single(
        self, config: SingleRunConfig
    ) -> Tuple[float, Optional[TokenTimingStats]]:
        assert self._llm is not None, "call load() before run_single()"

        t0 = time.perf_counter()
        outputs = self._llm.generate(self._prompts, self._sampling_params)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        # Attempt to extract TTFT from RequestMetrics (vLLM ≥ 0.4).
        token_timing: Optional[TokenTimingStats] = None
        ttft_vals: list[float] = []
        for out in outputs:
            m = getattr(out, "metrics", None)
            if m is not None:
                scheduled = getattr(m, "first_scheduled_time", None)
                first_tok = getattr(m, "first_token_time", None)
                finished = getattr(m, "finished_time", None)
                if scheduled and first_tok and finished:
                    ttft_ms = (first_tok - scheduled) * 1000.0
                    ttft_vals.append(ttft_ms)

        if ttft_vals:
            avg_ttft = float(np.mean(ttft_vals))
            # Rough TPOT: (total - ttft) / (new_tokens - 1)
            decode_ms = max(latency_ms - avg_ttft, 0.0)
            tpot = decode_ms / max(config.new_tokens - 1, 1)
            token_timing = TokenTimingStats(
                ttft_ms=avg_ttft,
                tpot_ms=tpot,
            )

        return latency_ms, token_timing

    # ------------------------------------------------------------------
    # Override benchmark() to keep CUDA memory accounting correct.
    # vLLM pre-allocates its KV-cache pool at engine init time, so we
    # must capture peak after the first generate() call, not after warmup.
    # ------------------------------------------------------------------

    def benchmark(self, config: SingleRunConfig) -> BenchStats:
        logger.info("vLLM warmup (%d run(s))", config.warmup_runs)
        for _ in range(config.warmup_runs):
            self.run_single(config)

        reset_cuda_peak()

        logger.info("vLLM timing %d run(s)", config.runs)
        latencies: list[float] = []
        token_timings: list[TokenTimingStats] = []

        for i in range(config.runs):
            lat, tt = self.run_single(config)
            latencies.append(lat)
            if tt is not None:
                token_timings.append(tt)
            logger.debug("  run %d/%d  %.2f ms", i + 1, config.runs, lat)

        agg_tt: Optional[TokenTimingStats] = None
        if token_timings:
            agg_tt = TokenTimingStats(
                ttft_ms=float(np.mean([t.ttft_ms for t in token_timings])),
                tpot_ms=float(np.mean([t.tpot_ms for t in token_timings])),
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
