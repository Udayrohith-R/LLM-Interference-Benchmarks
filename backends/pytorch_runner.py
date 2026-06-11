"""
PyTorch + Hugging Face Transformers backend.

Timing methodology
------------------
Wall-clock via time.perf_counter() is unreliable for GPU work because it
includes Python scheduling jitter.  This backend uses CUDA Events instead:

  start_evt.record()          ← before model.generate()
  _CUDATimingLogitsProcessor  ← records one CUDA event per decode step
  torch.cuda.synchronize()    ← flush the CUDA queue before reading times

Token-level timing via LogitsProcessor
---------------------------------------
HF's greedy/sampling loop calls LogitsProcessor once per forward pass:
  - Step 0  : full prompt prefill  → TTFT = start_evt → token_evts[0]
  - Steps 1+: single-token decode  → ITL[i] = token_evts[i] elapsed from
                                              token_evts[i-1]
  TPOT = mean(ITL[1:])  (decode only, excluding prefill)

CPU fallback
-------------
On CPU, CUDA events are not available.  We fall back to wall-clock
time.perf_counter() and skip token-level timing.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from backends.base import InferenceBackend
from utils.config import SingleRunConfig
from utils.metrics import TokenTimingStats, now_ms

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# CUDA-event LogitsProcessor
# ---------------------------------------------------------------------------

class _CUDATimingLogitsProcessor(LogitsProcessor):
    """
    Records one CUDA Event at the end of each generate() forward pass.

    token_evts[0] → prefill complete (≈ TTFT marker)
    token_evts[k] → k-th decode token generated  (k ≥ 1)
    """

    def __init__(self, use_cuda: bool) -> None:
        self._use_cuda = use_cuda
        self.token_evts: List[torch.cuda.Event] = []

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self._use_cuda:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self.token_evts.append(evt)
        return scores  # pass-through; we only hook for timing

    def extract_token_timing(
        self, start_evt: torch.cuda.Event
    ) -> Optional[TokenTimingStats]:
        if not self._use_cuda or not self.token_evts:
            return None
        return TokenTimingStats.from_cuda_events(start_evt, self.token_evts)


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class PyTorchBackend(InferenceBackend):
    """HF Transformers greedy-decode backend with CUDA-event timing."""

    def __init__(self) -> None:
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._input_ids: Optional[torch.Tensor] = None
        self._attn_mask: Optional[torch.Tensor] = None
        self._use_cuda: bool = False
        self._gen_kwargs: dict = {}

    # ------------------------------------------------------------------
    # InferenceBackend.load
    # ------------------------------------------------------------------

    def load(self, config: SingleRunConfig) -> None:
        logger.info("Loading model '%s' (dtype=%s, device=%s)", config.model, config.dtype, config.device)
        torch_dtype = _DTYPE_MAP.get(config.dtype, torch.float16)
        self._use_cuda = config.device.startswith("cuda")

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,  # stream from disk; avoids double-peak
        )
        self._model.to(config.device)
        self._model.eval()

        # Pre-build the synthetic prompt once — avoids rebuilding every run.
        self._input_ids, self._attn_mask = self._build_input(config)

        self._gen_kwargs = dict(
            max_new_tokens=config.new_tokens,
            do_sample=False,          # greedy: deterministic & reproducible
            use_cache=True,           # KV cache must be on for decode timing
            pad_token_id=self._tokenizer.eos_token_id,
        )
        logger.info(
            "Model loaded — params: %.2fB",
            sum(p.numel() for p in self._model.parameters()) / 1e9,
        )

    # ------------------------------------------------------------------
    # InferenceBackend.run_single
    # ------------------------------------------------------------------

    def run_single(
        self, config: SingleRunConfig
    ) -> Tuple[float, Optional[TokenTimingStats]]:
        assert self._model is not None, "call load() before run_single()"

        timing_proc = _CUDATimingLogitsProcessor(use_cuda=self._use_cuda)
        logits_processors = LogitsProcessorList([timing_proc])

        if self._use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        t0 = now_ms()  # wall-clock backup

        with torch.inference_mode():
            self._model.generate(
                input_ids=self._input_ids,
                attention_mask=self._attn_mask,
                logits_processor=logits_processors,
                **self._gen_kwargs,
            )

        if self._use_cuda:
            end_evt.record()
            torch.cuda.synchronize()
            latency_ms = start_evt.elapsed_time(end_evt)
        else:
            latency_ms = now_ms() - t0

        token_timing: Optional[TokenTimingStats] = None
        if self._use_cuda:
            token_timing = timing_proc.extract_token_timing(start_evt)

        return latency_ms, token_timing

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_input(
        self, config: SingleRunConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a reproducible synthetic prompt of ~prompt_tokens length.

        We tile a short base sentence rather than using random token IDs
        so that attention patterns are realistic (coherent text, not noise).
        """
        assert self._tokenizer is not None
        base = "Benchmark prompt for LLM inference latency measurement. "
        enc = self._tokenizer(base, return_tensors="pt")
        input_ids: torch.Tensor = enc["input_ids"]  # (1, seq)
        reps = max(1, int(np.ceil(config.prompt_tokens / input_ids.shape[1])))
        # Tile batch dim AND sequence dim separately to get correct shape.
        input_ids = input_ids.repeat(1, reps)[:, : config.prompt_tokens]
        input_ids = input_ids.repeat(config.batch_size, 1)
        attn_mask = torch.ones_like(input_ids)
        return input_ids.to(config.device), attn_mask.to(config.device)


# ---------------------------------------------------------------------------
# Module-level entrypoint (keeps run_benchmarks.py tidy)
# ---------------------------------------------------------------------------

def run_pytorch_generate(
    config: SingleRunConfig,
) -> Tuple[list, float, Optional[int], str]:
    """
    Legacy shim kept for backward-compatibility with direct callers.
    Prefer instantiating PyTorchBackend directly.
    """
    from utils.metrics import get_cuda_peak_memory_bytes, reset_cuda_peak

    backend = PyTorchBackend()
    backend.load(config)
    stats = backend.benchmark(config)
    notes = ""
    if config.device.startswith("cuda") and stats.peak_gpu_mem_bytes is None:
        notes = "CUDA peak memory unavailable"
    return (
        stats.latencies_ms,
        stats.tokens_per_sec,
        stats.peak_gpu_mem_bytes,
        notes,
    )
