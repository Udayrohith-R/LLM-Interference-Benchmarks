"""
TensorRT-LLM backend.

Prerequisites
-------------
TensorRT-LLM requires a pre-built engine directory.  Build one with the
trtllm-build CLI (see TRT-LLM docs), then pass --trt-engine-dir to this
harness.  Example:

    # 1. Clone TensorRT-LLM and install.
    # 2. Build an engine:
    trtllm-build \\
        --checkpoint_dir ./llama3-8b-hf \\
        --output_dir ./engines/llama3-8b-fp16 \\
        --dtype float16 \\
        --max_batch_size 8 \\
        --max_input_len 512 \\
        --max_output_len 256

    # 3. Run this harness:
    python benchmarks/run_benchmarks.py \\
        --backend tensorrt-llm \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --trt-engine-dir ./engines/llama3-8b-fp16 \\
        --batch-size 4 \\
        --prompt-tokens 256 \\
        --new-tokens 128

Timing methodology
------------------
TRT-LLM's ModelRunner.generate() is fully synchronous from the Python side
after the internal CUDA stream flush.  We use CUDA Events around
runner.generate() for accurate GPU-side latency.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from backends.base import InferenceBackend
from utils.config import SingleRunConfig
from utils.metrics import TokenTimingStats, now_ms

logger = logging.getLogger(__name__)


class TensorRTLLMBackend(InferenceBackend):
    """
    TensorRT-LLM ModelRunner backend.

    The runner is importable only when tensorrt_llm is installed
    (https://github.com/NVIDIA/TensorRT-LLM).  All imports are deferred
    to load() so that importing this module does not crash in environments
    without TRT-LLM.
    """

    def __init__(self) -> None:
        self._runner = None
        self._input_ids = None
        self._use_cuda = False

    # ------------------------------------------------------------------
    # InferenceBackend.load
    # ------------------------------------------------------------------

    def load(self, config: SingleRunConfig) -> None:
        if not config.trt_engine_dir:
            raise ValueError(
                "TensorRT-LLM backend requires --trt-engine-dir pointing to a "
                "pre-built engine directory.  See backends/tensorrt_llm_runner.py "
                "for build instructions."
            )

        engine_dir = Path(config.trt_engine_dir)
        if not engine_dir.is_dir():
            raise FileNotFoundError(f"TRT engine directory not found: {engine_dir}")

        try:
            import tensorrt_llm  # noqa: F401
            from tensorrt_llm.runtime import ModelRunner
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt_llm is not installed.  Follow the official TRT-LLM "
                "installation guide: https://github.com/NVIDIA/TensorRT-LLM"
            ) from exc

        logger.info("Loading TRT-LLM engine from %s", engine_dir)
        self._runner = ModelRunner.from_dir(
            engine_dir=str(engine_dir),
            rank=0,
        )

        self._use_cuda = config.device.startswith("cuda")
        self._input_ids = self._build_input_ids(config)
        logger.info("TRT-LLM engine loaded")

    # ------------------------------------------------------------------
    # InferenceBackend.run_single
    # ------------------------------------------------------------------

    def run_single(
        self, config: SingleRunConfig
    ) -> Tuple[float, Optional[TokenTimingStats]]:
        assert self._runner is not None, "call load() before run_single()"

        import torch

        if self._use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        t0 = now_ms()

        # TRT-LLM ModelRunner.generate() accepts a list of 1-D token tensors.
        batch = [self._input_ids[i] for i in range(self._input_ids.shape[0])]
        self._runner.generate(
            batch_input_ids=batch,
            max_new_tokens=config.new_tokens,
            end_id=-1,              # no early stopping
            pad_id=0,
            temperature=1.0,
            top_k=1,                # greedy
        )

        if self._use_cuda:
            end_evt.record()
            torch.cuda.synchronize()
            latency_ms = start_evt.elapsed_time(end_evt)
        else:
            latency_ms = now_ms() - t0

        # TRT-LLM does not expose per-token timing through ModelRunner.
        # TTFT / TPOT would require hooking into the C++ runtime directly
        # or using the profiling hooks in tensorrt_llm.profiler.
        return latency_ms, None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_input_ids(self, config: SingleRunConfig):
        """
        Build a synthetic (batch_size, prompt_tokens) int32 tensor.

        TRT-LLM engines are built with a fixed vocab size; we use token ID 1
        (typically a safe non-special token) tiled to prompt_tokens length.
        Replace this with real tokeniser output in production.
        """
        import torch

        ids = torch.ones(
            (config.batch_size, config.prompt_tokens),
            dtype=torch.int32,
            device=config.device,
        )
        return ids
