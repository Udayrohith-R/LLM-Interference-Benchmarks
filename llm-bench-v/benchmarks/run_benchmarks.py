"""
LLM inference benchmark harness — main entry point.

Single run
----------
    python benchmarks/run_benchmarks.py \\
        --backend pytorch \\
        --model distilgpt2 \\
        --device cuda \\
        --dtype float16 \\
        --batch-size 1 \\
        --prompt-tokens 128 \\
        --new-tokens 128

From a YAML/JSON config file
------------------------------
    python benchmarks/run_benchmarks.py --config benchmarks/configs/small.yaml

Sweep mode (enumerate batch/seq combinations from config)
----------------------------------------------------------
    python benchmarks/run_benchmarks.py --config benchmarks/configs/sweep_batch_size.yaml

Outputs written to results/
  latest.json      — last single run (or last sweep entry)
  latest.md        — human-readable Markdown summary
  sweep_<name>.json — full sweep array (only in sweep mode)
  sweep_<name>.md   — comparison Markdown table
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import BenchConfig, SingleRunConfig
from utils.hardware import get_hardware_info
from utils.metrics import SCHEMA_VERSION, BenchResult, BenchStats, compute_bench_stats
from utils.render import render_rich_table, render_summary_md, render_sweep_md, render_sweep_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _make_backend(backend: str):
    if backend == "pytorch":
        from backends.pytorch_runner import PyTorchBackend
        return PyTorchBackend()
    elif backend == "vllm":
        from backends.vllm_runner import VLLMBackend
        return VLLMBackend()
    elif backend == "tensorrt-llm":
        from backends.tensorrt_llm_runner import TensorRTLLMBackend
        return TensorRTLLMBackend()
    else:
        raise ValueError(f"Unknown backend: '{backend}'")


# ---------------------------------------------------------------------------
# Core: run one SingleRunConfig and return a BenchResult
# ---------------------------------------------------------------------------

def run_one(cfg: SingleRunConfig, hardware: dict) -> BenchResult:
    logger.info(
        "Starting run: backend=%s model=%s bs=%d prompt=%d new=%d",
        cfg.backend, cfg.model, cfg.batch_size, cfg.prompt_tokens, cfg.new_tokens,
    )
    backend = _make_backend(cfg.backend)
    backend.load(cfg)
    stats: BenchStats = backend.benchmark(cfg)

    result = BenchResult(
        schema_version=SCHEMA_VERSION,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        config=cfg.model_dump(),
        stats=stats,
        hardware=hardware,
        latencies_ms=stats.latencies_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_single(result: BenchResult, output_dir: Path, name: str = "latest") -> None:
    d = result.to_dict()
    json_path = output_dir / f"{name}.json"
    md_path = output_dir / f"{name}.md"
    json_path.write_text(json.dumps(d, indent=2))
    md_path.write_text(render_summary_md(d))
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)


def _write_sweep(results: list, output_dir: Path, name: str) -> None:
    dicts = [r.to_dict() for r in results]
    json_path = output_dir / f"sweep_{name}.json"
    md_path = output_dir / f"sweep_{name}.md"
    json_path.write_text(json.dumps(dicts, indent=2))
    md_path.write_text(render_sweep_md(dicts))
    logger.info("Wrote sweep JSON  → %s", json_path)
    logger.info("Wrote sweep table → %s", md_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="LLM inference benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default="",
                    help="Path to YAML/JSON BenchConfig file.")
    ap.add_argument("--backend", type=str, default="pytorch",
                    choices=["pytorch", "vllm", "tensorrt-llm"])
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup-runs", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--trt-engine-dir", type=str, default=None,
                    help="Pre-built TensorRT-LLM engine directory.")
    ap.add_argument("--output-dir", type=str, default="results",
                    help="Directory to write JSON/Markdown outputs.")
    ap.add_argument("--name", type=str, default="latest",
                    help="Output file stem (ignored in sweep mode).")
    ap.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # ---- Build BenchConfig from file or CLI flags ----
    if args.config:
        bench_cfg = BenchConfig.from_file(args.config)
        logger.info("Loaded config from %s (name=%s)", args.config, bench_cfg.name)
    else:
        bench_cfg = BenchConfig(
            backend=args.backend,
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            new_tokens=args.new_tokens,
            warmup_runs=args.warmup_runs,
            runs=args.runs,
            trt_engine_dir=args.trt_engine_dir,
            output_dir=args.output_dir,
            name=args.name,
        )

    output_dir = _ensure_output_dir(bench_cfg.output_dir)
    run_configs: List[SingleRunConfig] = bench_cfg.expand_sweep()
    hardware = get_hardware_info()
    is_sweep = bench_cfg.sweep is not None

    logger.info(
        "Hardware: %s  CUDA=%s",
        hardware.get("device_name", hardware.get("cpu_model", "?")),
        hardware.get("cuda_version", "N/A"),
    )

    results: List[BenchResult] = []

    for i, cfg in enumerate(run_configs):
        logger.info(
            "[%d/%d] bs=%d prompt=%d new=%d",
            i + 1, len(run_configs), cfg.batch_size, cfg.prompt_tokens, cfg.new_tokens,
        )
        try:
            result = run_one(cfg, hardware)
        except Exception:
            logger.exception("Run failed — skipping")
            continue

        results.append(result)
        render_rich_table(result)
        _write_single(result, output_dir, name="latest")

    if not results:
        logger.error("All runs failed.  Check the logs above.")
        sys.exit(1)

    if is_sweep and len(results) > 1:
        render_sweep_table(results)
        _write_sweep(results, output_dir, name=bench_cfg.name)


if __name__ == "__main__":
    main()
