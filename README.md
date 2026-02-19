LLM Inference Benchmarks on NVIDIA GPUs

PyTorch (HF Transformers) vs vLLM (optional) vs TensorRT-LLM (optional)

This repo is a lightweight, reproducible benchmark harness to measure latency, throughput, and GPU memory usage for LLM text generation on NVIDIA GPUs.

It’s designed to be honest and practical:

Runs PyTorch + Hugging Face by default

Runs vLLM if installed

Includes a TensorRT-LLM placeholder runner (so you can integrate engines when ready)

Produces a single JSON report suitable for resumes, reports, or performance analysis

Note: LLM inference performance depends heavily on GPU type, driver/CUDA versions, batch size, prompt length, and generation length.
This project focuses on making results reproducible and comparable across backends.

Quick Start (PyTorch baseline)
1) Create env and install deps
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
2) Run a benchmark (GPU recommended)
python benchmarks/run_benchmarks.py --backend pytorch --model distilgpt2 --device cuda

This writes:

results/latest.json — full machine-readable report

results/latest.md — human-friendly summary

Optional: vLLM backend

vLLM is optional because it has platform-specific wheels.

pip install vllm
python benchmarks/run_benchmarks.py --backend vllm --model mistralai/Mistral-7B-Instruct-v0.2 --device cuda
TensorRT-LLM (optional)

TensorRT-LLM requires additional setup and engine build steps.
This repo includes a placeholder runner under backends/tensorrt_llm_runner.py so you can integrate TensorRT-LLM engines once available.

What this measures

Prefill + decode latency (wall-clock)

Tokens/sec (approx throughput)

Peak GPU memory usage (if CUDA is available)

Prompt length / generation length / batch size

Suggested NVIDIA interview talking points

How batch size and sequence length affect throughput vs latency

Why KV-cache and attention dominate decode time

How to profile bottlenecks using Nsight Systems / Nsight Compute

Tradeoffs between eager PyTorch, vLLM paged attention, and TensorRT-LLM engines

Repo structure

benchmarks/run_benchmarks.py — main CLI

benchmarks/configs/*.json — benchmark presets

backends/ — runtime-specific backends

utils/metrics.py — latency, throughput, GPU memory helpers

results/ — output artifacts

License

MIT
