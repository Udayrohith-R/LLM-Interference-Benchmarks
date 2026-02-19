# LLM Inference Benchmarks on NVIDIA GPUs  
**PyTorch (HF Transformers) vs vLLM (optional) vs TensorRT-LLM (optional)**

This repo is a lightweight, reproducible benchmark harness to measure **latency**, **throughput**, and **GPU memory usage** for LLM text generation on NVIDIA GPUs.

It’s designed to be honest and practical:
- Runs **PyTorch + Hugging Face** by default  
- Runs **vLLM** if installed  
- Includes a **TensorRT-LLM placeholder runner** (so you can integrate engines when ready)  
- Produces a single JSON report suitable for resumes, reports, or performance analysis  

> **Note:** LLM inference performance depends heavily on GPU type, driver/CUDA versions, batch size, prompt length, and generation length.  
> This project focuses on making results **reproducible** and **comparable** across backends.

---

## Quick Start (PyTorch baseline)

### 1) Create env and install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
