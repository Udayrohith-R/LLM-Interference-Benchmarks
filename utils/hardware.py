"""
Hardware fingerprinting.

Captures GPU model, memory, SM count, CUDA/driver version, and host CPU info
so that every BenchResult is self-describing and results from different
machines can't be confused with each other.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict, Optional


def _run(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not_installed"


def _cuda_info() -> Dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}

        props = torch.cuda.get_device_properties(0)
        info: Dict[str, Any] = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "device_count": torch.cuda.device_count(),
            "device_name": props.name,
            "total_memory_bytes": props.total_memory,
            "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
            "sm_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
            "max_threads_per_sm": props.max_threads_per_multi_processor,
        }

        # Driver version via nvml (optional; pynvml is not in base requirements).
        try:
            import pynvml  # type: ignore[import]

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            info["sm_clock_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(
                h, pynvml.NVML_CLOCK_SM
            )
            info["mem_clock_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(
                h, pynvml.NVML_CLOCK_MEM
            )
            info["memory_bus_width_bits"] = pynvml.nvmlDeviceGetMemoryBusWidth(h)
            # Theoretical memory bandwidth (GB/s)
            bw = (
                info["mem_clock_mhz"] * 1e6
                * info["memory_bus_width_bits"]
                / 8
                / 1e9
                * 2  # DDR
            )
            info["theoretical_mem_bw_gbps"] = round(bw, 1)
        except Exception:
            # pynvml absent or query failed — not fatal.
            pass

        return info

    except ImportError:
        return {"cuda_available": False, "torch_version": "not_installed"}


def get_hardware_info() -> Dict[str, Any]:
    """Return a fully serialisable dict describing the current host hardware."""
    uname = platform.uname()
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "os": f"{uname.system} {uname.release}",
        "arch": uname.machine,
        "cpu_model": platform.processor() or uname.processor or "unknown",
        "cpu_logical_cores": _logical_core_count(),
    }
    info.update(_cuda_info())
    return info


def _logical_core_count() -> Optional[int]:
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None
