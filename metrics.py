from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def try_cuda_peak_memory() -> Optional[int]:
    """Return peak allocated bytes if torch+cuda available, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return 0
    except Exception:
        return None
    return None

def get_cuda_peak_memory_bytes() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None

@dataclass
class BenchResult:
    backend: str
    model: str
    device: str
    dtype: str
    batch_size: int
    prompt_tokens: int
    new_tokens: int
    warmup_runs: int
    runs: int
    avg_latency_ms: float
    p95_latency_ms: float
    tokens_per_sec: float
    peak_gpu_mem_bytes: Optional[int]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
