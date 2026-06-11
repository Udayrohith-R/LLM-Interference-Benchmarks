"""
Benchmark configuration schema.

BenchConfig is the single source of truth for a run (or sweep).  It can be
loaded from a YAML/JSON file, constructed from CLI args, or built in code.
Pydantic v2 enforces types and cross-field constraints at construction time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Scalar run config
# ---------------------------------------------------------------------------

Backend = Literal["pytorch", "vllm", "tensorrt-llm"]
DType = Literal["float16", "bfloat16", "float32"]


class SingleRunConfig(BaseModel):
    """Parameters for one (backend, model, batch_size, seq_len) combination."""

    backend: Backend = "pytorch"
    model: str = "distilgpt2"
    device: str = "cuda"
    dtype: DType = "float16"

    batch_size: int = Field(1, ge=1)
    prompt_tokens: int = Field(128, ge=1)
    new_tokens: int = Field(128, ge=1)
    warmup_runs: int = Field(2, ge=0)
    runs: int = Field(5, ge=1)

    # Optional TRT-LLM engine path (only used by tensorrt-llm backend).
    trt_engine_dir: Optional[str] = None

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        valid_prefixes = ("cuda", "cpu", "mps")
        if not any(v.startswith(p) for p in valid_prefixes):
            raise ValueError(f"device must start with one of {valid_prefixes}, got '{v}'")
        return v

    @model_validator(mode="after")
    def _cross_validate(self) -> "SingleRunConfig":
        if self.backend == "tensorrt-llm" and self.trt_engine_dir is None:
            # Allow None so the runner can raise a clear error at runtime.
            pass
        return self


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

class SweepDimension(BaseModel):
    """Defines the axes to sweep.  All combinations are enumerated."""

    batch_sizes: List[int] = Field(default_factory=lambda: [1])
    prompt_token_variants: List[int] = Field(default_factory=lambda: [128])
    new_token_variants: List[int] = Field(default_factory=lambda: [128])

    @field_validator("batch_sizes", "prompt_token_variants", "new_token_variants", mode="before")
    @classmethod
    def _non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("sweep dimension lists must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Top-level bench config
# ---------------------------------------------------------------------------

class BenchConfig(SingleRunConfig):
    """
    Full benchmark config, optionally with a sweep definition.

    If `sweep` is present, `batch_size`, `prompt_tokens`, and `new_tokens`
    are used as defaults/documentation; the actual matrix comes from sweep.
    """

    name: str = "benchmark"
    output_dir: str = "results"
    sweep: Optional[SweepDimension] = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchConfig":
        path = Path(path)
        raw = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls.model_validate(data)

    def expand_sweep(self) -> List[SingleRunConfig]:
        """Return a flat list of SingleRunConfig for each sweep combination."""
        if self.sweep is None:
            return [SingleRunConfig.model_validate(self.model_dump(exclude={"sweep", "name", "output_dir"}))]

        runs: List[SingleRunConfig] = []
        for bs in self.sweep.batch_sizes:
            for pt in self.sweep.prompt_token_variants:
                for nt in self.sweep.new_token_variants:
                    cfg = self.model_copy(
                        update={"batch_size": bs, "prompt_tokens": pt, "new_tokens": nt}
                    )
                    runs.append(
                        SingleRunConfig.model_validate(
                            cfg.model_dump(exclude={"sweep", "name", "output_dir"})
                        )
                    )
        return runs
