"""
Regression checker for CI.

Usage
-----
# Save a baseline (e.g. on main branch):
python -m utils.regression save results/latest.json results/baseline.json

# Check a new result against baseline (exits non-zero on regression):
python -m utils.regression check results/latest.json results/baseline.json

Metrics checked
---------------
  mean_ms          — primary latency signal
  p95_ms           — tail-latency signal
  tokens_per_sec   — throughput (regression = decrease)

Thresholds (configurable via env vars or --threshold flag)
-----------
  REGRESSION_LATENCY_PCT   (default 10) — flag if latency increases > N %
  REGRESSION_THROUGHPUT_PCT (default 10) — flag if throughput drops > N %
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_LATENCY_PCT = float(os.environ.get("REGRESSION_LATENCY_PCT", "10"))
DEFAULT_THROUGHPUT_PCT = float(os.environ.get("REGRESSION_THROUGHPUT_PCT", "10"))


def _load(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def check_regression(
    current_path: str | Path,
    baseline_path: str | Path,
    latency_threshold_pct: float = DEFAULT_LATENCY_PCT,
    throughput_threshold_pct: float = DEFAULT_THROUGHPUT_PCT,
) -> bool:
    """
    Compare current to baseline.

    Returns True if all metrics are within threshold (no regression).
    Returns False and logs warnings if any threshold is breached.
    """
    current = _load(current_path)
    baseline = _load(baseline_path)

    cs = current["stats"]
    bs = baseline["stats"]

    regressions = []

    # --- latency checks (higher = worse) ---
    for metric in ("mean_ms", "p95_ms", "p99_ms"):
        c_val = cs.get(metric)
        b_val = bs.get(metric)
        if c_val is None or b_val is None or b_val == 0:
            continue
        pct_change = (c_val - b_val) / b_val * 100.0
        if pct_change > latency_threshold_pct:
            regressions.append(
                f"REGRESSION: {metric} degraded by {pct_change:.1f}% "
                f"({b_val:.2f} ms → {c_val:.2f} ms, threshold ±{latency_threshold_pct}%)"
            )
        else:
            logger.info(
                "%s: %.2f ms → %.2f ms  (%+.1f%%)",
                metric, b_val, c_val, pct_change,
            )

    # --- throughput check (lower = worse) ---
    c_tps = cs.get("tokens_per_sec")
    b_tps = bs.get("tokens_per_sec")
    if c_tps is not None and b_tps is not None and b_tps > 0:
        pct_change = (c_tps - b_tps) / b_tps * 100.0
        if pct_change < -throughput_threshold_pct:
            regressions.append(
                f"REGRESSION: tokens_per_sec dropped by {-pct_change:.1f}% "
                f"({b_tps:.1f} → {c_tps:.1f} tok/s, threshold ±{throughput_threshold_pct}%)"
            )
        else:
            logger.info(
                "tokens_per_sec: %.1f → %.1f  (%+.1f%%)",
                b_tps, c_tps, pct_change,
            )

    if regressions:
        for msg in regressions:
            logger.error(msg)
        return False

    logger.info("No regressions detected (latency threshold: ±%.0f%%, throughput: ±%.0f%%)",
                latency_threshold_pct, throughput_threshold_pct)
    return True


def save_baseline(source_path: str | Path, dest_path: str | Path) -> None:
    """Copy a result JSON to use as the regression baseline."""
    import shutil
    shutil.copy2(source_path, dest_path)
    logger.info("Saved baseline: %s", dest_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    ap = argparse.ArgumentParser(description="Regression checker for benchmark results")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Compare current result to baseline")
    p_check.add_argument("current", help="Path to current result JSON")
    p_check.add_argument("baseline", help="Path to baseline result JSON")
    p_check.add_argument("--latency-threshold", type=float, default=DEFAULT_LATENCY_PCT)
    p_check.add_argument("--throughput-threshold", type=float, default=DEFAULT_THROUGHPUT_PCT)

    p_save = sub.add_parser("save", help="Save a result as the new baseline")
    p_save.add_argument("source", help="Path to result JSON to promote to baseline")
    p_save.add_argument("dest", help="Baseline destination path")

    args = ap.parse_args()

    if args.cmd == "check":
        ok = check_regression(
            args.current,
            args.baseline,
            latency_threshold_pct=args.latency_threshold,
            throughput_threshold_pct=args.throughput_threshold,
        )
        sys.exit(0 if ok else 1)
    elif args.cmd == "save":
        save_baseline(args.source, args.dest)


if __name__ == "__main__":
    _main()
