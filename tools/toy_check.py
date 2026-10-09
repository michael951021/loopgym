"""Verdict for the toy known-answer runs (CLAUDE.md planted effect + null control).

Reads runs/toy_banana/metrics.jsonl and runs/toy_null/metrics.jsonl (training lines
only; eval lines carry kind == "eval"), prints the mean reward of steps 0-4 vs
55-59 and the bootstrap 95% CI of the per-step reward_mean slope, and exits 1 unless
the banana mean rises < 0.2 -> > 0.8 and the null slope CI contains 0.

Usage: python tools/toy_check.py [--dir runs]
"""
import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 2000
BOOT_SEED = 0
WINDOWS = [(0, 4), (55, 59)]


def load_pairs(path: Path) -> list[tuple[int, float]]:
    """(step, reward_mean) of training lines, sorted by step."""
    pairs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "eval":
            continue
        pairs.append((int(rec["step"]), float(rec["reward_mean"])))
    pairs.sort()
    return pairs


def window_mean(pairs: list[tuple[int, float]], lo: int, hi: int) -> float | None:
    vals = [r for s, r in pairs if lo <= s <= hi]
    return float(np.mean(vals)) if vals else None


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def slope_bootstrap(pairs: list[tuple[int, float]],
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """(slope, lo, hi): OLS slope of reward_mean on step, CI by resampling steps."""
    x = np.asarray([s for s, _ in pairs], dtype=np.float64)
    y = np.asarray([r for _, r in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    slopes = [ols_slope(x, y)]
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 2:  # degenerate resample; skip
            continue
        slopes.append(ols_slope(x[i], y[i]))
    slopes = np.asarray(slopes)
    return float(slopes[0]), float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def judge(name: str, path: Path) -> tuple[bool, str]:
    if not path.is_file():
        print(f"{name}: missing {path}")
        return False, f"{name} missing {path}"
    pairs = load_pairs(path)
    early = window_mean(pairs, *WINDOWS[0])
    late = window_mean(pairs, *WINDOWS[1])
    slope, lo, hi = slope_bootstrap(pairs)
    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"{name}: steps 0-4 {fmt(early)}  steps 55-59 {fmt(late)}  "
          f"slope {slope:.5f} [{lo:.5f}, {hi:.5f}]")
    if name == "toy_banana":
        ok = (early is not None and late is not None and early < 0.2 and late > 0.8)
        why = "< 0.2 -> > 0.8"
    else:
        ok = lo <= 0.0 <= hi
        why = "slope CI contains 0"
    print(f"{name}: {'PASS' if ok else 'FAIL'} (want {why})")
    return ok, f"{name} {why}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    ok1, _ = judge("toy_banana", d / "toy_banana" / "metrics.jsonl")
    ok2, _ = judge("toy_null", d / "toy_null" / "metrics.jsonl")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
