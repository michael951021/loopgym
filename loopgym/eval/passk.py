"""pass@k (Chen et al. 2021) and task-clustered summary.

pass@k(n, c): unbiased estimate given n samples of which c pass, for the
probability that a fresh draw of k passes at least once:
    E[1 - C(n - c, k) / C(n, k)]
CI is a bootstrap over tasks (groups), never over individual samples."""

from __future__ import annotations

from loopgym.eval.stats import bootstrap_ci


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for n samples, c correct, k draws (Chen et al. 2021)."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if not 0 <= c <= n:
        raise ValueError("need 0 <= c <= n")
    if k > n:
        return 0.0
    # 1 - C(n-c, k)/C(n, k) as a product (no factorial overflow):
    # prod_{i=0}^{k-1} (n - c - i) / (n - i)
    term = 1.0
    for i in range(k):
        term *= (n - c - i) / (n - i)
    return 1.0 - term


def summarize(records, k: int) -> dict:
    """Per-task pass@k plus the task-clustered bootstrap CI of its mean.

    `records`: iterable of dicts with keys "task" (group id), "n" (samples),
    "c" (correct). Returns {"k", "pass_at_k", "ci" (lo, hi), "per_task"}.
    """
    per_task = {}
    values, groups = [], []
    for r in records:
        per_task[r["task"]] = pass_at_k(r["n"], r["c"], k)
        values.append(per_task[r["task"]])
        groups.append(r["task"])
    if not values:
        raise ValueError("records must be non-empty")
    m, lo, hi = bootstrap_ci(values, groups=groups)
    return {"k": k, "pass_at_k": m, "ci": (lo, hi), "per_task": per_task}
