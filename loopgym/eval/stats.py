"""Bootstrap confidence intervals, clustered by task (groups).

Resampling individual samples when each task was measured many times counts
repeats as evidence and makes CIs far too narrow (draftlab shipped ~100x too
narrow). When `groups` is given, each group (task) is resampled as a unit."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values,
    groups=None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and (1 - alpha) central bootstrap CI of the mean of `values`.

    With `groups`, values sharing a group are resampled together, so the CI
    reflects the number of groups, not the number of samples.

    Returns (mean, lo, hi).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    mean = float(values.mean())
    if groups is None:
        ranks = np.arange(values.size)
    else:
        _, ranks = np.unique(np.asarray(groups), return_inverse=True)
    g = int(ranks.max()) + 1

    g_sum = np.zeros(g)
    np.add.at(g_sum, ranks, values)
    g_cnt = np.bincount(ranks, minlength=g).astype(np.float64)

    rng = np.random.default_rng(seed)
    # For each replicate, draw `g` group ids with replacement; the replicate's
    # mean is the weighted mean of group sums by draw counts.
    idx = rng.integers(0, g, size=(n_boot, g))
    means = g_sum[idx].sum(axis=1) / g_cnt[idx].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)
