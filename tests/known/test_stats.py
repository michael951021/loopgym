"""Bootstrap CIs (loopgym/eval/stats.py). draftlab shipped CIs ~100x too narrow; these pin the definition."""
import math

import numpy as np

from loopgym.eval.stats import bootstrap_ci


def test_constant_values_give_a_point():
    m, lo, hi = bootstrap_ci([0.7] * 30)
    assert abs(m - 0.7) < 1e-12 and abs(lo - 0.7) < 1e-12 and abs(hi - 0.7) < 1e-12


def test_binomial_width_matches_normal_approximation():
    v = [0, 1] * 50                                       # p = 0.5, n = 100: 95% half-width ~ 1.96 * 0.05 = 0.098
    m, lo, hi = bootstrap_ci(v, n_boot=4000, seed=1)
    assert m == 0.5 and 0.07 < (hi - lo) / 2 < 0.13


def test_width_shrinks_like_one_over_sqrt_n():
    rng = np.random.default_rng(0)
    w = []
    for n in (50, 800):
        _, lo, hi = bootstrap_ci(rng.random(n), n_boot=3000, seed=2)
        w.append(hi - lo)
    assert 2.8 < w[0] / w[1] < 5.2                        # sqrt(800 / 50) = 4


def test_cluster_bootstrap_does_not_count_repeats_as_evidence():
    """10 tasks, each sampled 20 times with the same outcome: the CI must be about as wide as 10 samples give,
    not 200 (resample tasks, not samples)."""
    task_vals = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    v = np.repeat(task_vals, 20)
    groups = np.repeat(np.arange(10), 20)
    _, lo_c, hi_c = bootstrap_ci(v, groups=groups, n_boot=4000, seed=3)
    _, lo_t, hi_t = bootstrap_ci(task_vals, n_boot=4000, seed=3)
    _, lo_n, hi_n = bootstrap_ci(v, n_boot=4000, seed=3)
    assert abs((hi_c - lo_c) - (hi_t - lo_t)) < 0.08
    assert (hi_n - lo_n) < 0.5 * (hi_c - lo_c)             # the naive one is far too narrow
