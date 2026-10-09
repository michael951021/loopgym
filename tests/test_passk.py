"""pass@k against hand-computed values and the math.comb reference; summarize
returns per-task pass@k with a task-clustered CI."""
import math
import random

import numpy as np
import pytest

from loopgym.eval.passk import pass_at_k, summarize
from loopgym.eval.stats import bootstrap_ci


def test_pass_at_k_known_answers():
    for k in (1, 2, 3, 5):
        assert pass_at_k(5, 2, 1) == 0.4            # 1 - C(3,1)/C(5,1) = 1 - 3/5
        assert pass_at_k(5, 2, 2) == 0.7            # 1 - C(3,2)/C(5,2) = 1 - 3/10
        assert pass_at_k(5, 0, k) == 0.0
        assert pass_at_k(5, 5, k) == 1.0
        assert pass_at_k(10, 10, k) == 1.0
        assert pass_at_k(10, 0, k) == 0.0


def test_pass_at_k_matches_comb_reference():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 30)
        c = rng.randint(0, n)
        k = rng.randint(1, 30)
        if k > n:
            assert pass_at_k(n, c, k) == 0.0
        else:
            num = math.comb(n - c, k) if k <= n - c else 0  # C(n-c, k) = 0 when k > n-c
            ref = 1 - num / math.comb(n, k)
            assert abs(pass_at_k(n, c, k) - ref) < 1e-12


def test_pass_at_k_increasing_in_k():
    # more draws can only help: pass@k non-decreasing in k for fixed n, c
    for (n, c) in [(20, 1), (20, 3), (50, 5), (100, 0), (7, 7)]:
        vals = [pass_at_k(n, c, k) for k in range(1, n + 1)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_summarize_per_task_values_and_bounds():
    recs = [
        {"task": "a", "n": 10, "c": 2},      # pass@1 = 1 - 8/10 = 0.2
        {"task": "b", "n": 10, "c": 0},      # 0
        {"task": "c", "n": 10, "c": 10},     # 1
    ]
    s = summarize(recs, k=1)
    assert s["per_task"]["a"] == pytest.approx(0.2, abs=1e-12)
    assert s["per_task"]["b"] == 0.0
    assert s["per_task"]["c"] == 1.0
    assert s["pass_at_k"] == pytest.approx(0.4, abs=1e-12)  # mean of (0.2, 0, 1)
    lo, hi = s["ci"]
    assert lo <= s["pass_at_k"] <= hi
    # clustered CI over 3 distinct tasks must be wider than one over 30 samples
    _, lo_n, hi_n = bootstrap_ci(np.repeat([0.2, 0.0, 1.0], 10), n_boot=4000, seed=0)
    assert (hi_n - lo_n) < (hi - lo)


def test_summarize_constant_tasks_give_point_ci():
    recs = [{"task": f"t{i}", "n": 4, "c": 2} for i in range(20)]  # all pass@1 = 1 - C(2,1)/C(4,1) = 0.5
    s = summarize(recs, k=1)
    assert s["pass_at_k"] == 0.5
    assert s["ci"][0] == 0.5 and s["ci"][1] == 0.5
