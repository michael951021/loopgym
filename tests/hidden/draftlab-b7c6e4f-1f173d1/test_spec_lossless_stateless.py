"""Hidden check, draftlab-b7c6e4f-1f173d1: lossless T>0 drafting (spec.py
absent at the bad commit) — the first token via the m>=1 accept/reject +
resample path must stay ~ the target's p; NGramDrafter must be stateless."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy import stats

from draftlab.jaxlm import model as M

N = 20_000


def _spec():
    try:
        from draftlab.jaxlm import spec as S
    except Exception as e:
        pytest.fail(f"draftlab.jaxlm.spec unavailable: {e!r}")
    return S


def _tiny(seed=7):
    cfg = M.Qwen2Cfg(vocab_size=16, hidden_size=64, n_layers=2, n_heads=4,
                     n_kv=2, head_dim=16, intermediate_size=128,
                     rope_theta=10000.0, norm_eps=1e-6, tie_word_embeddings=True)
    return cfg, M.init_params(cfg, jax.random.PRNGKey(seed))


def test_first_token_chi2_m1():
    """Round 1 drafts m=1 (FixedDrafter); the first token must be ~p (p > 0.01)."""
    S = _spec()
    cfg, params = _tiny()
    prompt = [1, 2, 3, 4, 1, 2, 3, 4]
    lg, cb = M.prefill(cfg, params, prompt, 32)
    x = np.asarray(lg)[-1].astype(np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    t = int(np.argmin(np.abs(p - 0.3)))
    counts, acc_tot = np.zeros(16, np.int64), 0
    for i in range(N):
        cache = jax.tree.map(jnp.asarray, cb)
        res = S.spec_generate(cfg, params, cache, prompt, S.FixedDrafter(t, 16),
                              k=1, n_new=2, temperature=1.0, rng=1000 + i)
        assert res.steps[0].drafted == 1, "round 1 must draft m=1"
        counts[res.tokens[0]] += 1
        acc_tot += res.steps[0].accepted
    chi2 = float(((counts - N * p) ** 2 / (N * p)).sum())
    pval = float(stats.chi2.sf(chi2, df=15))
    assert 0 < acc_tot < N, f"need both accept and reject (acc={acc_tot}/{N})"
    assert pval > 0.01, f"chi2={chi2:.1f} p={pval:.3g} — first token biased, not ~p"


def test_ngram_propose_stateless():
    """A second context with no internal repeats must yield no drafts."""
    S = _spec()
    d = S.NGramDrafter(vocab_size=16)
    d.propose([1, 2, 3, 4, 1, 2, 3, 4], 2)  # learns grams
    p2 = d.propose([7, 8, 9, 1, 2, 3, 4], 2)  # no internal repeats
    assert len(p2.tokens) == 0, f"draft state leaked across calls: {p2.tokens}"
