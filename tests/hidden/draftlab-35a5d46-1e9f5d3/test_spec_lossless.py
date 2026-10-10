"""Hidden check, draftlab-35a5d46-1e9f5d3.

Lossless T>0 model drafting + one target forward per step. The bad engine
drafts argmax but samples the target unscaled (P = softmax(lg) at T>0) and
pays two target decodes per step. Both tests fail there.
"""
import jax
import numpy as np
from jax.scipy import stats

from draftlab.jaxlm import model as M
from draftlab.jaxlm import spec as S


def _tiny16(seed):
    cfg = M.Qwen2Cfg(vocab_size=16, hidden_size=64, n_layers=2, n_heads=4,
                     n_kv=2, head_dim=16, intermediate_size=128,
                     rope_theta=10000.0, norm_eps=1e-6, tie_word_embeddings=True)
    return cfg, M.init_params(cfg, jax.random.PRNGKey(seed))


def test_identity_model_drafter_lossless_at_T():
    """T=0.7 with a model drafter equal to the target: the first emitted
    token must follow softmax(lg/0.7) (chi-square p > 0.01)."""
    cfg, params = _tiny16(1)
    prompt = [i % 7 for i in range(32)]
    lg, _ = M.prefill(cfg, params, prompt, len(prompt) + 8)
    pT = np.asarray(jax.nn.softmax(lg[-1] / 0.7), np.float64)
    N = 4000
    obs = np.zeros(cfg.vocab_size, np.int64)
    for i in range(N):
        dec = S.SpecDecoder(cfg, params, S.ModelDrafter(cfg, params, len(prompt) + 8),
                            max_len=len(prompt) + 8, k=2, temperature=0.7, seed=i)
        out, _ = dec.generate(prompt, 1)
        obs[out[0]] += 1
    chi2 = float(((obs - N * pT) ** 2 / (N * pT)).sum())
    pval = float(stats.chi2.sf(chi2, df=cfg.vocab_size - 1))
    assert pval > 0.01, f"chi2={chi2:.1f}, p={pval:.4g} (identity drafter not lossless at T)"


def test_one_target_forward_per_step(monkeypatch):
    """Each step scores the pending token plus all drafts in exactly one
    target decode call (k+1 positions), never a second bonus decode."""
    cfg, params = _tiny16(2)
    calls = []
    real = M.decode

    def counting(c, p, cache, ids):
        if p is params:
            calls.append(len(ids))
        return real(c, p, cache, ids)

    monkeypatch.setattr(M, "decode", counting)
    prompt = [i % 7 for i in range(30)]
    dec = S.SpecDecoder(cfg, params, S.NGramDrafter(cfg.vocab_size, max_n=4),
                        max_len=80, k=3, temperature=1.0, seed=1)
    out, st = dec.generate(prompt, 25)
    assert len(out) == 25
    assert len(calls) == len(st), f"{len(calls)} target decodes for {len(st)} steps"
    assert [c - 1 for c in calls] == [s.drafted for s in st]
