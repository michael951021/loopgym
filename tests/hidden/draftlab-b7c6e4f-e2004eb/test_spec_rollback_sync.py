"""Hidden check, draftlab-b7c6e4f-e2004eb: NGramDrafter sync after rollback
must keep proposals == a fresh first-occurrence matcher over the committed
ctx through all commit shapes (all-accepted, a+1==d, first-rejected); an empty
prompt raises ValueError (spec.py absent at the bad commit, so lazy-import)."""
import jax
import pytest

from draftlab.jaxlm import model as M


def _spec():
    try:
        from draftlab.jaxlm import spec as S
    except Exception as e:
        pytest.fail(f"draftlab.jaxlm.spec unavailable: {e!r}")
    return S


def _tiny(seed=7):
    cfg = M.Qwen2Cfg(vocab_size=16, hidden_size=64, n_layers=2, n_heads=4, n_kv=2,
                     head_dim=16, intermediate_size=128, rope_theta=1e4,
                     norm_eps=1e-6, tie_word_embeddings=True)
    return cfg, M.init_params(cfg, jax.random.PRNGKey(seed))


def _ref(ctx, k, max_n=4):
    h, out = list(ctx), []
    for _ in range(k):
        if len(h) < max_n:  # match_end needs history >= max_n
            break
        ans = next((ctx[i + n] for n in range(min(max_n, len(h)), 0, -1)
                   for i in range(len(ctx) - n) if ctx[i:i + n] == h[-n:]), None)
        if ans is None:
            break
        out.append(ans)
        h.append(ans)
    return out


def test_rollback_sync_all_commit_shapes():
    S = _spec()
    d = S.NGramDrafter(vocab_size=16, max_n=4)
    c0 = [1, 2, 3, 4, 5, 1, 2]
    assert _ref(c0, 3) == [3, 4, 5], "oracle sanity (hand-computed)"
    c = c0
    for grow in ([3, 4, 5, 1], [2, 3, 7], [8], [3, 4, 9], [8, 3, 9], [3, 4, 9]):
        c = c + grow
        toks, probs = d.propose(c, 3)
        want = _ref(c, 3)
        assert toks == want, f"got {toks}, want {want} after rollback sync"
        assert all(p[t] == 1.0 and abs(float(p.sum()) - 1.0) < 1e-6
                   for p, t in zip(probs, toks))


def test_empty_prompt_raises_valueerror():
    S = _spec()
    cfg, params = _tiny()
    with pytest.raises(ValueError):
        S.speculative_generate(cfg, params, S.NGramDrafter(cfg.vocab_size), [], 3, 4)
