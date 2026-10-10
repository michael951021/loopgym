"""The repo's T=1 chi-square losslessness test must exercise the residual
resample path. At 8a2123e the drafter reused the target's own params, so
q == p, every draft was accepted, and the chi-square degenerated to "target
sampling matches target" — a vacuous check that never ran accept/reject.
Restored behavior: the chi-square test uses a drafter that differs from the
target, so its spec_generate call has rejecting rounds (drafted > accepted).
We run the repo's own chi-square test with spec_generate wrapped to capture
the result, then assert >= 1 rejecting round and (independently) that the
stream is lossless: chi-square p > 0.01 vs the recorded target distributions.
n_tokens is clamped to 6000 for the 120 s verify window; rejection is a
per-round property of q != p, independent of stream length.
"""
import importlib


def test_chi_square_exercises_residual_resample():
    import numpy as np
    from scipy import stats

    ts = importlib.import_module("draftlab.tests.test_spec")
    cands = [f for n, f in vars(ts).items()
             if n.startswith("test_") and "chi" in n.lower()]
    assert cands, f"no chi-square test in {ts.__file__}"
    fixture = getattr(ts, "_tiny16", None) or getattr(ts, "tiny16", None)
    assert callable(fixture), "no tiny16 fixture in test_spec"
    cfg_params = fixture()

    spec = ts.S
    real, captured = spec.spec_generate, []

    def spy(*a, **kw):
        if "n_tokens" in kw:
            kw["n_tokens"] = min(int(kw["n_tokens"]), 6000)
        elif len(a) > 5:
            a = list(a)
            a[5] = min(int(a[5]), 6000)
            a = tuple(a)
        res = real(*a, **kw)
        captured.append(res)
        return res

    spec.spec_generate = spy
    try:
        cands[0](cfg_params)  # the repo's own assertions must hold too
    finally:
        spec.spec_generate = real

    assert captured, "chi-square test never called spec_generate"
    res = captured[-1]
    rej = sum(1 for st in res.steps if st.drafted > st.accepted)
    assert rej > 0, (f"vacuous chi-square: 0 rejecting rounds of {len(res.steps)}; "
                     "drafter matches target, residual resample never exercised")
    if res.probs is not None:
        v = np.asarray(res.probs).shape[-1]
        counts = np.bincount(res.tokens, minlength=v).astype(float)
        expected = np.asarray(res.probs).sum(axis=0)
        chi2 = float(((counts - expected) ** 2 / np.maximum(expected, 1e-6)).sum())
        p = float(stats.chi2.sf(chi2, v - 1))
        assert p > 0.01, f"stream not lossless at T=1: chi2={chi2:.1f}, p={p:.3g}"
