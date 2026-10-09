"""Hidden check, draftlab-alpha.

Asserts HFDrafter.dists returns the row that predicts each block token: the
per-position cross-entropy must equal the one implied by HF's labels= loss
(row t-1 predicts token t). Fails at the bad commit (off-by-one alignment:
dists slices logits[0,-n:] instead of rows len(ctx)-1..len(ctx)+n-2); passes
at the fix. Real 0.5B checkpoint, CPU, runs in seconds.
"""
import math
import os

import numpy as np

from draftlab.eval.alpha import HFDrafter

MODEL = os.environ.get("ALPHA_MODEL", "/work/models/qwen2.5-coder-0.5b-instruct")
CTX = 8
L = 16
FULL = list(range(1000, 1000 + L))  # fixed ids: deterministic, alignment is model-agnostic
IDS = FULL[CTX:]
N = len(IDS)


def test_alignment():
    drafter = HFDrafter(MODEL)
    torch = drafter.torch
    with torch.no_grad():
        logits = drafter.model(torch.tensor([FULL])).logits[0]
        loss_hf = float(drafter.model(torch.tensor([FULL]), labels=torch.tensor([FULL])).loss)
    lp = logits.float().numpy()
    rmax = lp.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(lp - rmax).sum(axis=-1)) + rmax[:, 0]

    def ce(t: int) -> float:
        # HF convention: logits row t-1 is the distribution over token t
        return float(lse[t - 1] - lp[t - 1, FULL[t]])

    # Anchor: this convention reproduces model(labels=).loss over all positions.
    assert abs(np.mean([ce(t) for t in range(1, L)]) - loss_hf) < 1e-2

    p, valid = drafter.dists(FULL[:CTX], IDS)
    assert valid.all(), f"all block positions must be valid, got {valid!r}"
    for j in range(N):
        got = -math.log(float(p[j, IDS[j]]) + 1e-30)
        want = ce(CTX + j)
        assert abs(got - want) < 1e-3, f"position {j}: CE {got} != {want}"
