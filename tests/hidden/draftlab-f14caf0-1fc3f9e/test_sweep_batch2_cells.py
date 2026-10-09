"""Hidden check draftlab-f14caf0-1fc3f9e: sweep batch2, lora_dpo k=8 cell complete.

Ticket state/tickets/bf03e6c66b37.md "Done when" (data-level parts (a),(b) of
tools/check_sweep_batch2.py): in results/sweep.parquet, all 16 T=0 cells
(ngram k=1; tiny/base/lora/lora_dpo k in {1,2,4,8}) hold 393 prompts each, and
pooled tpc(k) = sum(n_emit)/sum(n_steps) is non-decreasing in k per model
drafter. At the bad commit the lora_dpo k=8 cell is partial (108/393 in the
parquet, the rest still in the unfolded .part- sidecar).
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(importlib.util.find_spec("draftlab").origin).parent.parent
DF = pd.read_parquet(ROOT / "results" / "sweep.parquet")
N_PROMPTS = 393
KS = [1, 2, 4, 8]
MODELS = ["tiny", "base", "lora", "lora_dpo"]


def cell(drafter: str, k: int) -> pd.DataFrame:
    return DF[(DF["drafter"] == drafter) & (DF["k"] == k) & (DF["temp"] == 0.0)]


def test_a_all_16_t0_cells_present_393_prompts_each():
    missing = []
    for d, ks in [("ngram", [1])] + [(d, KS) for d in MODELS]:
        for k in ks:
            n = len(cell(d, k))
            if n != N_PROMPTS:
                missing.append(f"{d} k={k}: {n}/{N_PROMPTS}")
    assert not missing, "incomplete T=0 cells: " + "; ".join(missing)


def test_b_pooled_tpc_non_decreasing_in_k_per_model_drafter():
    for d in MODELS:
        vals = [
            float(cell(d, k)["n_emit"].sum() / cell(d, k)["n_steps"].sum())
            for k in KS
        ]
        assert all(vals[i + 1] >= vals[i] for i in range(len(KS) - 1)), (
            f"{d}: tpc {['%.4f' % v for v in vals]} decreases in k"
        )
