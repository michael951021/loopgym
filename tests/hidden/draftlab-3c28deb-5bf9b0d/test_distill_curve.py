"""Hidden check, draftlab-3c28deb-5bf9b0d.

Ticket (distill): train the tiny student; Done when val KL falls >= 30% from
step 0. The bad commit's best committed run (800 steps) reaches only a
29.57% drop, below the ticket's own threshold; the fix's 1200-step run
reaches 30.6%. Data check on the committed curve: the drop, and internal
consistency (endpoints match the curve, no cherry-picked final point).
"""
import json
import sys
from pathlib import Path


def curve() -> dict:
    for p in sys.path:
        f = Path(p or ".") / "results" / "distill_curve.json"
        if f.is_file():
            return json.loads(f.read_text())
    raise AssertionError("results/distill_curve.json not on sys.path")


def test_val_kl_drop_at_least_30pct():
    j = curve()
    s0, sf = j["val_kl_step0"], j["val_kl_final"]
    assert 0 < sf < s0, "val KL must go down"
    assert j["kl_drop_frac"] >= 0.30, f"drop {j['kl_drop_frac']:.4f} < 0.30"


def test_reported_numbers_match_curve():
    j = curve()
    c = j["curve"]
    assert len(c) >= 8
    assert c[0]["step"] == 0
    assert abs(c[0]["val_kl"] - j["val_kl_step0"]) < 1e-6
    assert c[-1]["step"] == j["steps"]
    assert abs(c[-1]["val_kl"] - j["val_kl_final"]) < 1e-6
    assert abs(j["kl_drop_frac"] - (j["val_kl_step0"] - j["val_kl_final"]) / j["val_kl_step0"]) < 1e-6


def test_final_point_is_best_val_not_cherry_picked():
    j = curve()
    c = j["curve"]
    assert c[-1]["val_kl"] == min(p["val_kl"] for p in c)
