"""Hidden check, draftlab-673042b-5bf9b0d.

Ticket (distill): 1200-step run (val KL 30.6% drop), alpha eval, teacher-init
control. Bad commit has no distill state; the fix completes the run and
records it. This check tests the recorded state: the NOTES distill section
exists, its numbers match the committed result files, the drop clears the
30% bar, the control is recorded and does not beat the main run, and the
alpha eval + self-check (alpha=1.0) are recorded.
"""
import json
import re
import sys
from pathlib import Path


def find(rel: str) -> Path:
    for p in sys.path:
        f = Path(p or ".") / rel
        if f.is_file():
            return f
    raise AssertionError(f"{rel} not on sys.path")


def sec() -> str:
    notes = find("NOTES.md").read_text()
    m = re.search(r"^## [^\n]*[Dd]istill[^\n]*\n(.*?)(?=^## |\Z)", notes, re.M | re.S)
    assert m, "NOTES.md has no distill section"
    return m.group(1)


def test_run_recorded_and_matches_curve():
    s = sec()
    j = json.loads(find("results/distill_curve.json").read_text())
    m = re.search(r"val\s*KL\s*([\d.]+)\s*(?:→|->)\s*([\d.]+)\s*\(([\d.]+)%", s)
    assert m, "main run 'val KL a -> b (pct%)' not recorded"
    s0, sf, pct = map(float, m.groups())
    assert abs(s0 - j["val_kl_step0"]) < 1e-3
    assert abs(sf - j["val_kl_final"]) < 1e-3
    st = re.search(r"(\d{2,5})\s*steps", s)
    assert st and j["steps"] == 1200 and int(st.group(1)) == j["steps"], "step count"
    drop = (j["val_kl_step0"] - j["val_kl_final"]) / j["val_kl_step0"]
    assert drop >= 0.30 and abs(pct - 100 * drop) < 0.05


def test_teacher_init_control_recorded():
    s = sec()
    j = json.loads(find("results/distill_curve_tinit.json").read_text())
    assert j.get("teacher_init") is True and j["steps"] == 1200
    m = re.search(r"[Tt]eacher[- ]init[^0-9]{0,40}([\d.]+)", s)
    assert m, "teacher-init control not recorded"
    assert abs(float(m.group(1)) - j["val_kl_final"]) < 1e-3
    main = json.loads(find("results/distill_curve.json").read_text())
    assert j["val_kl_final"] >= main["val_kl_final"] - 1e-3, "control beats main run"


def test_alpha_eval_recorded():
    s = sec()
    j = json.loads(find("results/alpha/tiny.json").read_text())
    assert j["drafter_type"] == "jax" and j["n_blocks"] == 400
    a = re.search(r"overall[^0-9]{0,8}([\d.]+)", s)
    g = re.search(r"greedy[^0-9]{0,4}([\d.]+)", s)
    sc = re.search(r"self-check[^0-9]{0,8}([\d.]+)", s)
    assert a and g and sc, "alpha eval (overall/greedy/self-check) not recorded"
    assert abs(float(a.group(1)) - j["overall"]["alpha"]) < 1e-3
    assert abs(float(g.group(1)) - j["overall"]["greedy"]) < 1e-3
    assert abs(float(sc.group(1)) - 1.0) < 1e-4, "self-check alpha must be 1.0"
