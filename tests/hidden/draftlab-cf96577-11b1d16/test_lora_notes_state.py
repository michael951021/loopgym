"""Hidden check, draftlab-cf96577-11b1d16: NOTES.md must record the lora a3
step_loss off-by-one fix (HF next-token alignment, retrain + re-eval) with
cited numbers matching results/alpha/lora.json. Stdlib only; worktree =
sys.path entry with NOTES.md + draftlab/; lora.json lazy (absent at bad).
"""
import json
import re
import sys
from pathlib import Path

KIND = {"Edit": "tool_use:Edit", "Write": "tool_use:Write", "Read": "tool_use:Read", "Bash": "tool_use:Bash",
        "tool_result": "tool_result", "code": "code", "thinking": "thinking", "text": "text"}

def _wt():
    w = next((Path(p) for p in sys.path if (Path(p) / "NOTES.md").exists()
              and (Path(p) / "draftlab").is_dir()), None)
    assert w, "draftlab worktree not on sys.path"
    return w

def _sec():
    m = re.search(r"## LoRA[^\n]*\n(.*?)(?=\n## |\Z)", (_wt() / "NOTES.md").read_text(), re.S)
    assert m, "no '## LoRA' section in NOTES.md"
    return m.group(1)

def _j():
    f = _wt() / "results" / "alpha" / "lora.json"
    assert f.exists(), "results/alpha/lora.json missing"
    return json.loads(f.read_text())

def test_off_by_one_fix_recorded():
    s = _sec()
    assert re.search(r"off-by-one", s, re.I), "step_loss off-by-one not recorded"
    assert re.search(r"labels\s*=\s*ids|\bHF\b|next-token", s), "HF alignment fix not recorded"

def test_overall_alpha_ci_match_lora_json():
    s, o = _sec(), _j()["overall"]
    m = re.search(r"(\d\.\d{4})\s*\[\s*(\d\.\d{4})\s*,\s*(\d\.\d{4})\s*\]", s)
    assert m, "no overall alpha + CI in LoRA section"
    a, lo, hi = (float(x) for x in m.groups())
    assert abs(a - o["alpha"]) <= 5e-4, f"cited {a} != lora.json {o['alpha']:.4f}"
    assert abs(lo - o["alpha_ci95"][0]) <= 5e-4 and abs(hi - o["alpha_ci95"][1]) <= 5e-4, "CI mismatch"

def test_tpc_speedup_match_lora_json():
    s, o = _sec(), _j()["overall"]
    t = re.search(r"tok/call@k8\s+(\d\.\d{2})", s)
    sp = re.search(r"\bspeedup\s+(\d\.\d{2})", s)
    assert t and sp, "tok/call@k8 or speedup not recorded"
    assert abs(float(t.group(1)) - o["tokens_per_call"]["8"]) <= 0.005, "tpc@8 mismatch"
    assert abs(float(sp.group(1)) - o["speedup"]["8"]) <= 0.005, "speedup@8 mismatch"

def test_per_kind_numbers_match_lora_json():
    s, bk = _sec(), _j()["by_kind"]
    pairs = re.findall(r"\b(Edit|Write|Read|Bash|tool_result|code|thinking|text)\s+(\d\.\d{2})", s)
    assert pairs, "no per-kind numbers in LoRA section"
    bad = [(k, v) for k, v in pairs if abs(float(v) - bk[KIND[k]]["alpha"]) > 0.005]
    assert not bad, f"per-kind mismatch vs lora.json: {bad}"
