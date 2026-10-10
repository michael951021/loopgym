"""Hidden check, draftlab-cf96577-11b1d16.

NOTES.md must record the lora a3 state: step_loss off-by-one (token-echoer)
caught by the anchor check and fixed to HF next-token alignment, retrain +
re-eval done, cited numbers matching results/alpha/lora.json. Stdlib only;
the draftlab worktree is the sys.path entry holding NOTES.md + draftlab/.
"""
import json
import re
import sys
from pathlib import Path

KIND = {"Edit": "tool_use:Edit", "Write": "tool_use:Write",
        "Read": "tool_use:Read", "Bash": "tool_use:Bash",
        "tool_result": "tool_result", "code": "code",
        "thinking": "thinking", "text": "text"}


def _wt():
    for p in sys.path:
        d = Path(p or ".")
        if (d / "NOTES.md").exists() and (d / "draftlab").is_dir():
            return d
    raise AssertionError("draftlab worktree (NOTES.md + draftlab/) not on sys.path")


def _section():
    n = (_wt() / "NOTES.md").read_text()
    m = re.search(r"## LoRA[^\n]*\n(.*?)(?=\n## |\Z)", n, re.S)
    assert m, "no '## LoRA' section in NOTES.md"
    return m.group(1)


def _lora_json():
    p = _wt() / "results" / "alpha" / "lora.json"
    assert p.exists(), "results/alpha/lora.json missing"
    return json.loads(p.read_text())


def test_off_by_one_fix_recorded():
    s = _section()
    assert re.search(r"off-by-one", s, re.I), "step_loss off-by-one not recorded"
    assert re.search(r"labels\s*=\s*ids|\bHF\b|next-token", s), \
        "fix to HF next-token alignment not recorded"


def test_overall_alpha_ci_match_lora_json():
    s, j = _section(), _lora_json()
    m = re.search(r"(\d\.\d{4})\s*\[\s*(\d\.\d{4})\s*,\s*(\d\.\d{4})\s*\]", s)
    assert m, "no overall alpha + CI in LoRA section"
    o = j["overall"]
    a, lo, hi = (float(x) for x in m.groups())
    assert abs(a - o["alpha"]) <= 0.0005, f"cited {a} != lora.json {o['alpha']:.4f}"
    assert abs(lo - o["alpha_ci95"][0]) <= 0.0005, "CI lo mismatch"
    assert abs(hi - o["alpha_ci95"][1]) <= 0.0005, "CI hi mismatch"


def test_tpc_speedup_match_lora_json():
    s, j = _section(), _lora_json()
    t = re.search(r"tok/call@k8\s+(\d\.\d{2})", s)
    sp = re.search(r"\bspeedup\s+(\d\.\d{2})", s)
    assert t and sp, "tok/call@k8 or speedup not recorded"
    o = j["overall"]
    assert abs(float(t.group(1)) - o["tokens_per_call"]["8"]) <= 0.005, "tpc@8 mismatch"
    assert abs(float(sp.group(1)) - o["speedup"]["8"]) <= 0.005, "speedup@8 mismatch"


def test_per_kind_numbers_match_lora_json():
    s, j = _section(), _lora_json()
    pairs = re.findall(
        r"\b(Edit|Write|Read|Bash|tool_result|code|thinking|text)\s+(\d\.\d{2})", s)
    assert pairs, "no per-kind numbers in LoRA section"
    for k, v in pairs:
        a = j["by_kind"][KIND[k]]["alpha"]
        assert abs(float(v) - a) <= 0.005, f"{k}: cited {v} != lora.json {a:.4f}"
