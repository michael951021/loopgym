"""Hidden check draftlab-fda7335-5bf9b0d: NOTES.md records the distill
milestone's completion claims — a >=1200-step run with a >=30% val-KL drop,
a teacher-init control with no gain, and an alpha eval of the distilled
drafter in (0, 1]. Claims, not the fix's exact numbers or wording."""
import re
import sys
from pathlib import Path

ROOT = next(p for p in sys.path if p and (Path(p) / "NOTES.md").is_file())
NOTES = (Path(ROOT) / "NOTES.md").read_text()


def distill_sec():
    lines = NOTES.splitlines()
    for i, ln in enumerate(lines):
        if "distill" in ln.lower():
            j = i
            while j > 0 and not lines[j].startswith("#"):
                j -= 1
            k = j + 1
            while k < len(lines) and not lines[k].startswith("#"):
                k += 1
            return "\n".join(lines[j:k])
    return ""


def test_distill_milestone_recorded():
    assert distill_sec().strip(), "NOTES.md has no distill section"


def test_run_of_at_least_1200_steps_recorded():
    sec = distill_sec()
    steps = [int(n) for n in re.findall(r"(\d{3,5})\s*steps?\b", sec, re.I)]
    assert steps and max(steps) >= 1200, f"no >=1200-step run recorded: {steps}"


def test_val_kl_drop_at_least_30pct_recorded():
    sec = distill_sec()
    drops = []
    for a, b in re.findall(r"val\s*KL\s*([\d.]+)\s*(?:→|->|to)\s*([\d.]+)",
                           sec, re.I):
        a, b = float(a), float(b)
        if 0 < b < a:
            drops.append((a - b) / a)
    assert drops and max(drops) >= 0.30, \
        f"no val-KL drop >=30% recorded: {drops}"


def test_teacher_init_control_recorded_with_no_gain():
    sec = distill_sec()
    assert re.search(r"teacher[\s_-]*init", sec, re.I), \
        "teacher-init control missing"
    assert re.search(r"no\s+gain|no\s+improv|no\s+benefit|did\s+not\s+help"
                     r"|not\s+better|\bworse\b", sec, re.I), \
        "teacher-init must be recorded as giving no gain"


def test_alpha_eval_of_distilled_drafter_recorded():
    sec = distill_sec()
    vals = [float(x) for x in re.findall(r"(?:alpha|α)\s*[=:]?\s*([\d.]+)",
                                         sec, re.I)]
    assert any(0 < v <= 1 for v in vals), \
        f"no alpha value in (0,1] recorded in distill section: {vals}"
