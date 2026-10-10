"""Hidden check draftlab-e6895d1-5bf9b0d: NOTES.md records the distill
milestone as done — a >=1200-step run with a >=30% val-KL drop, a
teacher-init control recorded as giving no gain, and an alpha eval of the
distilled drafter (value in (0, 1]). Claims of completion, not exact numbers."""
import re
import sys
from pathlib import Path

ROOT = next(p for p in sys.path if p and (Path(p) / "NOTES.md").is_file())
NOTES = (Path(ROOT) / "NOTES.md").read_text()


def distill_section():
    lines = NOTES.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("#") and "distill" in ln.lower())
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("#"):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def test_distill_section_present():
    assert distill_section().strip(), "no distill section in NOTES.md"


def test_run_recorded_1200_steps_kl_drop_at_least_30pct():
    sec = distill_section()
    steps = [int(n) for n in re.findall(r"(\d{3,5})\s*[-–]?steps?", sec, re.I)]
    assert steps and max(steps) >= 1200, f"no run of >=1200 steps: {steps}"
    drops = []
    for a, b in re.findall(r"val\s*KL\s*([\d.]+)\s*(?:→|->|to)\s*([\d.]+)",
                           sec, re.I):
        a, b = float(a), float(b)
        if 0 < b < a:
            drops.append((a - b) / a)
    assert drops and max(drops) >= 0.30, f"no val-KL drop >=30% recorded: {drops}"


def test_teacher_init_control_recorded_no_gain():
    sec = distill_section()
    assert re.search(r"teacher[\s_-]*init", sec, re.I), "teacher-init control missing"
    assert re.search(r"no\s+gain|no\s+improv|no\s+benefit|did\s+not\s+help"
                     r"|not\s+better|\bworse\b", sec, re.I), \
        "teacher-init must be recorded as giving no gain"


def test_alpha_eval_recorded():
    sec = distill_section()
    vals = [float(x) for x in re.findall(r"(?:alpha|α)\s*[=:]?\s*([\d.]+)",
                                         sec, re.I)]
    assert any(0 < v <= 1 for v in vals), f"no alpha value in (0,1] recorded: {vals}"
