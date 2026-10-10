"""Freeze loopgym v0: the fixed dataset version the evals and REPORT.md refer to.

Artifacts:
  data/loopgym/v0/tasks.jsonl   one record per verified, representable task
  data/loopgym/v0/checks/<id>/  byte copies of tests/hidden/<id>/test_*.py
  data/loopgym/v0/DATASET.md    source, verification rule, redaction, kill verdict
  results/loopgym_v0.json       counts by repo/kind, split sizes, mean diff size

Split: h = int(sha1(id), 16) % 100 -> "test" iff h >= 70 (30%), else "train".
Kill criterion: < 25 verified tasks -> loopgym is eval-only (training uses
Exercism); DATASET.md states the outcome.

`--check` re-derives everything from verified.jsonl/candidates.jsonl/
tests/hidden/ (via load_tasks) and compares against the frozen files; any
mismatch exits 1.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from loopgym.data import verify as V
from loopgym.env.loopgym import load_tasks

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data" / "loopgym" / "v0"
RESULTS = ROOT / "results" / "loopgym_v0.json"
DATASET_MD = V0 / "DATASET.md"
MIN_TASKS = 25
TEST_BUCKET = 70  # h >= 70 -> test (30% of id space)

_KIND_EXT = {}
for _k, _exts in {
    "code": {"py", "rs", "cpp", "hpp", "c", "h", "go", "rb", "ts", "tsx", "js"},
    "config": {"tf", "hcl", "yaml", "yml", "toml", "json", "ini", "conf", "env", "txt"},
    "shell": {"sh", "bash"},
    "docs": {"md", "rst", "tex"},
}.items():
    for _e in _exts:
        _KIND_EXT[_e] = _k
_KIND_ORDER = {"code": 0, "config": 1, "shell": 2, "docs": 3, "other": 4}


def split_of(name: str) -> str:
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    return "test" if h >= TEST_BUCKET else "train"


def kind_of(files: list) -> str:
    """Majority extension kind of the changed files; docs-only -> docs."""
    votes = Counter(_KIND_EXT.get(Path(p).suffix.lstrip("."), "other") for p in files)
    return max(votes, key=lambda k: (votes[k], -_KIND_ORDER[k]))


def _diff_lines(repo, bad, good):
    """Total added + deleted lines between bad and good (numstat)."""
    out = subprocess.run(["git", "-C", str(repo), "diff", "--numstat", bad, good],
                         capture_output=True, text=True, check=True).stdout
    total = 0
    for line in out.splitlines():
        a, d, _p = line.split("\t")
        total += int(a) if a != "-" else 0
        total += int(d) if d != "-" else 0
    return total


def _candidates():
    cands = {}
    for line in (ROOT / "data" / "loopgym" / "candidates.jsonl").read_text().splitlines():
        if line:
            c = json.loads(line)
            cands[c["id"]] = c
    return cands


def task_records():
    """[{record dict}] for every verified, representable task, sorted by id."""
    cands = _candidates()
    recs = []
    for t in load_tasks():
        cand = cands[t.name]
        recs.append({
            "id": t.name, "repo": t.repo, "bad": t.bad, "good": t.good,
            "split": split_of(t.name), "kind": kind_of(t.files),
            "fix_msg": cand.get("fix_msg", ""),
            "files": list(t.files),
            "diff_lines": _diff_lines(V.REPOS[t.repo], t.bad, t.good),
        })
    return recs


def summary(recs) -> dict:
    by_repo = dict(sorted(Counter(r["repo"] for r in recs).items()))
    by_kind = dict(sorted(Counter(r["kind"] for r in recs).items()))
    splits = dict(sorted(Counter(r["split"] for r in recs).items()))
    n = len(recs)
    sizes = sorted(r["diff_lines"] for r in recs)
    mid = n // 2
    median = sizes[mid] if n % 2 else (sizes[mid - 1] + sizes[mid]) / 2
    return {
        "version": "v0",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": n,
        "by_repo": by_repo,
        "by_kind": by_kind,
        "splits": splits,
        "test_frac": round(splits.get("test", 0) / n, 4) if n else 0.0,
        "mean_diff_lines": round(sum(sizes) / n, 1) if n else 0.0,
        "median_diff_lines": float(median) if n else 0.0,
        "kill_criterion": {"min_tasks": MIN_TASKS, "met": n >= MIN_TASKS,
                           "eval_only": n < MIN_TASKS},
    }


def dataset_md(recs, s: str) -> str:
    verdict = ("met" if s["kill_criterion"]["met"] else "NOT met") + (
        " -> loopgym is eval-only: training uses Exercism."
        if s["kill_criterion"]["eval_only"] else
        " -> loopgym is used for both training and evaluation.")
    lines = [
        "# loopgym v0",
        "",
        f"Frozen {s['generated']} by `python -m loopgym.data.freeze`; "
        f"verify with `python -m loopgym.data.freeze --check`.",
        "",
        "## Source",
        "Defects this machine's own agents shipped in the **public** repos "
        "draftlab and harness (bare clones at data/raw/repos/). Each task is a "
        "(bad, good) commit pair: the bad commit shipped the defect, the good "
        "commit fixed it; the prompt is the original ticket plus the changed "
        "files at the bad commit. Mined by loopgym/data/mine.py.",
        "**Redaction:** sources are draftlab and harness only. No task, file, "
        "or ticket in this version comes from apply or contrib-loop.",
        "",
        "## Verification rule",
        "A task is frozen only if its hidden check (tests/hidden/<id>/) was run "
        "3x at the bad commit and 3x at the good commit in pristine worktrees "
        "and scored 0/3 pass at bad and 3/3 pass at good (loopgym/data/verify.py). "
        "Frozen copies live in data/loopgym/v0/checks/<id>/.",
        "",
        "## Contents",
        f"- {s['n_tasks']} tasks: {s['by_repo']}; by kind {s['by_kind']} "
        "(kind = majority language of changed files: code/config/shell/docs/other).",
        f"- Splits by h = int(sha1(id),16) % 100, test iff h >= {TEST_BUCKET}: "
        f"train {s['splits'].get('train', 0)} / test {s['splits'].get('test', 0)} "
        f"(test {s['test_frac']:.0%}).",
        f"- Diff size (added + deleted lines, bad -> good): mean "
        f"{s['mean_diff_lines']} / median {s['median_diff_lines']:.0f} lines "
        "(the mean is skewed by a few large harness/deploy fixes).",
        f"- Kill criterion (< {MIN_TASKS} verified tasks -> eval-only): {verdict}",
        "",
        "## Regeneration",
        "`python -m loopgym.data.freeze` rewrites these artifacts from the "
        "current verified state; `--check` fails (exit 1) if the frozen files "
        "drift from it.",
        "",
    ]
    return "\n".join(lines)


def build():
    recs = task_records()
    s = summary(recs)
    V0.mkdir(parents=True, exist_ok=True)
    for old in V0.glob("checks/*"):
        if old.is_dir():
            shutil.rmtree(old)
    (V0 / "checks").mkdir(exist_ok=True)
    for r in recs:
        d = V0 / "checks" / r["id"]
        d.mkdir()
        for f in V.check_files(V.HIDDEN / r["id"]):
            shutil.copy2(f, d / f.name)
    (V0 / "tasks.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    DATASET_MD.write_text(dataset_md(recs, s))
    print(f"freeze v0: {s['n_tasks']} tasks {s['by_repo']} "
          f"train {s['splits'].get('train', 0)}/test {s['splits'].get('test', 0)} "
          f"mean diff {s['mean_diff_lines']} lines")
    return s


def check() -> int:
    """Re-derive and compare. Returns 0 iff the frozen v0 matches current state."""
    recs = task_records()
    s = summary(recs)
    errs = []
    if not (V0 / "tasks.jsonl").exists():
        errs.append("data/loopgym/v0/tasks.jsonl missing (run build)")
        frozen = []
    else:
        frozen = [json.loads(line) for line in
                  (V0 / "tasks.jsonl").read_text().splitlines() if line.strip()]
    if len(frozen) != len(recs):
        errs.append(f"tasks.jsonl: {len(frozen)} records, expected {len(recs)}")
    by_id = {r["id"]: r for r in frozen}
    for r in recs:
        f = by_id.get(r["id"])
        if f is None:
            errs.append(f"missing from tasks.jsonl: {r['id']}")
            continue
        for k in ("repo", "bad", "good", "split", "kind", "fix_msg",
                  "files", "diff_lines"):
            if f.get(k) != r[k]:
                errs.append(f"{r['id']}.{k}: frozen {f.get(k)!r} != "
                            f"derived {r[k]!r}")
    for fid in by_id:
        if fid not in {r["id"] for r in recs}:
            errs.append(f"stale record in tasks.jsonl: {fid}")
    # split rule + kill criterion
    if s["n_tasks"] and s["test_frac"] < 0.30:
        errs.append(f"test split {s['test_frac']:.2%} < 30%")
    if not s["kill_criterion"]["met"] and "eval-only" not in DATASET_MD.read_text():
        errs.append("kill criterion not met but DATASET.md lacks eval-only note")
    # checks/ mirrors tests/hidden/
    for r in recs:
        want = {f.name: f.read_bytes() for f in V.check_files(V.HIDDEN / r["id"])}
        got_dir = V0 / "checks" / r["id"]
        got = {f.name: f.read_bytes() for f in
               (got_dir.glob("test_*.py") if got_dir.is_dir() else [])}
        if got != want:
            errs.append(f"checks/{r['id']}: frozen files {sorted(got)} != "
                        f"hidden {sorted(want)} (or content differs)")
    for d in (V0 / "checks").iterdir() if (V0 / "checks").is_dir() else []:
        if d.is_dir() and d.name not in {r["id"] for r in recs}:
            errs.append(f"stale checks dir: {d.name}")
    # results json
    if not RESULTS.exists():
        errs.append("results/loopgym_v0.json missing")
    else:
        doc = json.loads(RESULTS.read_text())
        for k in ("version", "n_tasks", "by_repo", "by_kind", "splits",
                  "test_frac", "mean_diff_lines", "median_diff_lines",
                  "kill_criterion"):
            if doc.get(k) != s[k]:
                errs.append(f"results/loopgym_v0.json.{k}: {doc.get(k)!r} != "
                            f"derived {s[k]!r}")
    # DATASET.md content
    if not DATASET_MD.exists():
        errs.append("DATASET.md missing")
    else:
        md = DATASET_MD.read_text()
        for needle in ("draftlab", "harness", "apply", "contrib-loop",
                       "0/3", "3/3", str(MIN_TASKS)):
            if needle not in md:
                errs.append(f"DATASET.md missing statement: {needle!r}")
    if errs:
        print(f"freeze --check: {len(errs)} mismatch(es):")
        for e in errs[:25]:
            print(f"  - {e}")
        return 1
    print(f"freeze --check: OK ({s['n_tasks']} tasks, test {s['test_frac']:.0%}, "
          f"kill criterion {'met' if s['kill_criterion']['met'] else 'NOT met'})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
