"""Exercise audit (local runner) and train/val/test splits by sha1(name).

usable = reference scores 1.0 AND stub scores 0.0 on the local runner.
Splits: h = int(sha1(name).hexdigest(), 16) % 100 -> train h<70, val h<85, test h>=85.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .exercism import grade, load_exercise

ROOT = Path(__file__).resolve().parents[2]


def bucket(name):
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def split_names(names):
    out = {"train": [], "val": [], "test": []}
    for n in sorted(names):
        out[bucket(n)].append(n)
    return out


def _grade_pair(exdir, timeout=30):
    t = load_exercise(exdir)
    ref, ref_info = grade(t, t.reference(), "local", timeout=timeout)
    stub, stub_info = grade(t, t.stub(), "local", timeout=timeout)
    return t.name, ref, stub, ref_info, stub_info


def audit(exdir, workers=4, timeout=30):
    """Grade reference and stub of every exercise. Returns {name: record}."""
    exdir = Path(exdir)
    dirs = sorted(p for p in exdir.iterdir() if p.is_dir())

    def one(d):
        try:
            name, ref, stub, ref_info, stub_info = _grade_pair(d, timeout=timeout)
            reasons = []
            if ref != 1.0:
                reasons.append("ref=0.0" + (" (timeout)" if ref_info["timed_out"] else ""))
            if stub != 0.0:
                reasons.append("stub=1.0")
            return {
                "ref": ref, "stub": stub,
                "usable": ref == 1.0 and stub == 0.0,
                "reason": "; ".join(reasons) if reasons else "",
                "ref_tests": ref_info["tests"], "stub_tests": stub_info["tests"],
            }
        except Exception as e:  # missing/malformed exercise files
            return {"ref": None, "stub": None, "usable": False,
                    "reason": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = dict(zip((d.name for d in dirs), ex.map(one, dirs)))
    return records


def main():
    exdir = Path(os.environ.get("LOOPGYM_EXERCISES",
                                ROOT / "data" / "raw" / "exercism-python"
                                / "exercises" / "practice"))
    cfg = json.loads((ROOT / "data" / "raw" / "exercism-python" / "config.json").read_text())
    difficulty = {e["slug"]: e["difficulty"] for e in cfg["exercises"]["practice"]}

    records = audit(exdir, workers=4)
    usable = [n for n, r in records.items() if r["usable"]]
    unusable = {n: r for n, r in records.items() if not r["usable"]}

    splits_dir = ROOT / "results"
    splits_dir.mkdir(exist_ok=True)
    audit_doc = {
        "exercises": len(records),
        "usable": len(usable),
        "unusable": {n: {"reason": r["reason"], "ref": r["ref"], "stub": r["stub"]}
                     for n, r in sorted(unusable.items())},
    }
    (splits_dir / "env_audit.json").write_text(json.dumps(audit_doc, indent=1) + "\n")
    split_doc = {b: {n: difficulty[n] for n in names}
                 for b, names in split_names(usable).items()}
    (splits_dir / "splits.json").write_text(json.dumps(split_doc, indent=1) + "\n")

    sizes = {b: len(v) for b, v in split_doc.items()}
    print(f"exercises: {len(records)}")
    print(f"usable: {len(usable)}")
    for n, r in sorted(unusable.items()):
        print(f"  unusable: {n} ({r['reason']})")
    print(f"splits train/val/test: {sizes['train']}/{sizes['val']}/{sizes['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
