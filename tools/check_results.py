"""Check committed result numbers against the CLAUDE.md plausible ranges.

Loads results/eval/*.json, results/toy.json and the bench files; every section
grows with its experiment. A PASS row: number inside its written range (or
known answer holds). A PENDING row: section's result file not produced yet —
reported, not a failure. Exit 0 iff every checked row in the selected sections
passes. Ranges are never widened by this tool.

An out-of-range number can only pass via a range-author ruling committed at
results/rulings/<stem>.ruling.json ({"author", "decision": "accept"|"revise",
"rationale", "new_range" for revise}). Agents never write rulings (CLAUDE.md
rule 5); the tool applies them verbatim and cites the ruling in the table row.
Malformed rulings fail closed.

Usage: .venv/bin/python tools/check_results.py
           [--only baselines,benches,toy] [--results-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# CLAUDE.md "plausible ranges written before the runs" (point estimates).
BASELINE_RANGES = {
    "base05_val.json": ("0.5B val pass@1", 0.10, 0.55),
    "base15_val.json": ("1.5B val pass@1", 0.25, 0.75),
}
REFERENCE_TOL = 1e-9


def row(check: str, value: str, expected: str, ok: bool | None,
        note: str = "") -> dict:
    return {"check": check, "value": value, "expected": expected,
            "ok": ok, "note": note}


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_ruling(results: Path, fname: str) -> tuple[Path, object, str | None]:
    """Ruling file for a result file: <results>/rulings/<stem>.ruling.json.

    Returns (path, parsed-or-None, error-or-None); (path, None, None) when no
    ruling is on file."""
    path = results / "rulings" / (fname[:-5] + ".ruling.json")
    if not path.is_file():
        return path, None, None
    try:
        return path, json.loads(path.read_text()), None
    except (OSError, ValueError) as e:
        return path, None, f"ruling unreadable ({type(e).__name__})"


def check_ruling(r: object, p: float) -> tuple[bool, str, str]:
    """Apply a range-author ruling to an out-of-range number.

    Returns (ok, note, expected). Fail-closed: anything malformed keeps the
    row failed. Rulings are written by the range author only (rule 5); the
    tool applies them verbatim."""
    if not isinstance(r, dict):
        return False, "ruling malformed: not a JSON object", ""
    author = r.get("author")
    rationale = r.get("rationale")
    if not isinstance(author, str) or not author.strip():
        return False, "ruling malformed: 'author' missing or empty", ""
    if not isinstance(rationale, str) or not rationale.strip():
        return False, "ruling malformed: 'rationale' missing or empty", ""
    decision = r.get("decision")
    if decision == "accept":
        return True, f"in range by ruling — {author}: {rationale}", ""
    if decision == "revise":
        nr = r.get("new_range")
        if (not isinstance(nr, (list, tuple)) or len(nr) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           for x in nr)
                or nr[0] >= nr[1]):
            return False, ("ruling malformed: 'revise' needs new_range "
                           "[lo, hi] with lo < hi"), ""
        lo, hi = nr[0], nr[1]
        ok = lo <= p <= hi
        note = f"range ruled to [{lo}, {hi}] by {author}: {rationale}"
        if not ok:
            note += " — number outside ruled range"
        return ok, note, f"[{lo}, {hi}]"
    return False, ("ruling malformed: decision must be 'accept' or "
                   f"'revise', got {decision!r}"), ""


def section_baselines(results: Path = RESULTS) -> list[dict]:
    rows: list[dict] = []
    docs = {}
    for fname, (label, lo, hi) in BASELINE_RANGES.items():
        d = load_json(results / "eval" / fname)
        if d is None:
            rows.append(row(label, "missing", f"[{lo}, {hi}]", False,
                            "baseline file missing"))
            continue
        p = d["pass_at_k"]
        docs[fname] = d
        ok = lo <= p <= hi
        expected, note = f"[{lo}, {hi}]", ""
        if not ok:
            path, r, err = load_ruling(results, fname)
            if r is None and err is not None:
                note = err
            elif r is None:
                note = (f"out of pre-run range; fix the cause or await the "
                        f"range-author ruling (CLAUDE.md rule 5) at "
                        f"{path.relative_to(results)}")
            else:
                ok, note, exp2 = check_ruling(r, p)
                if exp2:
                    expected = f"[{lo}, {hi}] -> ruled {exp2}"
        rows.append(row(label, f"{p:.4f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]",
                        expected, ok, note))
    if len(docs) == 2:
        p05, p15 = docs["base05_val.json"]["pass_at_k"], docs["base15_val.json"]["pass_at_k"]
        rows.append(row("1.5B > 0.5B", f"{p15:.4f} vs {p05:.4f}",
                        "p15 > p05", p15 > p05))
        c05, c15 = docs["base05_val.json"]["ci"], docs["base15_val.json"]["ci"]
        disjoint = c15[0] > c05[1]
        rows.append(row("CIs 1.5B vs 0.5B",
                        f"{c05} vs {c15}",
                        "non-overlapping or overlap reported",
                        True,
                        "non-overlapping" if disjoint
                        else "overlap reported: CIs intersect"))
    ref = load_json(results / "eval" / "reference_exercism_val.json")
    if ref is None:
        rows.append(row("reference val pass@1", "missing", "1.0", False,
                        "reference file missing"))
    else:
        p = ref["pass_at_k"]
        rows.append(row("reference val pass@1", f"{p:.4f}", "1.0",
                        abs(p - 1.0) <= REFERENCE_TOL))
    return rows


def section_benches(results: Path = RESULTS) -> list[dict]:
    rows: list[dict] = []
    b = load_json(results / "bench_logprob.json")
    if b is None:
        rows.append(row("logprob bench", "missing", "file", False,
                        "bench not run yet"))
    else:
        cells = {(c["impl"], c["T"]): c for c in b["cells"]}
        ts = b["T"]
        for T in ts:
            naive = cells.get(("naive", T))
            if naive is None:
                continue
            for impl in ("torch", "cpp", "triton", "cuda"):
                c = cells.get((impl, T))
                if c is None:
                    continue
                rows.append(row(
                    f"logprob {impl} T={T} peak < 1/8 naive peak",
                    f"{c['peak_extra_bytes'] / 2**20:.1f} MiB",
                    f"< {naive['peak_extra_bytes'] / 8 / 2**20:.1f} MiB",
                    c["peak_extra_bytes"] < naive["peak_extra_bytes"] / 8))
    lb = load_json(results / "lgbox_bench.json")
    if lb is None:
        rows.append(row("lgbox bench", "missing", "file", False,
                        "bench not run yet"))
    else:
        c1 = {c["runner"]: c for c in lb["cells"] if c["concurrency"] == 1}
        if "local" in c1 and "http" in c1:
            overhead = c1["http"]["lat_mean_s"] - c1["local"]["lat_mean_s"]
            rows.append(row(
                "lgbox overhead per grade @ c=1",
                f"{overhead * 1000:.1f} ms", "< 50 ms",
                overhead < 0.05))
        else:
            rows.append(row("lgbox overhead per grade @ c=1",
                            "missing cells", "< 50 ms", False,
                            "local/http concurrency-1 cells absent"))
    return rows


def section_toy(results: Path = RESULTS) -> list[dict]:
    """results/toy.json: written by the toy known-answer runs (banana + null).

    Expected keys: banana_early, banana_late, null_slope_ci (lo, hi)."""
    d = load_json(results / "toy.json")
    if d is None:
        return [row("toy known answers", "missing", "file", None,
                    "toy runs not done yet (runs/toy_*)")]
    rows = []
    rows.append(row("banana early < 0.2", f"{d['banana_early']:.3f}", "< 0.2",
                    d["banana_early"] < 0.2))
    rows.append(row("banana late > 0.8", f"{d['banana_late']:.3f}", "> 0.8",
                    d["banana_late"] > 0.8))
    lo, hi = d["null_slope_ci"]
    rows.append(row("null slope CI contains 0",
                    f"[{lo:.5f}, {hi:.5f}]", "0 in CI", lo <= 0 <= hi))
    return rows


SECTIONS = {
    "baselines": section_baselines,
    "benches": section_benches,
    "toy": section_toy,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only",
                    help=f"comma-separated subset of {sorted(SECTIONS)}")
    ap.add_argument("--results-dir", default=str(RESULTS),
                    help="results directory (default: <repo>/results)")
    args = ap.parse_args(argv)
    results = Path(args.results_dir)
    which = (args.only.split(",") if args.only else list(SECTIONS))
    for w in which:
        if w not in SECTIONS:
            raise SystemExit(f"unknown section {w!r}; have {sorted(SECTIONS)}")
    failed = 0
    for w in which:
        print(f"== {w} ==")
        for r in SECTIONS[w](results):
            status = "PASS" if r["ok"] else ("PENDING" if r["ok"] is None
                                             else "FAIL")
            if r["ok"] is False:
                failed += 1
            note = f"  ({r['note']})" if r["note"] else ""
            print(f"  {status:7s} {r['check']:<44s} {r['value']:>28s} "
                  f"want {r['expected']}{note}")
    print(f"{'FAIL' if failed else 'OK'}: {failed} failed row(s) "
          f"in {', '.join(which)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
