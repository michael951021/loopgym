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
           [--only baselines,benches,toy,grpo05,grpo15]
           [--results-dir DIR] [--runs-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))  # for loopgym.eval.stats in section_grpo05

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


def section_baselines(results: Path = RESULTS, runs: Path | None = None) -> list[dict]:
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


def section_benches(results: Path = RESULTS, runs: Path | None = None) -> list[dict]:
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


def section_toy(results: Path = RESULTS, runs: Path | None = None) -> list[dict]:
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


# grpo05 pre-run plausible ranges (state/tickets/grpo05ok.md).
GRPO05_RUN_NAME = "grpo05_exercism"
GRPO05_HACK_N = 20


def _slope_ci(steps, values, n_boot: int = 2000, seed: int = 0):
    import numpy as np

    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    slope = float(np.polyfit(steps, values, 1)[0])
    rng = np.random.default_rng(seed)
    n = steps.size
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # case resample: keep (step, reward) pairs
        slopes.append(float(np.polyfit(steps[idx], values[idx], 1)[0]))
    slopes.sort()
    lo = slopes[int(0.025 * n_boot)]
    hi = slopes[int(0.975 * n_boot)]
    return slope, (lo, hi)


def section_grpo05(results: Path = RESULTS, runs: Path | None = None) -> list[dict]:
    """Analyse the 0.5B exercism GRPO run into a claim with CIs, or a null.

    Inputs: runs/grpo05_exercism/metrics.jsonl, the two val evals, and the
    optional 20-completion hack screen. Writes results/grpo05_summary.json.
    A null (paired CI containing 0) is a passing result; a gain whose CI
    excludes 0 passes only with a clean hack screen; a significant val
    regression or out-of-range trend fails."""
    return _grpo_section("grpo05", "base05", GRPO05_RUN_NAME, "configs/grpo05_exercism.toml",
                         "40-step LoRA GRPO, exercism, 0.5B", results, runs)


def section_grpo15(results: Path = RESULTS, runs: Path | None = None) -> list[dict]:
    """The 1.5B GPU run (scripts/gpu_run.sh), same analysis as grpo05: paired val
    pass@1 against base15 at the baseline protocol (evalgrpo15: n=16, T=0.8)."""
    return _grpo_section("grpo15", "base15", "grpo15_gpu", "configs/grpo15_gpu.toml",
                         "100-step LoRA GRPO, exercism, 1.5B, 2x3090", results, runs)


def _grpo_section(tag: str, base_tag: str, run_name: str, config: str, desc: str,
                  results: Path, runs: Path | None) -> list[dict]:
    runs = runs or ROOT / "runs"
    metrics = runs / run_name / "metrics.jsonl"
    base = load_json(results / "eval" / f"{base_tag}_val.json")
    grpo = load_json(results / "eval" / f"{tag}_val.json")
    if not metrics.is_file() or base is None or grpo is None:
        absent = [name for name, ok in (
            (f"runs/{run_name}/metrics.jsonl", metrics.is_file()),
            (f"results/eval/{base_tag}_val.json", base is not None),
            (f"results/eval/{tag}_val.json", grpo is not None)) if not ok]
        return [row(f"{tag} run + val evals", "missing", "metrics.jsonl + "
                    f"{base_tag}_val.json + {tag}_val.json", None,
                    "run/evals not complete yet: " + ", ".join(absent))]

    import numpy as np
    from loopgym.eval.stats import bootstrap_ci

    rows_all = [json.loads(line) for line in metrics.read_text().splitlines()
                if line.strip()]
    tr = [r for r in rows_all if "kind" not in r]
    ev = [r for r in rows_all if r.get("kind") == "eval"]
    steps = [r["step"] for r in tr]
    rm = [r["reward_mean"] for r in tr]
    kl = [r["kl"] for r in tr]
    cf = [r["clip_frac"] for r in tr]
    fs = [r["frac_groups_all_same"] for r in tr]
    n = len(tr)

    slope, (slo, shi) = _slope_ci(steps, rm)

    tb = sorted(base["per_task"])
    tg = sorted(grpo["per_task"])
    tasks_ok = tb == tg
    d = np.array([grpo["per_task"][t]["pass_at_1"]
                  - base["per_task"][t]["pass_at_1"] for t in tb]) \
        if tasks_ok else np.array([0.0])
    delta, dlo, dhi = (bootstrap_ci(d, groups=tb, n_boot=2000, seed=0)
                       if tasks_ok else (0.0, 0.0, 0.0))
    verdict = ("regression" if dhi < 0 else "gain" if dlo > 0 else "null")

    hack = load_json(results / f"{tag}_hackcheck.json")
    hack_clean = (hack is not None and hack.get("n") == GRPO05_HACK_N
                  and hack.get("n_matches") == GRPO05_HACK_N)

    rows: list[dict] = []
    if not tasks_ok:
        rows.append(row("val task sets match",
                        f"{sorted(set(tg) - set(tb))} vs {sorted(set(tb) - set(tg))}",
                        "identical task sets", False,
                        "paired comparison impossible"))
    if tasks_ok:
        if verdict == "null":
            ok, note = True, "null result: CI contains 0 (valid)"
        elif verdict == "regression":
            ok, note = False, "significant val regression — investigate"
        else:
            ok = hack_clean
            note = ("gain; hack screen clean" if hack_clean else
                    f"gain with CI excluding 0 needs a clean "
                    f"{GRPO05_HACK_N}-completion hack screen "
                    f"(results/{tag}_hackcheck.json: 20/20 sidecar match)")
        rows.append(row(
            f"{tag} vs {base_tag} val pass@1 (paired)",
            f"{grpo['pass_at_k']:.4f} vs {base['pass_at_k']:.4f} "
            f"(Δ {delta:+.4f} [{dlo:+.4f}, {dhi:+.4f}])",
            "CI contains 0 (null) or gain w/ clean hack screen", ok, note))
    rows.append(row(
        "train reward slope",
        f"{slope:+.5f}/step [{slo:+.5f}, {shi:+.5f}]",
        "in [-0.05, +0.05]/step (pre-run range)",
        -0.05 <= slope <= 0.05,
        "significant decline in train reward" if shi < 0 else ""))
    kl_last10 = float(np.mean(kl[-10:])) if n >= 10 else float(np.mean(kl))
    rows.append(row(
        "KL trend (first10 -> last10)",
        f"{float(np.mean(kl[:10])):.2e} -> {kl_last10:.2e}",
        "last-10 mean < 0.5 nats (pre-run range)", kl_last10 < 0.5))
    rows.append(row(
        "clip_frac range", f"{min(cf):.3f}–{max(cf):.3f}", "in [0, 1]",
        min(cf) >= 0.0 and max(cf) <= 1.0))
    rows.append(row(
        "frac_groups_all_same", f"{float(np.mean(fs)):.3f} mean of {n} steps",
        "reported", True,
        f"{100 * float(np.mean(fs)):.1f}% of groups all-same reward: "
        f"zero advantage signal"))
    if ev:
        rows.append(row(
            "in-run eval points",
            "; ".join(f"step {e['step']}: {e['reward_mean']:.3f} "
                      f"({','.join(e['tasks'])})" for e in ev),
            "reported", True))
    if hack is None:
        rows.append(row("hack screen (20 completions)", "missing",
                        f"{GRPO05_HACK_N}/{GRPO05_HACK_N} sidecar match",
                        None, f"run tools/grpo05_hackcheck.py for {tag}"))
    else:
        flagged = sorted({f for c in hack.get("completions", [])
                          for f in c.get("flags", [])})
        rows.append(row(
            "hack screen (20 completions)",
            f"{hack.get('n_matches', 0)}/{hack.get('n', 0)} sidecar match, "
            f"{hack.get('n_flagged', 0)} flagged",
            f"{GRPO05_HACK_N}/{GRPO05_HACK_N} (determinism known answer)",
            hack_clean,
            f"flags for review: {flagged}" if flagged else "no mechanical flags"))

    summary = {
        "run": f"runs/{run_name}",
        "config": config,
        "steps": n,
        "train_reward": {
            "first5_mean": float(np.mean(rm[:5])),
            "last5_mean": float(np.mean(rm[-5:])),
            "slope_per_step": slope,
            "slope_ci": [slo, shi],
        },
        "val": {
            base_tag: {"pass_at_k": base["pass_at_k"], "ci": list(base["ci"]),
                       "n_per_task": base["n"]},
            tag: {"pass_at_k": grpo["pass_at_k"], "ci": list(grpo["ci"]),
                       "n_per_task": grpo["n"]},
            "paired_delta": {"point": delta, "ci": [dlo, dhi],
                             "n_tasks": len(tb),
                             "method": "task-clustered bootstrap, n_boot=2000, "
                                       "seed=0 (loopgym.eval.stats)"},
            "verdict": verdict,
        },
        "kl": {"max": max(kl), "first10_mean": float(np.mean(kl[:10])),
               "last10_mean": kl_last10},
        "clip_frac": {"min": min(cf), "max": max(cf)},
        "frac_groups_all_same": {"mean": float(np.mean(fs)), "max": max(fs)},
        "eval_points": ev,
        "hack_screen": None if hack is None else
            {"n": hack.get("n"), "n_matches": hack.get("n_matches"),
             "n_flagged": hack.get("n_flagged")},
    }
    claim = (f"{tag} ({desc}): {verdict} on val "
             f"pass@1 (paired Δ {delta:+.4f}, CI [{dlo:+.4f}, {dhi:+.4f}])")
    if verdict == "null":
        claim += ("; train reward declined "
                  f"({slope:+.4f}/step, CI [{slo:+.4f}, {shi:+.4f}]) while "
                  f"{100 * float(np.mean(fs)):.0f}% of groups carried zero "
                  "advantage signal — no detectable learning at this scale")
    summary["claim"] = claim
    (results / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    return rows


SECTIONS = {
    "baselines": section_baselines,
    "benches": section_benches,
    "toy": section_toy,
    "grpo05": section_grpo05,
    "grpo15": section_grpo15,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only",
                    help=f"comma-separated subset of {sorted(SECTIONS)}")
    ap.add_argument("--results-dir", default=str(RESULTS),
                    help="results directory (default: <repo>/results)")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"),
                    help="runs directory (default: <repo>/runs)")
    args = ap.parse_args(argv)
    results = Path(args.results_dir)
    runs = Path(args.runs_dir)
    which = (args.only.split(",") if args.only else list(SECTIONS))
    for w in which:
        if w not in SECTIONS:
            raise SystemExit(f"unknown section {w!r}; have {sorted(SECTIONS)}")
    failed = 0
    for w in which:
        print(f"== {w} ==")
        for r in SECTIONS[w](results, runs):
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
