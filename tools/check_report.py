"""Recompute every number REPORT.md states, from results/ and the GRPO run log.

Contract: each measured claim is computed from the committed result files and
rendered with a fixed format; the rendered string must appear in REPORT.md,
else the report is stale or invented -> exit 1. Range claims (CLAUDE.md
plausible ranges) are re-checked too; an out-of-range number passes only via a
committed range-author ruling (same fail-closed rule as check_results.py).
Structure: the sections the ticket requires must exist; the GPU section may
state results only when results/gpu/ has scale files, otherwise it must say
PENDING.

`--emit` prints the canonical strings (for authoring the report).

Usage: .venv/bin/python tools/check_report.py [--report PATH]
                                             [--results-dir DIR] [--emit]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_results import check_ruling, load_ruling  # noqa: E402  (same repo)

RANGE_05 = (0.10, 0.55)          # CLAUDE.md pre-run ranges
RANGE_15 = (0.25, 0.75)
RANGE_LG = (0.0, 0.25)
RANGE_STEP_MIN = (5.0, 25.0)

REQUIRED_SECTIONS = ["Question", "Task sources", "Method", "Results",
                     "Failed", "GPU", "Corrections"]


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def row(section: str, check: str, want: str, in_report: bool,
        ok: bool | None, note: str = "") -> dict:
    return {"section": section, "check": check, "want": want,
            "in_report": in_report, "ok": ok, "note": note}


def ci_str(ci: list[float], n: int = 3) -> str:
    return "[" + ", ".join(f"{x:.{n}f}" for x in ci) + "]"


def section_sources(rep: str) -> list[dict]:
    rows: list[dict] = []
    v0 = load_json(RESULTS / "loopgym_v0.json")
    mine = load_json(RESULTS / "mining.json")
    audit = load_json(RESULTS / "env_audit.json")
    splits = load_json(RESULTS / "splits.json")
    if v0 is None or mine is None or audit is None or splits is None:
        return [row("sources", "result files present", "loopgym_v0/mining/"
                "env_audit/splits .json", False, False, "missing results file")]
    checks = [
        ("exercism exercises", str(audit["exercises"])),
        ("exercism usable", str(audit["usable"])),
        ("exercism unusable", str(len(audit["unusable"]))),
        ("splits train/val/test",
         f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}"),
        ("v0 tasks", f"{v0['n_tasks']}"),
        ("v0 kinds code/config/docs",
         f"{v0['by_kind']['code']}/{v0['by_kind']['config']}/{v0['by_kind']['docs']}"),
        ("v0 repos draftlab/harness",
         f"{v0['by_repo']['draftlab']}/{v0['by_repo']['harness']}"),
        ("v0 split train/test",
         f"{v0['splits']['train']}/{v0['splits']['test']}"),
        ("v0 test_frac", f"{v0['test_frac']:.2f}"),
        ("v0 kill criterion", "met" if v0["kill_criterion"]["met"] else "not met"),
        ("mining candidates", str(mine["candidates"])),
        ("mining seeds recalled", mine["recalled"]),
    ]
    kc = v0["kill_criterion"]
    for label, want in checks:
        ok, note = None, ""
        if label == "v0 kill criterion":
            ok = kc["met"] == (v0["n_tasks"] >= kc["min_tasks"])
            note = (f"n_tasks {v0['n_tasks']} >= min {kc['min_tasks']}"
                    if kc["met"] else f"n_tasks {v0['n_tasks']} < {kc['min_tasks']}")
        rows.append(row("sources", label, want, want in rep, ok, note))
    return rows


def section_baselines(rep: str) -> list[dict]:
    rows: list[dict] = []
    b05 = load_json(RESULTS / "eval" / "base05_val.json")
    b15 = load_json(RESULTS / "eval" / "base15_val.json")
    ref = load_json(RESULTS / "eval" / "reference_exercism_val.json")
    l05 = load_json(RESULTS / "eval" / "base05_loopgym.json")
    l15 = load_json(RESULTS / "eval" / "base15_loopgym.json")
    if None in (b05, b15, ref, l05, l15):
        return [row("baselines", "eval files present",
                    "results/eval/*.json", False, False, "missing results file")]
    p05, p15 = b05["pass_at_k"], b15["pass_at_k"]
    p05s, p15s = f"{p05:.4f}", f"{p15:.4f}"
    in_range = RANGE_05[0] <= p05 <= RANGE_05[1]
    note05, ok05 = "", in_range
    if not in_range:
        _, ruling, err = load_ruling(RESULTS, "base05_val.json")
        ok05, note05, _ = check_ruling(ruling, p05) if err is None else (False, err, "")
        if ok05:
            note05 += "; report must cite the ruling"
            rows.append(row("baselines", "ruling cited in report", "ruling",
                            "ruling" in rep, ok05))
    rows.append(row("baselines", "0.5B val pass@1",
                    f"{p05s} {ci_str(b05['ci'])}", p05s in rep and ci_str(b05["ci"]) in rep,
                    ok05, note05 or f"range {RANGE_05}"))
    rows.append(row("baselines", "1.5B val pass@1",
                    f"{p15s} {ci_str(b15['ci'])}", p15s in rep and ci_str(b15["ci"]) in rep,
                    RANGE_15[0] <= p15 <= RANGE_15[1], f"range {RANGE_15}"))
    rows.append(row("baselines", "1.5B > 0.5B", f"{p15s} > {p05s}",
                    p15s in rep and p05s in rep, p15 > p05))
    rows.append(row("baselines", "reference val pass@1", f"{ref['pass_at_k']:.4f}",
                    f"{ref['pass_at_k']:.4f}" in rep, ref["pass_at_k"] == 1.0,
                    "gold solutions score 1 (known answer)"))
    for name, doc in (("0.5B loopgym-test", l05), ("1.5B loopgym-test", l15)):
        p, s = doc["pass_at_k"], f"{doc['pass_at_k']:.4f}"
        rows.append(row("baselines", f"{name} pass@1 ({doc['n_tasks']} tasks)",
                        f"{s} {ci_str(doc['ci'])}", s in rep and ci_str(doc["ci"]) in rep,
                        RANGE_LG[0] <= p <= RANGE_LG[1], f"range {RANGE_LG}"))
    rows.append(row("baselines", "loopgym test n_tasks", str(l05["n_tasks"]),
                    str(l05["n_tasks"]) in rep, l05["n_tasks"] == l15["n_tasks"]))
    return rows


def section_toy(rep: str) -> list[dict]:
    t = load_json(RESULTS / "toy.json")
    if t is None:
        return [row("toy", "results/toy.json present", "toy.json", False, False)]
    checks = [
        ("banana early", f"{t['banana_early']:.4f}", t["banana_early"] < 0.2, "< 0.2"),
        ("banana late", f"{t['banana_late']:.4f}", t["banana_late"] > 0.8, "> 0.8"),
        ("banana slope", f"{t['banana_slope']:.4f}",
         not _contains_zero(t["banana_slope_ci"]), "CI excludes 0"),
        ("banana slope CI", ci_str(t["banana_slope_ci"], 4),
         not _contains_zero(t["banana_slope_ci"]), ""),
        ("null early", f"{t['null_early']:.4f}", None, ""),
        ("null late", f"{t['null_late']:.4f}", None, ""),
        ("null slope CI", ci_str(t["null_slope_ci"], 4),
         _contains_zero(t["null_slope_ci"]), "CI contains 0 (null control)"),
    ]
    return [row("toy", c, w, w in rep, ok, n) for c, w, ok, n in checks]


def _contains_zero(ci: list[float]) -> bool:
    return ci[0] <= 0.0 <= ci[1]


def section_grpo05(rep: str) -> list[dict]:
    rows: list[dict] = []
    s = load_json(RESULTS / "grpo05_summary.json")
    g = load_json(RESULTS / "eval" / "grpo05_val.json")
    b = load_json(RESULTS / "eval" / "base05_val.json")
    hack = load_json(RESULTS / "grpo05_hackcheck.json")
    cfg = tomllib.loads((ROOT / "configs" / "grpo05_exercism.toml").read_text())
    metrics = ROOT / "runs" / "grpo05_exercism" / "metrics.jsonl"
    if None in (s, g, b, hack) or not metrics.is_file():
        return [row("grpo05", "run + summary files present",
                    "grpo05_summary.json, evals, metrics.jsonl", False, False,
                    "missing results file")]
    pg = g["pass_at_k"]
    rows.append(row("grpo05", "grpo05 val pass@1",
                    f"{pg:.4f} {ci_str(g['ci'])}",
                    f"{pg:.4f}" in rep and ci_str(g["ci"]) in rep, None))
    d = s["val"]["paired_delta"]
    dstr = (f"Δ {d['point']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    lo, hi = d["ci"]
    if lo <= 0 <= hi:
        ok, note = True, "null result (CI contains 0)"
    else:
        ok = hack["n_flagged"] == 0
        note = "gain needs a clean hack screen"
    rows.append(row("grpo05", "paired val delta", dstr, dstr in rep, ok, note))
    ts = s["train_reward"]
    sstr = (f"{ts['slope_per_step']:.5f}/step "
            f"[{ts['slope_ci'][0]:.5f}, {ts['slope_ci'][1]:.5f}]")
    rows.append(row("grpo05", "train reward slope", sstr, sstr in rep,
                    -0.05 <= ts["slope_per_step"] <= 0.05, "pre-run range"))
    kl = s["kl"]
    klstr = f"{kl['first10_mean']:.2e} → {kl['last10_mean']:.2e}"
    rows.append(row("grpo05", "KL first10 -> last10", klstr, klstr in rep,
                    kl["last10_mean"] < 0.5, "last-10 mean < 0.5 nats"))
    rows.append(row("grpo05", "KL max", f"{kl['max']:.2e}", f"{kl['max']:.2e}" in rep,
                    kl["max"] < 0.5, ""))
    rows.append(row("grpo05", "clip_frac max", f"{s['clip_frac']['max']:.1f}",
                    f"{s['clip_frac']['max']:.1f}" in rep,
                    0.0 <= s["clip_frac"]["max"] <= 1.0, ""))
    rows.append(row("grpo05", "frac_groups_all_same",
                    f"{s['frac_groups_all_same']['mean']:.3f}",
                    f"{s['frac_groups_all_same']['mean']:.3f}" in rep, None,
                    "zero-advantage-group fraction"))
    hs = s["hack_screen"]
    rows.append(row("grpo05", "hack screen sidecar match",
                    f"{hs['n_matches']}/{hs['n']}",
                    f"{hs['n_matches']}/{hs['n']}" in rep,
                    hs["n_matches"] == hs["n"], "determinism known answer"))
    rows.append(row("grpo05", "hack screen flagged", str(hs["n_flagged"]),
                    str(hs["n_flagged"]) in rep, None, "references_test_names only"))
    steps_rows = [json.loads(ln) for ln in metrics.read_text().splitlines()
                  if ln.strip() and json.loads(ln).get("kind") != "eval"]
    step_min = [r["step_s"] / 60 for r in steps_rows]
    med = f"{statistics.median(step_min):.2f} min"
    rng = f"[{min(step_min):.2f}, {max(step_min):.2f}]"
    rows.append(row("grpo05", "CPU step time median", med, med in rep,
                    RANGE_STEP_MIN[0] <= statistics.median(step_min)
                    <= RANGE_STEP_MIN[1], f"range {RANGE_STEP_MIN} min"))
    rows.append(row("grpo05", "CPU step time range", rng, rng in rep, None))
    cfgstr = f"batch {cfg['batch']}, group {cfg['group']}, {cfg['steps']} steps"
    rows.append(row("grpo05", "config (batch/group/steps)", cfgstr,
                    cfgstr in rep, len(steps_rows) == cfg["steps"], ""))
    return rows


def section_lgbox(rep: str) -> list[dict]:
    doc = load_json(RESULTS / "lgbox_bench.json")
    if doc is None:
        return [row("lgbox", "results/lgbox_bench.json present", "lgbox_bench.json",
                    False, False)]
    rows: list[dict] = []
    cell = {(c["runner"], c["concurrency"]): c for c in doc["cells"]}
    for name, (runner, conc) in (("local c1", ("local", 1)), ("http c1", ("http", 1)),
                                 ("local c16", ("local", 16)), ("http c16", ("http", 16))):
        c = cell[(runner, conc)]
        w = f"{c['grades_per_s']:.2f}"
        rows.append(row("lgbox", f"{name} grades/s ({c['n']} grades)",
                        w, w in rep, c["n_reward_1"] == c["n"],
                        "ref solution scores 1 on every grade"))
    c1l, c1h = cell[("local", 1)], cell[("http", 1)]
    c16l, c16h = cell[("local", 16)], cell[("http", 16)]
    rows.append(row("lgbox", "latency p50 c1 local/http (s)",
                    f"{c1l['lat_p50_s']:.4f}/{c1h['lat_p50_s']:.4f}",
                    f"{c1l['lat_p50_s']:.4f}" in rep and f"{c1h['lat_p50_s']:.4f}" in rep,
                    None))
    rows.append(row("lgbox", "latency p95 c16 local/http (s)",
                    f"{c16l['lat_p95_s']:.3f}/{c16h['lat_p95_s']:.3f}",
                    f"{c16l['lat_p95_s']:.3f}" in rep and f"{c16h['lat_p95_s']:.3f}" in rep,
                    None))
    ov = doc["checks"]["http_overhead_c1_lt_50ms"]
    ms = f"{ov['value_s'] * 1000:.1f} ms"
    rows.append(row("lgbox", "http overhead per grade @ c1", ms, ms in rep,
                    ov["pass"], "known answer: < 50 ms"))
    return rows


def section_logprob(rep: str) -> list[dict]:
    doc = load_json(RESULTS / "bench_logprob.json")
    if doc is None:
        return [row("logprob", "results/bench_logprob.json present",
                    "bench_logprob.json", False, False)]
    T = doc["T"][1]
    cell = {(c["impl"], c["T"]): c for c in doc["cells"] if not c.get("triton_interpret")}
    rows: list[dict] = []
    for impl in ("naive", "torch", "cpp"):
        c = cell[(impl, T)]
        w = f"{c['time_median_s']:.4f}"
        rows.append(row("logprob", f"{impl} T={T} median (s)", w, w in rep, None))
    naive = cell[("naive", T)]["peak_extra_bytes"]
    rows.append(row("logprob", f"naive T={T} peak (B)", str(naive),
                    str(naive) in rep, None, "T x V x 4 = 1.24 GB; measured peak 2x"))
    for impl in ("torch", "cpp"):
        c = cell[(impl, T)]
        ratio = c["peak_extra_bytes"] / naive
        rows.append(row("logprob", f"{impl} T={T} peak (B)",
                        str(c["peak_extra_bytes"]),
                        str(c["peak_extra_bytes"]) in rep,
                        c["peak_extra_bytes"] < naive / 8,
                        f"= {ratio:.3f} of naive (< 1/8)" if impl == "torch"
                        else f"= {ratio:.5f} of naive"))
    md = doc["checks"]["all_impls_match_f64_reference_1e-4"]
    worst = max(md["max_diffs"].values())
    rows.append(row("logprob", "worst diff vs f64 reference",
                    f"{worst:.3e}", f"{worst:.3e}" in rep, md["pass"],
                    f"tolerance {md['tolerance']}"))
    return rows


def section_gpu(rep: str, results: Path) -> list[dict]:
    rows: list[dict] = []
    sec = _section_text(rep, "GPU")
    gpu = results / "gpu"
    files = sorted(gpu.glob("scale_*.json")) if gpu.is_dir() else []
    if not files:
        rows.append(row("gpu", "results/gpu/ absent -> PENDING marker",
                        "PENDING", "PENDING" in sec, "PENDING" in sec,
                        "plan only; results land with the host GPU run"))
        return rows
    for f in files:
        doc = json.loads(f.read_text())
        for label, want in ((f.name + " steps", str(doc["steps"])),
                            (f.name + " wall (s)", f"{doc['wall_s']:.1f}"),
                            (f.name + " dist", doc["dist"])):
            rows.append(row("gpu", label, want, want in sec, None))
        peak = doc.get("peak_mem_gb") or {}
        if peak:
            mp = f"peak {max(peak.values()):.1f} GB"
            rows.append(row("gpu", f.name + " peak mem", mp, mp in sec, None))
    return rows


def _section_text(rep: str, name: str) -> str:
    lines = rep.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith("#"):
            on = name.lower() in ln.lower()
        elif on:
            out.append(ln)
    return "\n".join(out)


def section_structure(rep: str) -> list[dict]:
    rows: list[dict] = []
    heads = [ln.lower() for ln in rep.splitlines() if ln.startswith("#")]
    for name in REQUIRED_SECTIONS:
        ok = any(name.lower() in h for h in heads)
        rows.append(row("structure", f"section '{name}'", name, ok, ok))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(ROOT / "REPORT.md"))
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--emit", action="store_true",
                    help="print canonical claim strings and exit 0")
    a = ap.parse_args(argv)
    results = Path(a.results_dir)
    rep = Path(a.report)
    if not rep.is_file():
        print(f"FAIL {rep} missing", file=sys.stderr)
        return 1
    text = rep.read_text()
    rows = (section_structure(text) + section_sources(text)
            + section_baselines(text) + section_toy(text)
            + section_grpo05(text) + section_lgbox(text)
            + section_logprob(text) + section_gpu(text, results))
    if a.emit:
        for r in rows:
            print(r["want"])
        return 0
    bad = 0
    cur = ""
    for r in rows:
        if r["section"] != cur:
            cur = r["section"]
            print(f"== {cur} ==")
        status = "MISS " if not r["in_report"] else ("PASS " if r["ok"] is not False
                                                     else "FAIL ")
        if not r["in_report"] or r["ok"] is False:
            bad += 1
        print(f"  {status:<5} {r['check']:<34} {r['want']}"
              + (f"  ({r['note']})" if r["note"] else ""))
    print(f"OK: {bad} problem row(s) in {len({r['section'] for r in rows})} sections")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
