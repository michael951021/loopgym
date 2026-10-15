# report — REPORT and README

## Issue
The milestone's answer (does GRPO lift pass@1, honestly measured) is not yet a
portfolio artifact: REPORT.md does not exist, README.md is a 7-line stub, and
there is no script proving the report's numbers equal results/. Kevin (and any
reader) cannot check the claims.

## Question
Can REPORT.md state only numbers that a script recomputes from results/ (and
runs/grpo05_exercism/metrics.jsonl), and can tools/check_report.py prove it —
exiting 0 on the committed report and 1 on any stale or missing number?
Refuting observation: checker exits 0 after a report number is perturbed.

## Evidence
- results/: eval/*.json (base05/base15/grpo05 val, both loopgym tests,
  reference), toy.json, grpo05_summary.json, grpo05_hackcheck.json,
  lgbox_bench.json, bench_logprob.json, loopgym_v0.json, mining.json,
  env_audit.json, splits.json, rulings/base05_val.ruling.json.
- results/gpu/ absent (gpurun1/twonode/aws not run yet) → GPU/AWS section is
  plan + PENDING marker; checker enforces that.
- `tools/check_results.py` exits 0 today (model for the checker's style and
  the exact formatting of shared numbers).
- Smallest check (run first): after writing the artifacts,
  `.venv/bin/python tools/check_report.py` must fail on a deliberately stale
  number (perturb one value, expect exit 1) and pass on the committed text.
  Output logged under .agent-evidence/report-*.log.

## Done when
1. `.venv/bin/python tools/check_report.py` exits 0 on the committed REPORT.md.
2. Perturbing any reported number (or deleting the Corrections section) makes
   it exit 1 (tested in the run, not committed as a permanent test — the
   report is data, not code under test).
3. `.venv/bin/python tools/check_results.py` still exits 0.
4. REPORT.md contains: question; both task sources (Exercism 137 usable +
   loopgym v0 with its verification rule "bad 0/3, good 3/3" and kill-criterion
   outcome met: 26 ≥ 25); method; results tables (baselines, toy known
   answers, CPU GRPO with CIs, lgbox bench, logprob bench); what failed and
   why; GPU/AWS plan with PENDING results. README.md: mermaid architecture,
   quickstart, skills map (skill → files).
Known answers the report must carry next to the numbers: reference val = 1.0
(gold solutions), toy banana planted effect / null control, grpo05 paired Δ
recomputed by hand in grpo05_summary.json, lgbox 200/200 ref rewards, logprob
all-impls-vs-f64 ≤ 1e-4. Pre-run ranges (CLAUDE.md): 0.5B val [0.10, 0.55]
(out of range → ruling cited, not explained away), loopgym base [0.0, 0.25],
CPU step 5–25 min, lgbox overhead < 50 ms, fused peak < 1/8 naive.

## Stop when
Scope: no new runs, no retraining, no edits to results/, tests/known/, or
TODO.md; no GPU work. Stop/split: if any report number cannot be derived from
committed files (missing field), record it in NOTES.md and drop the claim
rather than invent it. Two attempts without new evidence → stop.

## Plan
1. state/tickets/report.md (this file).
2. tools/check_report.py: claim registry computed from results/; substring
   contract; structural sections; GPU PENDING branch.
3. REPORT.md with the exact strings the checker emits (`--emit`).
4. README.md: mermaid, quickstart, skills map.
5. Tamper test (expect exit 1), both checkers exit 0, `make check` subset,
   NOTES.md, commit, merge master.
