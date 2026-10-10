# Ticket: Check baselines against ranges (id: basecheck)

## Outcome — attempt 10 (2026-10-10)
Verdict re-run: exit 1, single FAIL row unchanged (base05 0.0764 [0.014,
0.153] < 0.10; base15 0.3056, ordering + CI-overlap reported, reference 1.0
all PASS); ruling file still absent (.agent-evidence/basecheck10-verdict.log).
Two new facts close out grading of the n=16 extension (samples s4-15, never
independently verified — atk2 covered only s0-3):
1. Eval pipeline (rollout.py, exercism.py, eval/, splits.py) is byte-identical
   between the n=4 job (a5c11cd) and the n=16 job (bf0b283): `git diff
   a5c11cd bf0b283 -- <paths>` empty. s4-15 used the exact code atk2
   re-verified on s0-3 (36/36 regen+regrade matches).
2. Independent regrade of 4 committed new-segment zero-rewards (food-chain
   s5/105, largest-series-product s5/305, point-mutations s5/405,
   scrabble-score s5/605): deterministic regen (T=0.8, top_p=1.0, 512 tok,
   sidecar seeds) + fresh spec-compliant parse (last ```python block) and
   grader (fresh dir, pristine tests, junit XML, tests>0 & no
   fail/err/skip) → 0/4 mismatches; zeros are genuine failures (14/14 LSP
   fail, 10/11 scrabble, 3/8 PM, 1 err food-chain)
   (.agent-evidence/basecheck10-newseg-regrade.log).
Also: 8 of 11 base05 passes sit in s4-15, so the extension's grading awards
passes too — new-segment grading verified in both directions. Ten pipeline
causes ruled out in total; the number is an honest measurement.
**Ruling handoff (range author, one file, then the checker decides
mechanically — channel E2E at atk8):** write
`results/rulings/base05_val.ruling.json`:
- accept: `{"author": "kevin", "decision": "accept",
  "rationale": "task-clustered CI [0.014, 0.153] straddles the 0.10 floor;
  ten pipeline causes ruled out (atk1-10)"}`
- or revise: same + `"decision": "revise", "new_range": [lo, hi]` with
  0.0764 inside. Malformed rulings fail closed (check_ruling, tools/
  check_results.py:61). Agents may not write rulings (rule 5; ticket
  rangerule).
## Outcome — attempt 9 (2026-10-10)
Verdict re-run: `--only baselines` still exits 1, single FAIL row (base05
0.0764 [0.014, 0.153] < 0.10), all other rows PASS; ruling file still absent
(results/rulings/ = .gitkeep only). New evidence, an 8th ruled-out cause:
independent recompute of pass@1 (hand: total-correct/total-samples) and the
task-clustered bootstrap CI (bootstrap_ci seed 0) from the saved JSONs'
per_task fields matches the committed values exactly for base05, base15 and
reference — the result files are internally consistent, so a stale/corrupt
results file is not the cause (.agent-evidence/basecheck9-recompute.log).
Regression: make-check pytest 162 passed / 1 skipped, ruff clean
(.agent-evidence/basecheck9-makecheck.log). Blocker unchanged: Kevin's
one-file ruling (ticket rangerule; agents may not write rulings, rule 5).

## Outcome — attempt 8 (2026-10-09)
No new cause left to chase (all seven ruled out atk1-7). New angle: prove the
unblock path end-to-end on the real input (unit tests use synthetic fixtures
only):
- Ruling-channel smoke, real base05_val.json
  (.agent-evidence/basecheck8-rulingsmoke.log): throwaway accept ruling at the
  exact path the tool derives (check_results.py:52, fname[:-5] +
  ".ruling.json") -> `--only baselines` exit 0, row notes "in range by ruling
  — <author>: <rationale>"; throwaway deleted -> exit 1 (fails closed).
  results/rulings/ back to .gitkeep only; never committed; expresses no
  decision (rule 5 intact).
- Regression: tests/test_check_results.py 11/11; master/branch in sync;
  verdict re-run: single FAIL row unchanged (base05 0.0764 [0.014, 0.153]);
  base15 0.3056 in range, 1.5B > 0.5B (overlap reported), reference 1.0.
Tool is fully ready: Kevin's one ruling file (ticket rangerule: accept the
CI [0.014, 0.153] straddling the 0.10 floor, or revise the range) makes
`--only baselines` exit 0 mechanically. Stopping; blocker unchanged.

## Scope + outcome — attempt 7 (2026-10-09)
All pipeline causes ruled out atk1-6; n=16 landed (base05 0.0764 [0.014, 0.153]);
blocker is the range-author ruling (TODO ticket rangerule). New angle this
attempt: the tool has no way to *express* that ruling. After an "accept"
ruling, `--only baselines` can never exit 0 without an agent editing
BASELINE_RANGES — i.e. the forbidden widening. rangerule says "after the
ruling, check_results.py decides basecheck", so the tool needs a ruling
channel. Fix: results/rulings/<stem>.ruling.json (author/decision/rationale,
new_range for revise), applied verbatim by the tool with the ruling cited in
the table row; malformed rulings fail closed; only the range author writes
these (rule 5). Basecheck itself stays blocked on the actual ruling.
Outcome: done. check_ruling/load_ruling in tools/check_results.py:44-105;
4 new tests (accept, revise incl. out-of-ruled-range, 5 malformed shapes +
unreadable file, ruling-path note) — tests/test_check_results.py 11/11.
Real run: single FAIL row unchanged (base05), now pointing at
rulings/base05_val.ruling.json. make check exit 0 (154 passed, 1 skipped;
.agent-evidence/basecheck7-makecheck.log). results/rulings/ committed with
.gitkeep. Unblock = Kevin commits one ruling file (accept or revise); then
--only baselines decides basecheck mechanically.

## Outcome — attempt 6 (2026-10-09)
Different approach: ruled out the one cause the previous six attempts never
checked — the model itself (the six ruled-out causes were split, grading,
metric, truncation, parse, prompt).
- Verdict re-run (.agent-evidence/basecheck6-verdict.log): single FAIL row
  unchanged, base05 0.0764 [0.014, 0.153] < 0.10; all other rows PASS
  (base15 0.3056 in range, 1.5B > 0.5B, CI overlap reported, reference 1.0).
- Model identity (.agent-evidence/basecheck6-probe.log):
  models/qwen2.5-coder-0.5b-instruct is 988,097,824 bytes bf16 = 494,048,912
  params, config Qwen2 896/24L/14H/kv2/V151936/tied-embed — matches
  Qwen2.5-Coder-0.5B. Capability probe: 4 hand-written trivial coding tasks
  (add, has_duplicate, histogram, word_lengths), the exact baseline prompt
  format (exercism SYSTEM + user template, HF chat template), greedy,
  assert-checked known answers: 4/4 PASS with correct ```python formatting.
  The policy is a working instruct coding model.
- All seven candidate causes ruled out. 0.0764 stands as an honest
  measurement; the 0.10 floor is mis-calibrated for this 9-task val set.
Blocker: rule 5 blocks "done" and agents may never widen a range. Added
TODO.md ticket (id: rangerule) — the range author must either accept the
number (task-clustered CI [0.014, 0.153] straddles the floor) or revise the
range in CLAUDE.md. Only that ruling unblocks the checker exit 0. Stopping.

## Issue
Baseline eval numbers (base05/base15 exercism val) must be checked against the
CLAUDE.md plausible ranges before any GRPO run: a bad prompt, grader, or
truncation would poison every later before/after comparison.

## Question
Are all committed result numbers inside their CLAUDE.md plausible ranges with
known answers holding? Current observation refuting "all good": base05 val
pass@1 = 0.0764 [0.014, 0.153] (n=16; n=4 was 0.0833) vs range 0.10–0.55 —
point estimate below the floor. base15 = 0.3056 [0.132, 0.486] inside
0.25–0.75.

## Evidence
- results/eval/base05_val.json (0.0833), base15_val.json (0.3611),
  reference_exercism_val.json (1.000 known answer, grading+prompt sanity).
- Commit 447e745 probe: 0.000 rewards are genuine (model emits trailing junk),
  not a grader bug.
- Ruling-out probes (all .agent-evidence/basecheck-*.log):
  - probe512: regenerate 9 sample-0 completions, same seeds, T=0.8, 512 tok:
    9/9 rewards match committed sidecar (pipeline deterministic); only 2/9
    capped without EOS (food-chain, vlq).
  - probe1024: those two tasks at 1024 tok, n=4: 0/8 recovered, all end in
    EOS → truncation not the cause.
  - probe-parse: 0-reward failures are model content: isogram check
    `len(dict) == len(set(dict))` is a tautology (always True; direct grade:
    14 tests, 8 failures, reward 0 — exactly the 8 non-isogram cases);
    point-mutations raises ValueError on length mismatch where tests require
    truncating comparison; parse picks the correct last ```python block.
  - Grading cross-check: reference val pass@1 = 1.000; junit counts match
    hand analysis of the model's own code.
- Verdict: n=4 0.0833 and n=16 0.0764 [0.014, 0.153] are honest 0.5B
  measurements at the committed protocol; below the pre-written 0.10 floor;
  CI straddles the floor. atk6 rules out the 7th cause (model identity):
  494M-param Qwen2 config + 4/4 trivial-task probe. atk9 rules out the 8th
  (stale/corrupt results file): recompute from saved per_task matches
  exactly (basecheck9-recompute.log). Rule 5: record as
  out-of-range, never widen the range.

## Done when
`.venv/bin/python tools/check_results.py --only baselines` exits 0: every
number in its CLAUDE.md range (base05 0.10–0.55, base15 0.25–0.75),
1.5B > 0.5B (CIs: non-overlapping, or overlap reported). Known answers:
reference val pass@1 == 1.0. If base05 stays out of range after ruling out
prompt/grading/truncation, record as suspected bug per rule 5 — never widen
the range.

## Stop when
Scope: this ticket only checks baselines (toy/bench sections grow with later
experiments; missing files in other sections skip, not fail). Stop/split:
after two attempts without new evidence on the base05 cause, record the
blocker in NOTES.md and stop.

## Plan
1. tools/check_results.py: load results JSONs, assert ranges, 1.5B>0.5B with
   CI-overlap report, print table, --only filter. Run → expect base05 FAIL.
2. Probe base05 cause: regenerate 0.5B sample-0 completions (same seeds,
   max_new_tokens 512), check new-token count / EOS to test truncation; grade
   with 1024 budget on the failing tasks to see which flip.
3. If a pipeline bug: fix + rerun affected baseline (compute job if >10 min).
   If genuine performance: document evidence, keep range, report per rule 5.
4. check exits 0; make check; NOTES.md; commit; merge master.

## Outcome — attempt 2 (2026-10-09)
`--only baselines` still exits 1 (base05 0.0833 < 0.10 floor); checker unit
tests 7/7. Attempt 2 re-verified the number end-to-end:
- Split audit: the val bucket (sha1%100 in [70,85)) holds 10 exercises;
  `ledger` is excluded from splits.json because its stub scores 1.0 (not
  usable) — legitimate. The 9 committed val tasks are exactly the usable
  ones. (parallel-letter-frequency's ref times out at the 30 s audit and
  passes at 60 s — an audit artifact, but it is in the TEST split, so it
  does not affect these val numbers.)
- Independent metric recompute: a hand-written Chen pass@1 over the
  sidecars matches the committed values exactly (0.083333 / 0.361111); the
  bootstrap CI differs only by RNG seed and still overlaps the 0.10 floor.
- Full re-grade (.agent-evidence/basecheck2-regrade36.log): all 36 base05
  samples regenerated from the recorded seeds (T=0.8, top_p=1.0, 512 tok)
  and re-graded from scratch — 0 mismatches, 0 timeouts, new pass@1
  0.0833 exactly. No pipeline bug; 0.0833 is an honest 0.5B measurement
  (failures are model content: tautology, wrong exceptions, trailing junk).
Resolution: n=16 re-run of both baselines added to TODO.md as a (run:)
compute job (resumes the sidecars, reuses samples 0–3, then re-runs the
checker verdict). Until it lands, rule 5 blocks "done"; the range author
must decide (accept with overlapping CI, or revise the floor).

## Outcome — attempt 5 (2026-10-09)
The n=16 gate job had run in the meantime: job/basen16 (8a7c865, 20:50) holds
complete n=16 output but its (run:) command ends in the checker, which exits 1
on a FAIL verdict, so the job can never exit 0 and its output was stranded
unmerged. Verified it independently: 144 sidecar rows/model (9 tasks x 16),
first 36 rows identical to the committed n=4 sidecars (resume determinism),
pass@1 recomputed from sidecars (0.076389/0.305556) matches the JSONs
exactly. Landed the 4 result files (new ticket landn16, checked off in
TODO.md), re-ran the verdict (.agent-evidence/basecheck5-verdict-n16.log):
- base05 0.0764 [0.014, 0.153] < 0.10 floor — FAIL (unchanged in substance;
  the larger-n re-run reproduced the out-of-range number with 12 fresh
  samples/task: honest 0.5B performance, range mis-calibration, not pipeline —
  all six causes ruled out atk1-4).
- base15 0.3056 [0.132, 0.486] in range; 1.5B > 0.5B; CI overlap reported;
  reference 1.0 — all PASS.
Rule 5: recorded as suspected bug in NOTES.md (range mis-calibration), range
never widened. "Done when" (checker exit 0) remains blocked on the range
author's ruling (accept the overlapping CI, or revise the 0.10 floor in
CLAUDE.md).

## Outcome — attempt 4 (2026-10-09)
New angle: job-dispatch forensics (cause-hunting was exhausted at atk3).
- Checker re-run first: single FAIL row unchanged (base05 0.0833 < 0.10);
  base15 in range, 1.5B > 0.5B, CI overlap reported, reference 1.0 — all PASS.
- `lib/jobs.py` (named by CLAUDE.md and TODO.md:5 as the job runner) exists in
  no branch; TASKLOG.md shows jobs actually run via the harness ("compute job
  base05 ran ... exit 0"). The n=16 re-run is a separate unchecked ticket
  (TODO.md:116) whose (run:) job has not run: both sidecars still 36 lines (n=4).
- TODO.md:116 was the only (run:) line lacking an (id:) tag; every executed job
  line has one, and TASKLOG names jobs by id. Added (id: basen16) so the gate
  job dispatches like the others.
Blocker unchanged: rule 5 holds until the n=16 job lands and the range author
rules (accept CI [0, 0.167] overlap, or revise the floor).

## Outcome — attempt 3 (2026-10-09)
Closed the last open cause with non-circular evidence (attempts 1–2 used
the pipeline itself; this is a direct spec comparison):
- Prompt-format byte audit: exercism.py:18-19 SYSTEM and :49-50 user
  template are byte-identical to the CLAUDE.md spec (system string,
  "\n\nComplete this file (`X`):\n```python\n<stub>\n```", append joined
  with "\n\n"); parse = LAST ```python block else whole reply (exercism.py:
  53-66), matching the spec. run.py:103-104 uses exactly Task.prompt()/
  parse(); defaults n=4 T=0.8 top_p=1.0 max_new_tokens=512 seed=0 match
  the committed JSONs; HFBackend forces top_k=0, rep=1.0 (rollout.py:77-81)
  so sampling is exactly temperature+top_p; seed scheme seed+idx*100+i
  matches the sidecars.
- Checker unit tests 7/7; `--only baselines` exits 1 on exactly one row:
  base05 0.0833 [0, 0.167] vs [0.10, 0.55]. base15 0.3611 [0.167, 0.583]
  in range; ordering + CI-overlap (reported: touch at 1/6) + reference
  1.000 all PASS.
All six candidate causes (split, grading, metric, truncation, parse,
prompt) are now ruled out. 0.0833 is an honest 0.5B measurement. Blocker
unchanged and now fully documented: the n=16 (run:) job at TODO.md:116
(after: -) must run, then the range author rules. Stopping per rule 5;
no code changes this attempt.
