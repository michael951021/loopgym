# vb2 — Hidden checks, batch 2 (candidates 13–24)

## Issue
Candidates 13–24 in data/loopgym/candidates.jsonl have no hidden checks, so none is an RL task
yet. Batch 1 verified 12/12 (state/verify/batch1.md); the yield of this batch decides whether
batch 3 (25–40, < 30% rule) is worth it.

## Question
Is state/verify/batch2.md correctly instantiated (12 unchecked steps, score order, fields matching
jsonl rows 13–24) — and can candidate 13, draftlab-b7c6e4f-1f173d1, get a `verified` verdict?
Refuted by a step/row field mismatch or a verify run that is not 0/3 bad and 3/3 good.

## Evidence
- Row 13: bad=b7c6e4f (jaxmodel close-out), good=1f173d1 (spec bias fix, same good commit as
  verified batch-1 candidate draftlab-a75ef10-1f173d1). Defect diff per mine.py is 35 NOTES.md
  lines, but the behaviour gap is real: draftlab/jaxlm/spec.py (lossless T>0 drafting) exists only
  at good. At bad, `from draftlab.jaxlm import spec` fails.
- verify.py:91 — any run with tests==0 ⇒ verdict `error`; a module-level import of spec in the
  check would collection-error at bad. So the check must import spec lazily inside tests (import
  failure ⇒ test failure, tests>0). jaxlm/__init__.py is a docstring at both commits, so
  `model` imports fine at bad (Qwen2Cfg/init_params/prefill present).
- Reuse batch-1's proven check shape (tests/hidden/draftlab-a75ef10-1f173d1/): 20k-trial first-token
  chi-square (good p=0.81) + NGramDrafter statelessness, with _spec() lazy import.
- Field-match output: .agent-evidence/batch2-instantiate.log.

## Done when
state/verify/batch2.md exists (12 unchecked steps, score order, `Verdict: (fill in)`); field-match
exits 0; for candidate 13: tests/hidden/draftlab-b7c6e4f-1f173d1/test_*.py (≤ 60 lines) exists and
`.venv/bin/python -m loopgym.data.verify draftlab-b7c6e4f-1f173d1` prints its verdict, recorded on
the step line, step checked. Known answer: the good side is the same commit as batch-1's verified
check (chi-square p=0.81 there, must stay > 0.01); bad side must fail, not collection-error.
TODO vb2 line left unchecked; NOTES.md updated; committed.

## Stop when
Scope: state/tickets/vb2.md, state/verify/batch2.md, tests/hidden/draftlab-b7c6e4f-1f173d1/, NOTES
lines for this area. Other candidates are for later runs (one per run). Two attempts at candidate
13 without `verified` ⇒ mark not-checkable with reason, check the step, stop.

## Plan
1. Instantiate batch2.md from templates/verify_step.md (rows 13–24, fix_msg ≤ 15 words).
2. Field-match script vs jsonl rows 13–24 via capture.py.
3. Write candidate-13 check (lazy import of spec; chi2 m=1 path + n-gram statelessness).
4. Run verify; record verdict; check step. NOTES, commit, merge master.
