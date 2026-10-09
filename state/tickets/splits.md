# Exercise audit and splits

## Issue
Training/eval need a fixed, leak-free task list where every task is gradeable. Nothing yet
verifies which of the 142 Exercism practice exercises actually grade (reference → 1.0, stub →
0.0) on the local runner, and there is no train/val/test split. Unusable tasks would inject
noise into the reward; without a split we could train on eval tasks (leak).

## Question
Which exercises are usable under the local runner (ref = 1.0 AND stub = 0.0), and does a
deterministic sha1(name) 70/15/15 split partition the usable set disjointly? Refutation: an
exercise with ref = 1.0 and stub = 0.0 missing from the usable list, or a usable name in two
buckets / no bucket.

## Evidence
- data/raw/exercism-python/exercises/practice: 142 dirs, each .meta/config.json (observed).
- Difficulty: data/raw/exercism-python/config.json → exercises.practice (slug, difficulty), 142
  entries, difficulties 1–9 (observed).
- loopgym/env/exercism.py: load_exercise, grade(task, files, "local", timeout).
- Acceptance: `.venv/bin/python -m loopgym.env.splits` — currently ModuleNotFoundError
  (expected failure); log .agent-evidence/splits-accept.log.

## Done when
Acceptance command exits 0, prints ≥ 125 usable of 142 and the three split sizes;
results/env_audit.json lists each unusable exercise with a reason; results/splits.json holds
usable names per bucket with difficulty. Known answers: sha1 published value pinned in
tests/test_splits.py; split buckets disjoint and complete (recomputed in the test); 3-fixture
mini-audit graded for real (good → usable, broken-ref → unusable). Plausible: 120–142 usable
(most practice refs are pure stdlib).

## Stop when
Scope: local runner only (no lgbox), no reward shaping, no data changes. If < 125 usable, that
is a finding (report the reasons), not a failure to hide. Two attempts without new evidence →
record blocker, stop or split.

## Plan
1. Run acceptance, record the expected ModuleNotFoundError.
2. loopgym/env/splits.py: ThreadPoolExecutor(4) over 142 dirs grading ref+stub; summarize →
   env_audit.json; sha1(name) % 100 → train<70, val<85, test → splits.json with difficulty.
3. tests/test_splits.py (sha1 known value, bucket thresholds, mini-audit end-to-end).
4. make check, commit.
