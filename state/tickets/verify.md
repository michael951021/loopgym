# verify

## Issue
Mined loopgym candidates claim a defect "fails at bad, passes at good" but nothing
verifies it. Without a gate, a bad candidate poisons the training set. The seed
`draftlab-alpha` (known to be real) must verify, or the harness is broken.

## Question
Does `verify(candidate, check_dir)` correctly build pristine bad/good worktrees from the
bare repo, run the hidden check 3× per side under the local-runner rules, and return
`verified` (bad 0/3, good 3/3) for a check that discriminates? Refuted if the seed does not
print `verified`, or if the verdict logic mislabels a bad 0/3 + good 3/3 result.

## Evidence
Seed record: `data/loopgym/candidates.jsonl` id `draftlab-alpha` (bad fda7335, good 2b93e71).
Defects (ticket_review.json): (1) off-by-one logit alignment in `HFDrafter.dists` (bad slices
`logits[0,-n:]`, fix slices `logits[L-1:L+n-1]`); (2) transposed bootstrap multinomial.
Check = per-position CE of `dists(ctx,ids)` must equal HF `labels=`-loss-aligned CE.
Probe (probe_alpha.log): model load 2.1s, fwd+loss 0.3s, anchor diff 1.2e-6.
Hidden check: `tests/hidden/draftlab-alpha/test_alpha_alignment.py`.

## Done when
`.venv/bin/python -m loopgym.data.verify draftlab-alpha` prints `verified` (exit 0), on the real
0.5B checkpoint. Known-answer check: the check's reference alignment reproduces
`model(input_ids=full, labels=full).loss` within 1e-2 (HF's own loss, independent of the code
under test) — proven in the probe. Plausible: bad side fails 3/3 (off-by-one rows), good side
passes 3/3. Regression: `tests/test_verify.py` (worktree + verdict logic on a fake candidate).
`make known` still green; `make check` green.

## Stop when
Only the verify harness + the draftlab-alpha hidden check + the `[draftlab]` pyproject extra +
a test. NOT the batch-1/2 hidden checks (separate tickets). Stop/split after 2 failed attempts
to make the seed print `verified`; record the blocker.

## Plan
1. Write `tests/hidden/draftlab-alpha/test_alpha_alignment.py` (per-position CE == labels= loss).
2. Write `loopgym/data/verify.py`: worktrees from bare repo, 3×/side pristine runs, verdict,
   append `data/loopgym/verified.jsonl`; CLI prints verdict.
3. Add `[draftlab]` extra to `pyproject.toml`.
4. `tests/test_verify.py`; run seed; `make check`/`make known`.
