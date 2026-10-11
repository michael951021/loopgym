# grpocfg — CPU GRPO config

## Issue
No training config exists for the gated 0.5B CPU GRPO run. The trainer (`loopgym/rl/train.py`)
cannot express the two things the ticket requires: a difficulty cap on the exercism train split,
and eval on the **val** split (current `eval_step` re-rolls the *train* step's tasks).

## Question
Can `configs/grpo05_exercism.toml` (0.5B LoRA r=16, train split difficulty ≤ 3, B=4, G=8,
384 new tokens, lr 2e-5, β 0.02, 40 steps, val eval n=2 every 20 steps, save_every 5) run
one real step end to end? Refutation: the 1-step probe exits non-zero, or its metrics line
lacks `reward_mean`/`frac_groups_all_same`, or the eval row is not over val-split tasks.

## Evidence
- `loopgym/rl/train.py:TrainCfg` (no difficulty/val-eval fields), `Run.eval_step` (line 225)
  grades the current step's tasks, not val.
- `results/splits.json`: `{bucket: {name: difficulty}}`; train 101, 67 at difficulty ≤ 3, val 9.
- `models/qwen2.5-coder-0.5b-instruct` present.
- Reproduction: the Done-when probe command. Output saved under `.agent-evidence/grpocfg-probe.log`.
- Observed: trainer works on toy (toybanana/toynull gates passed). Guess: 0.5B fp32 one step
  on CPU is ~5–25 min (CLAUDE.md plausible range for B=4 G=8 256 tok; 384 tok may run longer).

## Done when
`.venv/bin/python -m loopgym.rl.train --config configs/grpo05_exercism.toml --steps 1 --out runs/grpo_probe`
exits 0 and its `metrics.jsonl` step-1 line has `reward_mean` and `frac_groups_all_same`.
Known-answer checks (written before the run): reward_mean must land in 0.0–0.60 (base 0.5B
exercism pass@1 range 0.10–0.55 at T=0.8; T=1.0 sampling over 32 completions cannot plausibly
exceed it, and 0.0 with all groups identical flags the grader); trainer correctness itself is
pinned by `tests/test_train.py` (resume == uninterrupted, bit-exact) and the toy gates.
Regression coverage: new tests for the difficulty filter and the val-eval draw in tests/test_train.py.

## Observation (step-1 probe)
Probe exited 0 in 524 s; metrics line: reward_mean 0.25 (in 0.0–0.60 range; base 0.5B
exercism pass@1 0.10–0.55), frac_groups_all_same 0.25, loss 0.0957, kl 0.0 (exact: LoRA
B-init zero ⇒ policy == ref ⇒ k3 = 0; old = detach ⇒ ratio = 1). loss 0.0957 > CLAUDE.md's
initial-loss 0.05 range. Hypothesis (testable, not a finding): agg="token-mean" weights
per-sequence advantages by completion length, so length-reward correlation shifts the
mean; seq-mean must be exactly 0. Control: same probe with `--agg seq-mean`
(.agent-evidence/grpocfg-probe-seqmean.log) — |loss| < 1e-3 confirms artifact, else bug.
**Result: confirmed.** Same seed ⇒ identical completions (reward 0.250 again); seq-mean
loss = -0.0 (exact zero), token-mean 0.0957. Not a bug; recorded in NOTES.md. Config keeps
the trainer default agg="token-mean" (ticket does not pin agg).

## Stop when
Out of scope: the 40-step run (grpo05 job), GPU/vLLM paths, lgbox runner, loopgym task source.
Stop/split: after two probe attempts without new evidence (e.g. crash inside torch, OOM, or
reward 0.0 on all 32 completions that survives a hand-graded reference check on the same tasks),
record the blocker in NOTES.md and stop.

## Plan
1. Run probe now → expect failure (config missing).
2. `TrainCfg`: add `max_difficulty` (0 = off), `eval_n` (0 = legacy same-task eval);
   filter train sampler; stateless per-step val draw (seed + EVAL_SEED_OFFSET + s) so resume is safe.
   Add both to RESUME_CHECK with dataclass-default backfill for old checkpoints.
3. Write config; run probe to completion; verify metrics line.
4. Tests: filter correctness vs splits.json; val draw ⊆ val, deterministic per step. ruff + targeted pytest.
