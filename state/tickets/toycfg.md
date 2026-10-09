# toycfg — Toy known-answer configs

## Issue
The planted-effect (banana) and null-control (coinflip) runs from CLAUDE.md have no
runnable config or verdict script, so the "trainer learns signal, ignores noise" gate
before any real GRPO job cannot be executed or checked.

## Question
Can the 0.5B trainer, driven only by config files, reproduce both known answers —
banana reward < 0.2 → > 0.8 over 60 steps, and coinflip slope CI containing 0 — and
does a 2-step banana probe finish in < 5 min?

## Evidence
- `loopgym/env/toy.py`: banana/coinflip rewards, `ToyTask`.
- `loopgym/rl/train.py`: `TrainCfg` fields, CLI `--field value` overrides,
  metrics.jsonl records with `step`/`reward_mean` (eval lines carry `kind: "eval"`).
- `configs/smoke.toml`: shape of a config (tiny model).
- Reproduction: `python -m loopgym.rl.train --config configs/toy_banana.toml --steps 2
  --out runs/toy_probe` (evidence: .agent-evidence/toy_probe2.log).
- Guess (to be confirmed by the run): 24-token rollouts make each step far cheaper than
  the 256-token 5–25 min known range, so 2 steps < 5 min.

## Done when
1. `configs/toy_banana.toml` / `configs/toy_null.toml`: 0.5b, LoRA r=16 q/k/v/o, B=4,
   G=8, max_new_tokens 24, lr 5e-5, beta 0, 60 steps, save_every 10.
2. `tools/toy_check.py`: reads runs/toy_{banana,null}/metrics.jsonl (train lines only),
   prints mean reward steps 0–4 vs 55–59 and bootstrap CI of the per-step slope; exits
   1 unless banana < 0.2 → > 0.8 and null slope CI contains 0.
3. Check runs first and passes: 2-step banana probe exits 0 in < 5 min (range written
   here before the run: 1–5 min, from the 256-token 5–25 min known range scaled by
   token count).
4. `tests/test_toy_check.py`: synthetic pass/fail jsonl verdicts; `make check` green.

## Stop when
Out of scope: running the full 60-step jobs (separate TODO lines after this one),
vLLM, any other model. If the 2-step probe exceeds 5 min twice with no new evidence
(e.g. thread starvation), record it in NOTES.md and stop/split.

## Plan
1. Write both configs (clone smoke.toml shape, real-model values).
2. Write tools/toy_check.py (train-line filter, window means, bootstrap slope CI).
3. Test it on synthetic pass/fail data; add tests/.
4. Run the 2-step probe with OMP_NUM_THREADS=8, time it, run `make check`.
