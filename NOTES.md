# Notes

Current state for the next run (keep < 4 KB).

## Foundations (boot — done 2026-10-09)
- `make bootstrap` = idempotent toolchain (tools/bootstrap.sh, pins tools/versions.txt):
  uv 0.12.24, .venv py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1,
  shellcheck 0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin; evidence
  .agent-evidence/boot-cleanslate.log (exit 0).
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest tests (minus known) +
  cargo test lgbox/ + shellcheck + hadolint + terraform init -backend=false && validate
  (last three skip until lgbox/, deploy/ exist). `make known` = check_known.sh (kevin-only
  authors, clean worktree) + strict pytest tests/known. test_grpo 7/7 + test_reward 9/9 strict;
  still red: {test_logprob,test_stats}.py import loopgym.rl/kernels/eval (not built yet).
- venv: torch 2.14.1+cpu (CPU index [tool.uv.sources]), transformers 5.19.0, peft, safetensors,
  accelerate, numpy, pydantic, httpx, pytest, ruff, tomli-w; uv.lock committed.

## Models (models — done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI (`python -m loopgym.models`), tiny_config (2L/d64/4H/2KV/V512);
  both models in models/: 0.5b 954 MB / 1.5b 3.09 GB, BF16, single model.safetensors; params
  494,032,768 / 1,543,714,304 == live HF count; tokenizers byte-identical.
- GOTCHA: 151936 = config vocab (embedding dim), tokenizer length 151,643 (kernel/training use 151936).
  Evidence: .agent-evidence/models-{fetch,info2}.log, tests/test_models.py.

## Env: exercism + toy + splits (done 2026-10-09)
- `loopgym/env/exercism.py`: Task (stub/reference/prompt/parse) + load_exercise; runner_names()=["local"]:
  fresh tmpdir, pristine tests + ONLY solution file, pytest junit XML, new process group, SIGKILL on
  timeout; reward 1 iff tests>0, no failures/errors/skips. Evidence: .agent-evidence/exercism-{accept2,makecheck3}.log.
- `loopgym/env/toy.py`: ToyTask (prompt "Name a fruit.", parse→reply.txt), load_toy("banana"|"coinflip");
  grade(task, files, step, index) → (reward, info), in-process; `env.grade` dispatches on ToyTask
  (kwargs w/ defaults, exercism path unchanged). banana = 1 iff "banana" (ci) in decode of first 20
  tokens (tokenizer cached); coinflip = 1 w.p. 0.5 from random.Random(f"...:{step}:{index}"), ignores
  text. Known: "I like Banana"=1, "apple"=0, "fig "*21+"banana"=0; coinflip 10k mean 0.4945,
  text-independent. 12/12 test_toy.py, known reward 9/9. Evidence: .agent-evidence/toy-{makecheck2,makeknown}.log.
- `loopgym/env/splits.py`: audit() grades ref+stub of all 142 practice exercises (4 threads, local);
  `python -m loopgym.env.splits` writes results/env_audit.json (unusable + reason) and
  results/splits.json. bucket(name) = sha1 % 100: train<70, val<85, test. 137/142 usable;
  splits 101/9/27. Unusable: alphametics, error-handling (ref=0), ledger, markdown (stub=1 — leaky
  tests), parallel-letter-frequency (ref timeout). Evidence: .agent-evidence/splits-{accept,makecheck}.log.

## Pinned sources (vsrc — done 2026-10-09)
- `tools/fetch_src.sh`: latest stable vllm/deepspeed/trl from PyPI → read-only
  .tools/src/<name>-<version>/; pins in tools/versions.txt (vllm=0.31.0, deepspeed=0.19.7, trl=1.15.0);
  never installs into .venv; vllm wheel = aarch64, deepspeed = sdist.
  Evidence: .agent-evidence/vsrc-{fetch,rerun,makecheck}.log (exit 0).

## RL: GRPO math (grpo — done 2026-10-09)
- `loopgym/rl/grpo.py`: group_advantages(rewards[B,G], scale="std"|"none", eps=1e-6) = (r−mean)/
  (sample_std+eps) per row (Bessel std; constant row → 0); kl_k3(logp,ref) = exp(d)−d−1,
  d=ref−logp; policy_loss(logp,old,ref,adv[B],mask[B,T], clip_eps, kl_coef, agg) →
  (loss, {clip_frac, kl, ratio_mean}): ratio=exp(logp−old), obj=min(ratio·A, clip(ratio)·A)−β·k3;
  token-mean = −Σobj·mask/Σmask, seq-mean = per-row masked mean then row mean; clip_frac
  sign-aware over masked tokens. 7/7 tests/known/test_grpo.py (incl. 200k-sample k3 unbiasedness);
  make check green. Evidence: .agent-evidence/grpo-{first,known1,makecheck}.log.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared): clear with `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag); 1.99.0 verified.
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- bootstrap.sh version probes are format-fragile (shellcheck/tflint output); fix the awk on version bumps.
- tests/known/ has I001/F401 (unsorted/unused imports) — Kevin's, read-only; left as-is.

## Data
- tests/known/ = Kevin's known-answer tests (read-only, rule 1); data/raw/ = host export:
  exercism-python, draftlab+harness mirrors, ticket_review.json, runs.jsonl.
