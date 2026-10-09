# Notes

Current state for the next run (keep < 4 KB).

## Foundations (boot — done 2026-10-09)
- `make bootstrap` = idempotent toolchain (tools/bootstrap.sh, pins in tools/versions.txt):
  uv 0.12.24, .venv py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1,
  shellcheck 0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin; evidence
  .agent-evidence/boot-cleanslate.log (exit 0).
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + `pytest -q tests --ignore=tests/known`
  + cargo test lgbox/ + shellcheck + hadolint + `terraform init -backend=false && validate`;
  last three skip until lgbox/ and deploy/ exist. `make known` = check_known.sh (kevin-only
  authors, clean worktree) + strict pytest tests/known. Still red: {test_grpo,test_logprob,
  test_stats}.py import loopgym.rl/kernels/eval (not built); test_reward.py 9/9 strict.
- venv: torch 2.14.1+cpu (CPU index [tool.uv.sources]), transformers 5.19.0, peft, safetensors,
  accelerate, numpy, pydantic, httpx, pytest, ruff, tomli-w; uv.lock committed.

## Models (models — done 2026-10-09)
- `loopgym/models.py`: fetch (snapshot_download, weights+tokenizer+min metadata), info,
  tiny_config (2L/d64/4H/2KV/V512); CLI `python -m loopgym.models {fetch,info}`; evidence
  .agent-evidence/models-{fetch,info2}.log, tests/test_models.py.
- Both models in models/ (bind mount): 0.5b 954 MB / 1.5b 3.09 GB, single model.safetensors, BF16;
  params 494,032,768 / 1,543,714,304 == live HF count; tokenizers byte-identical. GOTCHA: 151936 =
  config vocab (embedding dim), tokenizer length 151,643 (kernel/training use 151936).

## Env: exercism + toy (done 2026-10-09)
- `loopgym/env/exercism.py`: Task (stub/reference/prompt/parse) + load_exercise
  (.meta/config.json; paasio has 2 test files); runner_names()=["local"]. Local: fresh tmpdir
  w/ pristine tests + ONLY solution file, pytest --junitxml, new process group, SIGKILL on timeout;
  reward 1 iff XML tests>0, no failures/errors/skips (os._exit → no XML → 0); evidence
  .agent-evidence/exercism-{accept2,makecheck3}.log.
- `loopgym/env/toy.py`: ToyTask (prompt "Name a fruit.", parse→{"reply.txt": reply},
  stub/reference) + load_toy("banana"|"coinflip"); grade(task, files, step, index) → (reward, info),
  in-process, no runner; banana = 1 iff "banana" (ci) in decode of first 20 tokens (0.5B tokenizer,
  cached); coinflip = 1 w.p. 0.5, random.Random(f"...:{step}:{index}"), ignores text.
  `env.grade` dispatches on ToyTask (step/index kwargs, defaults; exercism path unchanged).
  Known answers: banana("I like Banana")=1 (token 75824), "apple"=0, "fig "*21+"banana"=0
  (22 tokens); coinflip 10k mean 0.4945, per-key identical across texts; 12/12 test_toy.py,
  make check green, known reward 9/9 strict.
  Evidence: .agent-evidence/toy-{makecheck2,makeknown}.log.

## Pinned sources (vsrc — done 2026-10-09)
- `tools/fetch_src.sh` (idempotent): latest non-prerelease of vllm/deepspeed/trl from pypi.org
  JSON → wheel (sdist if no wheel) → unpacked read-only in .tools/src/<name>-<version>/; pins in
  tools/versions.txt (vllm=0.31.0, deepspeed=0.19.7, trl=1.15.0); never installs into .venv;
  vllm wheel = aarch64 (alphabetical min cp38-abi3).
  Checks: rg load_weights vllm-*/ (13), DeepSpeedConfig deepspeed-*/ (2); trl sha256 == PyPI;
  evidence .agent-evidence/vsrc-{fetch,rerun,makecheck}.log (exit 0).

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared): clear with `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag); 1.99.0 verified.
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- bootstrap.sh version probes are format-fragile (shellcheck `version:`, tflint 2 lines); fix the awk on version bumps.
- tests/known/ has I001/F401 (unsorted/unused imports) — Kevin's, read-only; left as-is.

## Data
- tests/known/ = Kevin's known-answer tests (read-only, rule 1); data/raw/ = host export:
  exercism-python, draftlab+harness mirrors, ticket_review.json, runs.jsonl.
