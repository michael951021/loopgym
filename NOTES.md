# Notes

Current state for the next run (keep < 4 KB).

## Foundations (boot — done 2026-10-09)
- `make bootstrap` = idempotent toolchain (tools/bootstrap.sh, pins tools/versions.txt):
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
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + pytest tests (minus known) +
  cargo test in lgbox/ + shellcheck + hadolint + terraform init -backend=false && validate
  (last three skip until lgbox/, deploy/ exist). `make known` = tools/check_known.sh
  (kevin-only, clean worktree) + strict pytest tests/known.
- venv: torch 2.14.1+cpu, transformers 5.19.0, peft, safetensors, accelerate, numpy,
  pydantic, httpx, pytest, ruff, tomli-w; locked in uv.lock.
- `make known` fails until rl/kernels/eval exist (pre-existing).
- Evidence: .agent-evidence/boot-cleanslate.log (clean-slate, exit 0, 67s).

## Models (models — done 2026-10-09)
- `loopgym/models.py`: fetch (snapshot_download, weights+tokenizer+min metadata), info,
  tiny_config (2L/d64/4H/2KV/V512); CLI `python -m loopgym.models {fetch,info}`; evidence
  .agent-evidence/models-{fetch,info2}.log, tests/test_models.py.
- Both models in models/ (bind mount): 0.5b 954 MB / 1.5b 3.09 GB, single model.safetensors, BF16;
  params 494,032,768 / 1,543,714,304 == live HF count; tokenizers byte-identical. GOTCHA: 151936 =
  config vocab (embedding dim), tokenizer length 151,643 (kernel/training use 151936).
## Models (done 2026-10-09)
- `loopgym/models.py`: fetch/info CLI (`python -m loopgym.models`), tiny_config (2L/d64/
  4H/2KV/V512). Both models in models/ (bind mount): 0.5b 954 MB, 1.5b 3.09 GB, BF16,
  single model.safetensors; params 494,032,768 / 1,543,714,304 (== HF live count).
- GOTCHA: 151936 = config vocab_size (embedding dim); tokenizer length is 151,643.
- Evidence: .agent-evidence/models-info2.log; tests/test_models.py.

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
## Env (exercism — done 2026-10-09)
- `loopgym/env/`: Task (stub/reference/prompt/parse), load_exercise (.meta/config.json;
  paasio: 2 test files); grade(task, files, runner="local", timeout=30) → (reward, info).
  Local: fresh tmpdir, pristine tests + ONLY solution file, pytest junit XML, new process
  group, SIGKILL on timeout; reward 1 iff tests>0 and failures/errors/skips=0.
- 9/9 known reward tests strict; make check green.
  Evidence: .agent-evidence/exercism-accept2.log.

## Pinned sources (vsrc — done 2026-10-09)
- `tools/fetch_src.sh` (idempotent): latest non-prerelease of vllm/deepspeed/trl from pypi.org
  JSON → wheel (sdist if no wheel) → unpacked read-only in .tools/src/<name>-<version>/; pins in
  tools/versions.txt (vllm=0.31.0, deepspeed=0.19.7, trl=1.15.0); never installs into .venv;
  vllm wheel = aarch64 (alphabetical min cp38-abi3).
  Checks: rg load_weights vllm-*/ (13), DeepSpeedConfig deepspeed-*/ (2); trl sha256 == PyPI;
  evidence .agent-evidence/vsrc-{fetch,rerun,makecheck}.log (exit 0).
## Splits (done 2026-10-09)
- `loopgym/env/splits.py`: audit() grades ref+stub of all 142 practice exercises (4 threads,
  local); `python -m loopgym.env.splits` (37 s) writes results/env_audit.json (unusable +
  reason) and results/splits.json (bucket: {name: difficulty}; difficulty from
  config.json:exercises.practice). bucket(name) = sha1 % 100: train<70, val<85, test.
- 137/142 usable; splits 101/9/27. Unusable: alphametics, error-handling (ref=0),
  ledger, markdown (stub=1 — leaky tests), parallel-letter-frequency (ref timeout).
- tests/test_splits.py: sha1 known value, buckets, disjointness, 3-fixture mini-audit
  (real pytest). GOTCHA: fixture "bad ref" must actually fail the test.
- Evidence: .agent-evidence/splits-accept.log (exit 0), splits-makecheck.log.

## Pinned sources (done 2026-10-09)
- `tools/fetch_src.sh` (idempotent, re-run 0.6 s): latest stable vllm/deepspeed/trl from
  PyPI JSON → wheel (sdist if none) → read-only .tools/src/<name>-<version>/; pins in
  tools/versions.txt (vllm=0.31.0, deepspeed=0.19.7, trl=1.15.0). Never installs in .venv.
  deepspeed = sdist, flattened one dir. Evidence: .agent-evidence/vsrc-*.log (exit 0).

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared): clear with `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag); 1.99.0 verified.
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared): clear with
  `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- bootstrap.sh version probes are format-fragile (shellcheck `version:`, tflint 2 lines); fix the awk on version bumps.
- tests/known/ has I001/F401 (unsorted/unused imports) — Kevin's, read-only; left as-is.
- bootstrap.sh version probes format-fragile (shellcheck `version:`, tflint two lines);
  fix the awk if a bump changes output.
- tests/known/ has unsorted/unused imports — Kevin's, read-only; left as-is.

## Data
- tests/known/ = Kevin's known-answer tests (read-only, rule 1); data/raw/ = host export:
  exercism-python, draftlab+harness mirrors, ticket_review.json, runs.jsonl.
## Data / known tests
- tests/known/ = Kevin's known-answer tests (read-only, rule 1).
- data/raw/ = host export (tools/host_export.py): exercism-python, draftlab + harness
  mirrors, ticket_review.json, runs.jsonl.
