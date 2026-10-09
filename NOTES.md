# Notes

Current state for the next run (keep < 4 KB).

## Foundations (boot — done 2026-10-09)
- `make bootstrap` = idempotent toolchain (tools/bootstrap.sh, pins in tools/versions.txt):
  uv 0.12.24, .venv py3.12.14, rust 1.99.0 minimal (.tools/{rust,cargo}), hadolint 2.15.1,
  shellcheck 0.11.0, terraform 1.16.5, tflint 0.64.0 in .tools/bin.
- `make check` = ruff (E4/E7/E9/F, tests/known excluded) + `pytest -q tests --ignore=tests/known`
  + cargo test in lgbox/ + shellcheck + hadolint + `terraform init -backend=false && validate`;
  the latter three skip with a message until lgbox/ and deploy/ exist. `make known` =
  tools/check_known.sh (kevin-only authors, clean worktree) + strict pytest tests/known.
- venv: torch 2.14.1+cpu (CPU index via [tool.uv.sources]; nvidia-guard in
  tests/test_bootstrap.py), transformers 5.19.0, peft, safetensors, accelerate, numpy,
  pydantic, httpx, pytest, ruff, tomli-w. Deps locked in uv.lock (committed).
- loopgym/ has __init__.py, models.py, env/; `make known` still fails until rl/kernels/eval exist.
- Evidence: .agent-evidence/boot-cleanslate.log (clean-slate, exit 0, 67s), boot-accept3.log.

## Models (models — done 2026-10-09)
- `loopgym/models.py`: fetch (snapshot_download, weights+tokenizer+min metadata), info,
  tiny_config (2L/d64/4H/2KV/V512). CLI: `python -m loopgym.models {fetch,info}`.
- Both models downloaded to models/ (bind mount): 0.5b 954 MB, 1.5b 3.09 GB, single
  model.safetensors each, BF16. info: 494,032,768 / 1,543,714,304 params, vocab=151936,
  tokenizers byte-identical; param count == HF from_pretrained live count (0.5b).
- GOTCHA: 151936 is config vocab_size (embedding dim); tokenizer length is 151,643. Kernel and
  training numbers refer to 151936.
- Evidence: .agent-evidence/models-fetch.log, models-info2.log; unit tests in tests/test_models.py.

## Env (exercism — done 2026-10-09)
- `loopgym/env/`: Task (stub/reference/prompt/parse) + load_exercise (.meta/config.json;
  paasio has 2 test files); grade(task, files, runner="local", timeout=30) → (reward, info);
  runner_names()=["local"]. Local: fresh tmpdir w/ pristine tests + ONLY solution file,
  `python -m pytest -q -p no:cacheprovider --junitxml=r.xml` new process group, SIGKILL on
  timeout; reward 1 iff XML tests>0 and failures/errors/skips=0 (os._exit → no XML → 0).
- 9/9 known reward tests strict, make check green; prompt == CLAUDE.md template.
  Evidence: .agent-evidence/exercism-accept2.log, exercism-makecheck3.log.

## Pinned sources (vsrc — done 2026-10-09)
- `tools/fetch_src.sh` (idempotent): latest non-prerelease of vllm/deepspeed/trl from
  pypi.org JSON → wheel (sdist if no wheel) via curl → unpacked read-only in
  .tools/src/<name>-<version>/; pins recorded in tools/versions.txt (vllm=0.31.0,
  deepspeed=0.19.7, trl=1.15.0). Re-run skips in 0.6 s. Never installs into .venv.
- vllm wheel chosen = aarch64 (alphabetical min among cp38-abi3 wheels); source identical,
  fine for reading. deepspeed has no wheel → sdist, flattened one dir level.
- Checks: rg 'def load_weights' in vllm-*/ (13 files) and 'class DeepSpeedConfig' in
  deepspeed-*/ (2 files); trl wheel sha256 == PyPI digest.
- Evidence: .agent-evidence/vsrc-fetch.log, vsrc-rerun.log, vsrc-makecheck.log (all exit 0).

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared between agents): clear with
  `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag vs channel toml); 1.99.0 verified.
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- bootstrap.sh version probes are format-fragile (shellcheck prints `version:`, tflint two
  lines); fix the awk if a version bump changes output.
- tests/known/ has unsorted/unused imports (I001/F401) — Kevin's, read-only; left as-is.

## Data / known tests
- tests/known/ = Kevin's known-answer tests (read-only, CLAUDE.md rule 1).
- data/raw/ = host export (tools/host_export.py): exercism-python, draftlab + harness mirrors,
  ticket_review.json, runs.jsonl.
