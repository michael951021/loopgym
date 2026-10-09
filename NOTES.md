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
- loopgym/ has __init__.py + models.py; `make known` still fails until env/kernels exist (expected).
- Evidence: .agent-evidence/boot-cleanslate.log (clean-slate, exit 0, 67s), boot-accept3.log.

## Models (models — done 2026-10-09)
- `loopgym/models.py`: fetch (snapshot_download, weights+tokenizer+min metadata), info,
  tiny_config (2L/d64/4H/2KV/V512). CLI: `python -m loopgym.models {fetch,info}`.
- Both models downloaded to models/ (bind mount): 0.5b 954 MB, 1.5b 3.09 GB, single
  model.safetensors each, BF16. info: 494,032,768 / 1,543,714,304 params, vocab=151936,
  tokenizers byte-identical, 2.3 s. Param counts == HF from_pretrained live count (0.5b).
- GOTCHA: 151936 is config vocab_size (embedding dim); tokenizer length is 151,643. Kernel and
  training numbers refer to 151936.
- Evidence: .agent-evidence/models-fetch.log, models-info2.log; unit tests in tests/test_models.py.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS (shared between agents): clear with
  `find DIR -mindepth 1 -delete`, never `rm DIR` (EBUSY).
- rust pin: static.rust-lang.org 404s for 1.100.0/1.101.0 (dist lag vs channel toml); 1.99.0 verified.
- ruff 0.16 defaults drifted (adds I/EXE) — rule set pinned in [tool.ruff.lint].
- Version probes in bootstrap.sh are format-fragile: shellcheck prints `version:` (lowercase),
  tflint prints two lines. Fix the awk if a version bump changes output.
- tests/known/ has unsorted/unused imports (I001/F401) — Kevin's, read-only; left as-is.

## Data / known tests
- tests/known/ = Kevin's known-answer tests (read-only, CLAUDE.md rule 1).
- data/raw/ = host export (tools/host_export.py): exercism-python, draftlab + harness mirrors,
  ticket_review.json, runs.jsonl.
