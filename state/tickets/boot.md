# boot — Bootstrap toolchain and make targets

## Issue
No toolchain, no Makefile: every later ticket (kernels, RL, lgbox, deploy) needs one pinned
Python/Rust/linter environment and one `make check` gate, or parallel agents break each other
silently. `make known` must stay Kevin's judge over tests/known.

## Question
Can a fresh checkout run `make bootstrap && make check` and get a working CPU venv (torch,
transformers, peft importable), a minimal Rust toolchain, and pinned lint/validate binaries —
all idempotently, from tools/versions.txt pins only?
Refuting observation: any of the three commands exits nonzero on a clean tree, or re-running
bootstrap after a fresh .tools/.venv does not converge to the same working state.

## Evidence
- Observed: repo has only tests/known/ (Kevin's), tools/host_export.py, data/raw/. No Makefile,
  pyproject.toml, .venv contents, or .tools contents. Network to pypi/github/hashicorp OK.
- Failing check (run first): `make bootstrap && make check && bash tools/check_known.sh` →
  "No rule to make target 'bootstrap'" (no Makefile). Output: .agent-evidence/boot-accept.log.
- Observed (attempt 1): .venv/.tools/data/models/ckpt/runs are bind mounts shared between agents —
  bootstrap may clear their contents (find -mindepth 1 -delete) but never rm the dir (EBUSY).
- Guesses to verify by running: uv sync honors [tool.uv.sources] for the torch CPU index; uv venv -p 3.12
  finds a 3.12 interpreter without download.

## Done when
On the real tree: `make bootstrap && make check && bash tools/check_known.sh` all exit 0, and
`.venv/bin/python -c "import torch, transformers, peft"` exits 0, with torch from the CPU index
(no nvidia-* packages in the venv). Re-running `make bootstrap` after `rm -rf .tools .venv`
converges again (idempotency). Smoke test tests/test_bootstrap.py (imports) is the regression
coverage for the toolchain.

## Stop when
Not doing: any loopgym source, model download, GPU anything. If a pinned binary's download layout
changed upstream twice without progress, pin a known-good older version in tools/versions.txt and
note it. Two failed bootstrap attempts with no new evidence → record blocker, stop.

## Plan
1. Pin latest versions (uv, hadolint, shellcheck, terraform, tflint, stable rust) in tools/versions.txt.
2. pyproject.toml (loopgym package + deps, torch via CPU index), minimal loopgym/__init__.py.
3. tools/bootstrap.sh (idempotent: uv binary, uv venv 3.12 + editable install, rustup minimal,
   pinned binaries) + tools/check_known.sh + Makefile (bootstrap/check/known, skip-with-message
   for lgbox/, deploy/) + .gitignore additions.
4. Run bootstrap under capture.py; run make check + check_known.sh + import check; fix.
5. NOTES.md, commit, merge master.
