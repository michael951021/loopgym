# Ticket: Pinned third-party sources (id: vsrc, after: boot)

## Issue
Later tickets (vLLM rollouts, DeepSpeed/FSDP training) call fast-moving third-party APIs from
memory, which drifts. Rule 6 requires checking every called API against the pinned source in
`.tools/src/` with `rg`, not from memory. We have no local copy of vllm/deepspeed/trl source yet.

## Question
Can a single idempotent script `tools/fetch_src.sh` download the latest *stable* (non-prerelease)
release of vllm, deepspeed and trl from PyPI, unpack each into `.tools/src/<name>-<version>/` for
reading only, and record the exact versions in `tools/versions.txt` — such that the known symbols
`def load_weights` (vllm) and `class DeepSpeedConfig` (deepspeed) exist in the unpacked trees?
Refutation: either rg check prints nothing, or the tree is missing one of the three names.

## Evidence
- Starting point: `tools/versions.txt` (KEY=VERSION pins), `tools/bootstrap.sh` (style: set -euo
  pipefail, curl, pinned .tools).
- Check: run `bash tools/fetch_src.sh` (network: PyPI), then
  `rg -l "def load_weights" .tools/src/vllm-*/` and
  `rg -l "class DeepSpeedConfig" .tools/src/deepspeed-*/`.
- Log: `.agent-evidence/vsrc-fetch.log` (capture.py).
- Observed vs guess: versions chosen are what PyPI's JSON reports as latest non-prerelease at run
  time; which files contain the symbols is untested until the rg check runs.

## Done when
- `bash tools/fetch_src.sh` exits 0 (idempotent: re-run skips re-download).
- `rg -l "def load_weights" .tools/src/vllm-*/` prints ≥1 file.
- `rg -l "class DeepSpeedConfig" .tools/src/deepspeed-*/` prints ≥1 file.
- `tools/versions.txt` has `vllm=`, `deepspeed=`, `trl=` lines matching the unpacked dir suffixes.
- `make check` shellcheck passes on tools/fetch_src.sh.
- No import of the packages into `.venv` (read-only: unpacked tree only).
- Known-answer check: the recorded versions are real PyPI releases (URL in JSON resolves, file
  downloaded byte-for-byte), and the unpacked trees contain the two named symbols.

## Stop when
- PyPI unreachable (no network) → record blocker, stop.
- A package's latest stable has neither wheel nor sdist (impossible on PyPI) → record, stop.
- After two failed download/parse attempts without new evidence → record blocker, stop/split.
- Out of scope: installing anything, GPU code, other packages, modifying .venv.

## Plan
1. Write `tools/fetch_src.sh`: per name, curl PyPI JSON, pick latest non-prerelease, pick wheel
   (else sdist), curl it, unpack to `.tools/src/<name>-<version>/` (wheel=zip, sdist=tar), append
   `name=version` to `tools/versions.txt` (idempotent).
2. Run under capture.py; inspect log.
3. Run both rg acceptance checks.
4. `make check` (at least shellcheck + pytest/ruff unaffected).
5. Update NOTES.md, commit, merge master.
