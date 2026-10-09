# Task log

<!-- One entry per completed TODO task, appended by the agent. Never edited or condensed — for
     a rolling handoff of current state see NOTES.md instead. -->
## Bootstrap toolchain and make targets
- Goal: Bootstrap toolchain and make targets
- Approach: Pinned real latest versions from upstream release endpoints, wrote the toolchain scripts first, then debugged against the acceptance command: fixed a bind-mount .venv (clear contents, never rm), a 404 rust dist pin (1.99.0), ruff 0.16 default-rule drift (pinned E4/E7/E9/F, excluded read-only tests/known), and two format-fragile version probes.
- Blockers: shared .venv/.tools are bind mounts and rm fails with EBUSY; rust dist server lags its channel toml (1.100.0 404s) — both recorded in NOTES.md gotchas for later tickets.
## Model downloads
- Goal: Model downloads
- Approach: acceptance-first (info failed on missing module, then passed on real downloads); params/dtype read from safetensors headers without loading weights; unit tests pin tiny_config and the header reader
- Blockers: none — one spec subtlety (151936 = config vocab, tokenizer length is 151643) resolved by checking config.json and noted in NOTES.md
## Pinned third-party sources
- Goal: Pinned third-party sources
- Approach: curl PyPI JSON per package, python3 stdlib picks latest non-prerelease + wheel-over-sdist, unzip/tar into .tools/src, sed/append pin; verified with the two rg checks, PyPI sha256 digest, idempotent re-run, make check
- Blockers: none
