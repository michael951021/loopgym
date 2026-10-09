# docker — Dockerfiles for lgbox + trainer (hadolint-linted)

## Issue
The AWS nodes and a k8s deployment need runnable images, but nothing exists under
deploy/docker/. The Rust grader and the GPU trainer have no image, so the Scale-out
scripts (aws, gpurun) have nothing to point at. The sandbox has no Docker daemon and no
GPU (CLAUDE.md rule 2/4), so the images cannot be built here — they must at least lint
clean and be correct-by-construction so Kevin's build does not waste GPU hours.

## Question
Do `deploy/docker/lgbox.Dockerfile` (multi-stage cargo build → distroless/cc, non-root)
and `deploy/docker/trainer.Dockerfile` (CUDA runtime base + a `.venv-gpu` venv) pass
hadolint with zero findings and encode the right runtime contracts? Refuted if
`.tools/bin/hadolint deploy/docker/*.Dockerfile` is non-zero, or the lgbox runtime stage
is not non-root + glibc-capable distroless, or the trainer has no CUDA base and a venv
not named `.venv-gpu` with the CUDA build of torch.

## Evidence
- Observed: lgbox/src/main.rs — `lgbox serve --port 7777 --workers 4 --queue 64
  --backend process`, binds 0.0.0.0:7777; process backend needs no firecracker rootfs.
- Observed: tools/versions.txt pins rust=1.99.0, uv=0.12.24; pyproject requires-python
  >=3.12 and pulls torch from the CPU index (the GPU image must install CUDA torch instead).
- Verified real, pinned base tags (web, 2026-10-09): rust:1.99.0-slim,
  gcr.io/distroless/cc-debian12:nonroot, nvidia/cuda:12.9.1-runtime-ubuntu24.04.
- Smallest check: `.tools/bin/hadolint deploy/docker/*.Dockerfile` -> 0 findings (log
  .agent-evidence/docker-makecheck2.log; also full `make check` exit 0, 69s).
- Fixed: Makefile hadolint `find` matched only `Dockerfile*`, which skipped `*.Dockerfile`;
  changed to `\( -name 'Dockerfile*' -o -name '*.Dockerfile' \)` so `make check` actually
  lints these files.
- Observed: loopgym/env/lgbox.py:72 http runner sends cmd=[sys.executable,"-m","pytest",...],
  so the process backend runs pytest INSIDE the box. A pure distroless/cc image has no
  interpreter -> it cannot grade via the process backend. Kept distroless/cc (task spec,
  minimal server image); documented in the Dockerfile comment.

## Done when
`.tools/bin/hadolint deploy/docker/*.Dockerfile` exits 0 (already wired into `make check`).
`.dockerignore` present at repo root. No build attempted (sandbox has no Docker).
Known-answer checks (observable in the files, since we cannot build): (1) every base tag is
real + pinned, not `:latest` (no DL3006); (2) lgbox final stage is distroless/cc + non-root
uid, binary copied from the cargo release stage; (3) trainer base is a CUDA runtime image
and the venv is created at `.venv-gpu` with a CUDA torch. Plausible range written before the
run: zero hadolint findings.
Documented limitation (also in the lgbox.Dockerfile comment): distroless/cc has no python, so
the process backend cannot grade in this image. Process-backend grading on AWS needs a
python-runtime base (next ticket); the firecracker backend grades in the VM rootfs instead.

## Stop when
Scope: deploy/docker/{lgbox,trainer}.Dockerfile + .dockerignore + this ticket only. No
Terraform, no k8s manifests, no `docker build`, no new Makefile targets. Two attempts without
a clean hadolint run → record the exact remaining findings in NOTES.md and stop/split.

## Plan
1. deploy/docker/lgbox.Dockerfile — 2 stages: rust build → distroless/cc nonroot.
2. deploy/docker/trainer.Dockerfile — CUDA runtime base, uv-managed CPython 3.12, `.venv-gpu`
   with CUDA torch + loopgym.
3. .dockerignore at repo root (keep .venv/.tools/ckpt/models out of the build context).
4. Run hadolint, fix findings to zero.
5. make check (hadolint line); NOTES + commit + merge master.
