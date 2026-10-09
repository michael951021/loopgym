# syntax=docker/dockerfile:1
# Trainer: CUDA runtime base + a uv-managed CPython 3.12 venv (.venv-gpu) carrying the
# CUDA build of torch + the loopgym package. Built for GPU nodes (AWS/k8s); no GPU here.
# Build from the repo root:  docker build -f deploy/docker/trainer.Dockerfile .

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ARG UV_VERSION=0.12.24
WORKDIR /app

# fetch uv (static binary) from its pinned GitHub release; uv owns the venv afterwards.
# ca-certificates is a rolling trust store and curl is only a one-shot fetcher; pinning
# either to an exact version breaks the build on the next archive update, so take the
# latest from the (itself-pinned) base image.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL --retry 3 \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/uv.tgz \
    && tar -xzf /tmp/uv.tgz -C /tmp \
    && install -m0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv.tgz /tmp/uv-x86_64-unknown-linux-gnu

# package manifest + sources, so `uv pip install .` has something to install.
COPY pyproject.toml ./pyproject.toml
COPY loopgym ./loopgym
COPY csrc ./csrc
COPY configs ./configs

# .venv-gpu (repo-root convention, CLAUDE.md / gpu_run.sh): install the CUDA build of
# torch first (the pyproject CPU index is sandbox-only), then the loopgym package + deps.
RUN uv venv -p 3.12 /app/.venv-gpu \
    && uv pip install --python /app/.venv-gpu/bin/python --torch-backend=cuda torch \
    && uv pip install --python /app/.venv-gpu/bin/python .

ENV PATH="/app/.venv-gpu/bin:$PATH" \
    TORCH_CUDA_ARCH_LIST="8.6"

CMD ["python", "-m", "loopgym.rl.train"]
