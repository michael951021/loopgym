# syntax=docker/dockerfile:1
# lgbox: the Rust grading server (untrusted-code sandbox).
# Multi-stage: cargo release build -> distroless non-root runtime.
# Build from the repo root:  docker build -f deploy/docker/lgbox.Dockerfile .

# --- build: compile the release binary ----------------------------------------
FROM rust:1.99.0-slim AS build
WORKDIR /src
# manifest + lock first so the dependency-fetch layer is cached across source edits
COPY lgbox/Cargo.toml lgbox/Cargo.lock ./
COPY lgbox/src ./src
RUN cargo build --release --locked

# --- runtime: minimal glibc (cc) base, runs as non-root ------------------------
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/lgbox /usr/local/bin/lgbox
# distroless :nonroot = uid/gid 65532; numeric so it resolves without a /etc/passwd entry
USER 65532:65532
EXPOSE 7777
# Minimal server image (0.0.0.0:7777 > 1024, so non-root can bind). The process backend
# executes the grader's `python -m pytest` inside this container, which needs a Python
# runtime that pure distroless lacks -- for process-backend grading on AWS, supply a python
# runtime base (next ticket); the firecracker backend grades in the VM rootfs instead.
ENTRYPOINT ["/usr/local/bin/lgbox", "serve", "--backend", "process"]
CMD ["--port", "7777", "--workers", "4", "--queue", "64"]
