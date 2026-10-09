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
## Exercism tasks and local grader
- Goal: Exercism tasks and local grader
- Approach: read the known test for the exact API, verified the exercism layout via .meta/config.json (paasio has 2 test files), wrote the runner with start_new_session + SIGKILL process group and XML-based reward, then ran the strict known tests first (ImportError), implemented, and added unit tests for parse/prompt.
- Blockers: none
## Exercise audit and splits
- Goal: Exercise audit and splits
- Approach: wrote failing acceptance first, then audit() with ThreadPoolExecutor over load_exercise/grade, pure sha1-bucket split function, and tests/test_splits.py (published sha1 value, bucket/disjointness checks, 3-fixture mini-audit graded with real pytest).
- Blockers: none (one self-inflicted fixture bug — "bad ref" x+2 actually passed f(2)==4 — caught by the test and fixed).
## lgbox server with process backend
- Goal: lgbox server with process backend
- Approach: std::process::Command in spawn_blocking with pre_exec setpgid+setrlimit, try_wait vs deadline then killpg(SIGKILL); mpsc(Q) queue with semaphore-capped workers; raw-HTTP integration tests on 127.0.0.1:0 covering echo, sleep-30 timeout survivors, fork bomb, 2 GB alloc, 400s, 503 count.
- Blockers: transient shared-machine load pushed uid-1000 task count past RLIMIT_NPROC=64 so sh's fork hit EAGAIN; fixed by execing /bin/sleep directly and documenting the interaction in NOTES.md.
## lgbox metrics
- Goal: lgbox metrics
- Approach: lgbox metrics ticket (id: lgboxmet) complete.
- Blockers: none
## Python client and http runner
- Goal: Python client and http runner
- Approach: (no summary given)
- Blockers: none
## lgbox throughput bench (job)
- Goal: lgbox throughput bench (job)
- Approach: script auto-spawns release lgbox server on a free port, sets LOOPGYM_LGBOX_URL so the real http runner path is graded, ThreadPoolExecutor per cell, nearest-rank p50/p95, reward-1.0 + c1-overhead<50ms checks in the JSON
- Blockers: none
## lgbox bench run
- Goal: lgbox bench run
- Approach: compute job lgboxbench ran bash -c 'cd lgbox && PATH=/work/.tools/cargo/bin:$PATH cargo build --release && cd .. && .venv/bin/python tools/bench_lgbox.py' (exit 0)
- Blockers: none
## HF rollout backend
- Goal: HF rollout backend
- Approach: read transformers 5.19 _sample source to confirm scores are post-warper fp32 logits, then built the backend + 3-test file on the real 0.5B and closed the 0.4-nat gap by disabling the model's shipped top_k/repetition-penalty defaults
- Blockers: the instruct model's generation_config.json silently applied top_k=20 and repetition_penalty=1.05, giving a 0.4-nat log-prob gap until overridden
## One GRPO step from completions
- Goal: One GRPO step from completions
- Approach: wrote failing tests first on tiny_config; verified mask against an independent log_softmax+gather, step-0 loss −9.9e-9 < 1e-6, +adv logp rose +0.211 nats / −adv fell −0.105 after one step, kl>0 post-step proves adapter disabled in ref; fixed transformers 5.x hidden_states[-1] and unbatched-kernel flattening along the way.
- Blockers: none — the only real snag was my test's own advanced-indexing bug (slice+vector gives a cross-product, not a diagonal), which briefly looked like an implementation bug.
## FSDP2 path and equivalence test
- Goal: FSDP2 path and equivalence test
- Approach: Verified pinned torch 2.14.1 FSDP2 defaults to AVG reduce-scatter (per-rank per-token-mean loss → full-batch mean gradient, no 1/W scaling). Two bugs found and fixed during verification: (1) worker called rank() before the group existed (init happens inside wrap_fsdp) so both ranks trained on slice 0 (diff 2e-3 = 2×LR); (2) reference backward used the raw token-sum loss while ranks use the per-token mean — Adam's eps regime diverges up to ~lr/6 (observed 1.253e-4); reference now mirrors the ranks (two b-row forwards, mean loss) so only FSDP's reduction is under test → bit-exact. train.py --dist fsdp wiring left to w2 (owns train.py); dist.py exposes the hooks. Evidence: .agent-evidence/fsdp_equiv6.log, fsdp_make_check.log.
- Blockers: none.
## Toy known-answer configs
- Goal: Toy known-answer configs
- Approach: Cloned smoke.toml shape for the two configs, wrote the checker over metrics.jsonl train lines with OLS slope bootstrap, tested verdicts on synthetic pass/fail data, then ran the real 2-step CLI probe (46 s, exit 0, range 1-5 min written in the ticket before the run) and make check/known.
- Blockers: none — noted observation: step-1 banana reward 0.281 exceeds the 0.2 gate start (instruct model); flagged in NOTES.md and the commit for the 60-step run to confirm or rule out per rule 5.
## vLLM rollout backend with weight sync
- Goal: vLLM rollout backend with weight sync
- Approach: (no summary given)
- Blockers: none
## C++ CPU kernel forward
- Goal: C++ CPU kernel forward
- Approach: torch.utils.cpp_extension.load JIT into .tools/build (-O3 -fopenmp); a Python autograd.Function wrapping the C++ forward reuses the factored-out torch chunked backward as placeholder until cppbwd, keeping the full known suite (f64 gradcheck, real-model test) green at every commit point.
- Blockers: torch 2.14's cpp_extension.load hard-requires ninja on PATH (no make fallback); resolved by adding the ninja wheel to the venv dep set plus a PATH guard in the loader.
## C++ CPU kernel backward
- Goal: C++ CPU kernel backward
- Approach: templated per-chunk C++ backward mirroring the fwd's chunked structure; discriminating pre-state probe (no logprob_bwd symbol) + new regression test vs naive log-softmax autograd; gradcheck f64 as known answer
- Blockers: first regression run failed on a dropped sign (-gp written +gp); caught by the f64 1e-10 comparison, fixed in one line
