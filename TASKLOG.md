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
## Log-prob bench script
- Goal: Log-prob bench script
- Approach: Reused the f64 reference/tolerance from tests/known, probed torch profiler output to find a reliable CPU peak (function_events self sums were broken; [memory] event cumulative Total Allocated max works), benched triton-under-interpreter at T=1 with a flag since interpreter time is overhead, not kernel performance
- Blockers: torch 2.14 kineto function_events memory accounting returned bogus peaks (negative parent self values) — worked around by parsing the exported chrome trace's cumulative allocations
## Log-prob bench run
- Goal: Log-prob bench run
- Approach: compute job benchlp ran OMP_NUM_THREADS=8 .venv/bin/python tools/bench_logprob.py (exit 0)
- Blockers: none
## CUDA kernel source, compile-checked
- Goal: CUDA kernel source, compile-checked
- Approach: acceptance-first (pre-run fail logged); .venv-nvcc from pip CUDA 13.2.86 wheels (cu12 wheels ship no nvcc driver; 12.9 cannot compile on gcc15/glibc2.43); pure-CUDA fwd/bwd mirroring csrc/logprob_cpu.cpp math; torch binding + cuda_ext.py register "cuda" only where torch.cuda.is_available() (host); binding g++ -c bonus against torch headers
- Blockers: 13.x nvcc resolves the toolkit root from argv[0] (needs <root>/bin/nvcc wrapper); nvidia-nvvm must match ptxas version or PTX is rejected
## Eval CLI
- Goal: Eval CLI
- Approach: (no summary given)
- Blockers: none
## Baseline 0.5B on Exercism val
- Goal: Baseline 0.5B on Exercism val
- Approach: compute job base05 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.eval.run --model models/qwen2.5-coder-0.5b-instruct --source exercism --split val --n 4 --out results/eval/base05_val.json (exit 0)
- Blockers: none
## Baseline 1.5B on Exercism val
- Goal: Baseline 1.5B on Exercism val
- Approach: compute job base15 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.eval.run --model models/qwen2.5-coder-1.5b-instruct --source exercism --split val --n 4 --out results/eval/base15_val.json (exit 0)
- Blockers: none
## Toy banana run
- Goal: Toy banana run
- Approach: compute job toybanana ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.rl.train --config configs/toy_banana.toml --resume-if-any (exit 0)
- Blockers: none
## Toy null run
- Goal: Toy null run
- Approach: compute job toynull ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.rl.train --config configs/toy_null.toml --resume-if-any (exit 0)
- Blockers: none
## Re-run toy banana, berry prompt
- Goal: Re-run toy banana, berry prompt
- Approach: compute job toybanana2 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.rl.train --config configs/toy_banana.toml --resume-if-any (exit 0)
- Blockers: none
## Re-run toy null, berry prompt
- Goal: Re-run toy null, berry prompt
- Approach: compute job toynull2 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.rl.train --config configs/toy_null.toml --resume-if-any (exit 0)
- Blockers: none
## Re-run toy null, fresh
- Goal: Re-run toy null, fresh
- Approach: compute job toynull3 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.rl.train --config configs/toy_null.toml --resume-if-any (exit 0)
- Blockers: none
## Re-run toy banana, fresh
- Goal: Re-run toy banana, fresh
- Approach: compute job toybanana3 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.rl.train --config configs/toy_banana.toml --resume-if-any (exit 0)
## Check toy known answers
- Goal: Check toy known answers
- Approach: Ran tools/toy_check.py on the landed toybanana3/toynull3 metrics, generated toy.json from the same load_pairs/window/slope-bootstrap functions, verified via check_results and the make-check test scope
- Blockers: none
## Freeze loopgym v0
- Goal: Freeze loopgym v0
- Approach: reuse load_tasks()/verified records as the source of truth, add a freeze module with build/--check, pin the split rule to a published SHA-1 vector, run the acceptance chain end-to-end
- Blockers: none
## kevin: stop dispatching human-only lines
- Goal: kevin: stop dispatching human-only lines
- Approach: Wrote acceptance test first (failed as expected), implemented the stdlib guard + tag, tagged TODO.md:133, documented the contract, verified on the real TODO.md and full suite.
- Blockers: dispatcher code is host-side only, so its actual wiring to the gate is Kevin's one-line change — nothing in the sandbox to call it from.
## kevin: base05 range ruling
- Goal: kevin: base05 range ruling
- Approach: re-ran the acceptance checks on the real results/ + ruling; compared ruling byte-for-byte to the ticket's verbatim paste; recorded verdict in NOTES.md and the ticket
- Blockers: none — 40+ prior dispatches blocked on Kevin's host-side commit, which landed just before this run
## base05 floor: record ruling verdict
- Goal: base05 floor: record ruling verdict
- Approach: Ran the real checker on committed results with the landed ruling file, captured the log (rangerule-verdict.log), ran the 11 checker tests, updated ticket + NOTES verdict line, committed.
- Blockers: none
## Check baselines against ranges
- Goal: Check baselines against ranges
- Approach: ran the acceptance check on the committed results, verified the known answer (reference val pass@1 = 1.0), a negative control (tampered base15 → exit 1) and the 11 checker regression tests, then logged the verdict
- Blockers: none (prior attempts' blocker — the range-author ruling — had already landed)
## CPU GRPO config
- Goal: CPU GRPO config
- Approach: acceptance probe first, minimal config + trainer extension with regression tests, then the real 0.5B probe and a seq-mean control on identical completions (loss 0.0 exact) proving the out-of-range step-1 token-mean loss (0.0957) is a length-weighting artifact, not a trainer bug.
- Blockers: none
## Baselines on loopgym v0 (after loader)
- Goal: Baselines on loopgym v0 (after loader)
- Approach: compute job baselg2 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.eval.run --model models/qwen2.5-coder-0.5b-instruct --source loopgym --split test --n 4 --out results/eval/base05_loopgym.json && .venv/bin/python -m loopgym.eval.run --model models/qwen2.5-coder-1.5b-instruct --source loopgym --split test --n 4 --out results/eval/base15_loopgym.json (exit 0)
- Blockers: none
## GRPO step memory fix
- Goal: GRPO step memory fix
- Approach: This run continued an interrupted one whose code fix was already committed (dba1534). I verified the fix against the ticket contract (state/tickets/grpomem.md), re-ran tests/test_grpomem.py (3/3) and the full suite, confirmed the 2-step measurement log, applied the mem bump per the ticket's x1.3 rule, recorded the peak and condensed NOTES.md (4030 bytes < 4 KB), added the plan_lint why: lines under the open TODO tasks, committed (75022cd), and merged master (already up to date).
- Blockers: none.
## AWS spend guardrails
- Goal: overspend-Impossible caps for the AWS scaling study (ticket awsguard)
- Approach: test-first (stub aws on PATH, no creds — rule 4). deploy/aws/terraform with variable validation (nodes in {1,2}, g4dn.xlarge only, max_price <= 0.60, max_hours 1..3 whole, budget <= 10), one-time spot terminate-on-interruption, shutdown-first user_data, IMDSv2, gp3 delete_on_termination, one-time scheduler at(+15m) terminate with its own role, idle-CPU alarm -> arn:aws:automate terminate, budget + 100%-actual deny-on-launch-role action. scripts/aws_sweep.sh is tag-based and state-independent (ec2 describe-* + tag-filters + resourcegroupstaggingapi catch-all; --kill = cancel spot, terminate, del alarm/schedule, bounded wait-terminated, del volumes/SGs, re-check). scripts/aws_launch.sh preflight (also --dry-run): sweep clean, caller is the launch role, G+VT spot quota <= 8 vCPU (paginated), estimate nodes*(max_hours+0.25)*(max_price+EBS+hIPv4) <= budget; EXIT/INT/TERM trap destroy + sweep, exit 1 printing the --kill command on leftovers.
- Blockers: none. Fixes found while wiring: terraform requires -chdir=DIR (equals form); my user_data test filtered the shebang then asserted it first (self-contradiction, fixed the test); README wrapped the quota name across lines so the substring test missed it; ruff E741 on loop var l in the test; Makefile check gate pointed at deploy/aws/*.tf so it would have silently skipped validate once the ticket's deploy/aws/terraform layout landed — re-pointed the gate.
## AWS scaling study under $10
- Goal: AWS scaling study under $10
- Approach: extended the one-template launch.py generator with a one-aws topology and --steps/--out overrides; aws_launch.sh reuses awsguard's preflight verbatim, pins node0 to 10.0.0.10 (c10d master), and tests everything through a stubbed aws CLI extracted from test_aws_guard.py so no real AWS call is possible in the sandbox.
- Blockers: shellcheck false positive — this pinned version drops the trap's reference to cleanup() when a top-level exit follows the traps (SC2329), so the disable is inline with the reason documented.
## Fix grpo05: OOM (ran out of memory)
- Goal: Fix grpo05: OOM (ran out of memory)
- Approach: Prior attempt's fix was committed but unverified (session killed at step 2, no ckpt). Relaunched the resume-safe scratch acceptance run in the background with a tree-Pss monitor, polled it to completion, confirmed the fails-first evidence (mb=8 repro PEAKMEM 54.9 GB), raised the (mem:) tag, ran make check (exit 0: ruff, 274 passed/1 skipped, cargo, tf validate), updated NOTES.md and the ticket, committed (3bff2af), merged master (already up to date).
- Blockers: Session teardown killed the first acceptance attempt mid-run (recovered: --resume-if-any restarted fresh, deterministic seeds made steps 1-2 bit-identical); also pytest tests/ without make check's ignores fails collection on 21 worktree-scoped tests/hidden/ checks (pre-existing on master, noted in NOTES.md).
## Fix grpo05: OOM (ran out of memory)
- Goal: Fix grpo05: OOM (ran out of memory)
- Approach: reproduced the task sampler (Random(1337)) to bound every step's row length without running the job, cross-checked against the known beer-song prompt length, then verified the resume/discard mechanics in train.py against the existing test suite.
- Blockers: none — the prior session was killed mid floor-run; that run turned out to be unnecessary once the width probe closed the step-40 gap.
## CPU GRPO run
- Goal: CPU GRPO run
- Approach: compute job grpo05 ran OMP_NUM_THREADS=12 .venv/bin/python -m loopgym.rl.train --config configs/grpo05_exercism.toml --resume-if-any (exit 0)
- Blockers: none
## Eval GRPO checkpoint
- Goal: Eval GRPO checkpoint
- Approach: compute job evalgrpo05 ran OMP_NUM_THREADS=8 .venv/bin/python -m loopgym.eval.run --model models/qwen2.5-coder-0.5b-instruct --adapter ckpt/grpo05_exercism --source exercism --split val --n 4 --out results/eval/grpo05_val.json (exit 0)
- Blockers: none
## Analyse the CPU GRPO run
- Goal: Analyse the CPU GRPO run
- Approach: ticket first (state/tickets/grpo05ok.md) with pre-run ranges; extended check_results.py with the grpo05 section (slope CI via one-index case resampling — caught a decoupled-resample bug that faked a null slope CI by comparing tool vs an independent probe; paired Δ via loopgym.eval.stats.bootstrap_ci, hand-verified +0.0347 = (3×1/16 − 1/4)/9); new tools/grpo05_hackcheck.py regenerates + re-grades the first 20 val completions with mechanical flags; 4 new tests (null/gain-gate/regression/missing, hand-checked deltas); make check green (278 passed; .agent-evidence/grpo05ok-makecheck.log). Committed 12b4917; git merge master already up to date. Out-of-ticket observation recorded in NOTES.md: step-0 loss 0.083 vs the CLAUDE.md "<0.05 at step 0 (β=0)" range — length-weighted token-mean with β=0.02 (k3≈0 at step 0); not a check this ticket pins, flagged for review, not explained away.
- Blockers: none.
## REPORT and README
- Goal: REPORT and README
- Approach: Built the checker's claim registry from results/ and runs/grpo05_exercism, wrote the report from its --emit canonical strings, then verified exit 0 on the committed text, exit 1 on three tamper variants, and ruff + pytest green
- Blockers: none
## Fix cleanclone: non-zero exit (exit-code)
- Goal: Fix cleanclone: non-zero exit (exit-code)
- Approach: three clone repros (25→1→0 failures) each surfaced new evidence: missing .venv → bootstrap fix, absent host assets → skipif convention, missing pandas/pyarrow → extra; verified the exact job command exit 0 (194s, fresh log, no pipe) plus shared-tree make check exit 0 (330s).
- Blockers: a false-green clone run — capture.py exits 2 on an existing --log path and my | tail pipe masked that rc; fixed by fresh log names and no pipe, and recorded in the ticket.
