# Notes

## Gotchas
- Bind mounts to Kevin's host (never `rm DIR`): /work, .git, /home/agent, .venv,
  .tools, data, models, ckpt, runs.
- ruff I/EXE pinned, whole tree; known/ I001/F401. .venv: no pip (uv).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.
- make check: pytest ignores tests/known + tests/hidden (hidden checks are
  worktree-scoped; fail collection in the shared tree, pre-existing).

## Scheduling
- `(human-only[: P])`: tools/sched_guard.py --line <id> (0 ok/1 skip/2 unknown); tagged held; hostgate DONE (030ad80).

## Cleanclone (make check in a fresh clone)
- `check: bootstrap` (idempotent); bootstrap: `mkdir -p .venv` + fetch_src.sh
  (pins from versions.txt; unpinned -> PyPI latest).
- Skip convention: tests reading gitignored host assets (models/, data/,
  .tools/src) skipif-when-absent; module-level reads break collection -> guard.
  Evidence: .agent-evidence/cleanclone-fix-{1..6}.log.
- draftlab extra += pandas+pyarrow (hidden check does to_parquet); uv.lock was
  stale vs extras, sync rewrote it.

## Eval / Models / Env / Kernel
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar resume; lgsource = one
  ```diff block (ref = git diff bad..good, empty patch -> 0.0, t=120).
- `check_results.py`: PENDING != FAIL; out-of-range only via range-author ruling
  (base05 LANDED 64ecd1b).
- `tiny_config` 2L/d64/V512. GOTCHA: cfg vocab 151936 vs tok 151643.
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup
  SIGKILL; `lgbox.py` http iff LOOPGYM_LGBOX_URL; `loopgym.py` ```path blocks.
- `token_logprobs(h,w,t,impl,chunk)`: chunked bwd softmax (no [T,V]); torch, cpp,
  triton (TRITON_INTERPRET=1), cuda host-only compile.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known. `batch.py` grpo_step:
  micro-batch accumulation; logp = base Qwen2Model + head weight + chunked kernel
  (never [rows,T,V]); ref=disable_adapter; kl==0 -> ref skipped; FSDP2/ZeRO-3 via
  engine arg; loss_factor scales backward only.
- `rollout.py` HFBackend: left pad, fp32 scores, row-chunked generate, per-chunk
  seed. `rollout_vllm.py` (host): bf16 enable_lora r16; build_llm() lazy (vLLM
  never imported in sandbox); check_vllm_hf fp32 mean|Δ|<0.02 nats GATED.
- test_train_dist: 2 ranks x b == single 2b (1e-5/1e-6) PASS fsdp+deepspeed.
  Toy gates PASS: banana 0.148 -> 1.000; null slope CI contains 0.
- `dist.py` FSDP2: wrap root = Qwen2ForCausalLM (children first) so generate/fwd/
  save() reach a coordinating root.
- `ds.py` ZeRO-3: pinned 0.19.7 aliases `zero` as attribute, NOT submodule ->
  `import deepspeed` + `deepspeed.zero.X` (rule 6). save(): optimizer gather is
  a collective -> every rank joins, rank0 persists.
- grpo05 (0.5B exercism, host 3090): micro_batch=2 (max T 3859; 28.7 GB < 39).
  Analysis NULL: paired val Δ CI ∋ 0, policy barely moved, 67.5% zero-adv groups.
  Step-0 loss 0.083 vs "<0.05 (β=0)" = length-weighted token-mean (cfg β=0.02).

## GPU / AWS (agents dry-run only; rule 4)
- scripts/gpu_run.sh: 2-GPU preflight, --stop-agents, .venv-gpu (uv pinned),
  known tests on CUDA, check_vllm_hf gate, 1.5B GRPO (grpo15_gpu.toml: vLLM GPU0,
  trainer GPU1, impl=triton), 3 evals -> results/gpu/. --dry-run 0/1; flag 2.
  GOTCHA: c10d rdzv HANGS in sandbox -> preflight needs getent hosts $(hostname).
- deploy/aws: no NAT/LB/EIP/S3; worker 10.0.0.10/11; shutdown-first user_data;
  spot terminate-on-interruption; estimate <= budget <= $10 (2x3h@$0.30 -> $2.02);
  aws_sweep.sh tag-based destroy; no aws CLI in sandbox -> `STUB_AWS`; tf -chdir=.

## Data / Report
- `freeze.py`: v0 = 26 verified, train 13/test 13, `--check` exit 1 on drift.
  hidden checks: synthetic fixtures, temp root, cross-check vs results/*.json.
- REPORT.md + README + check_report.py: every claimed number recomputes from
  results/ (perturb/drop -> exit 1).
