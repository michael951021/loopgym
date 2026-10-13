# Notes

## Gotchas
- Bind mounts to Kevin's host (never `rm DIR`): /work, .git, /home/agent, .venv,
  .tools, data, models, ckpt, runs.
- ruff I/EXE pinned, whole tree; known/ I001/F401. .venv: no pip (uv).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.
- make check: pytest ignores tests/known + tests/hidden; hidden checks are
  worktree-scoped (they fail collection in the shared tree; pre-existing).

## Scheduling
- `(human-only[: P])`: tools/sched_guard.py --line <id> (0 ok/1 skip/2 unknown); tagged held; hostgate DONE (030ad80).

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar resume. lgsource: one
  ```diff block; ref = git diff bad..good; empty patch -> 0.0; frozen v0/checks, t=120;
  e2e pass@1 0.0 (range 0-0.25): lgsource-e2e2.log.
- `check_results.py`: PENDING != FAIL; out-of-range only via range-author ruling; base05
  ruling LANDED (64ecd1b); basecheck 5/5 PASS.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  `lgbox.py` http iff LOOPGYM_LGBOX_URL; `splits.py` sha1%100 101/9/27 (137/142, val 9);
  `loopgym.py`: ```path blocks; grade = reply on bad worktree (outside task.files -> 0).

## Kernel
- `token_logprobs(h,w,t,impl,chunk)->[T]`: chunked bwd softmax (no [T,V]); torch,cpp,
  triton (TRITON_INTERPRET=1, C<=1024),cuda host-only. `_host_shim`: CPU test tensors ->
  cuda on a CUDA host (no-op here); triton/cuda untested on real GPU (gpu_run.sh step 5).

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: grpo_step (logp=token_logprobs; old=detach; ref=disable_adapter);
  step-0 β=0 ~0; kernel unbatched -> flatten.
- `rollout.py` HFBackend (left pad, output_scores fp32; T<=0 greedy); top_k=0, rep=1.0.
- `rollout_vllm.py` (host): bf16 enable_lora r16; sync_lora id+1/step;
  logprobs=0 -> sampler_logp; max_tokens explicit; 0.02-nat; build_llm() lazy
  (tests monkeypatch it; vLLM never imported in sandbox).
- gpurun: scripts/gpu_run.sh = Kevin's one-command GPU window: preflight nvidia-smi
  (2 GPU), stop agents only w/ --stop-agents, .venv-gpu via .tools/bin/uv pinned to
  tools/versions.txt (+.gpu-stamp), assert IMPLS superset {cuda,triton},
  pytest tests/known on CUDA, check_vllm_hf gate, GRPO 1.5B
  (configs/grpo15_gpu.toml: backend=vllm, trainer_device=cuda:1, impl=triton,
  B4 G8 384tok lr2e-5 beta0.02 100 steps), 3 evals w/ final adapter -> results/gpu/.
  --dry-run: plan + 19-file check, exit 0/1; unknown flag exit 2; shellcheck clean.
  --two-node-local (twonode): 0.5B toy rehearsal, 2 torchrun "nodes" on this host
  (GPU0+GPU1, 1 rank each); cmds from loopgym.rl.launch cmds, one template for
  single/two-local/two-aws differing only in gpu/nnodes/node_rank/nproc/
  --rdzv-endpoint (c10d 127.0.0.1:29500 local), fsdp+deepspeed x20 steps;
  report -> results/gpu/scale_local_{dist}.json (per-step gen_tok_s/step_s + PEAKMEM).
  GOTCHA: c10d rdzv HANGS in sandbox (C getaddrinfo on unresolvable hostname
  "sandbox"; Python shim no-op) -> preflight requires getent hosts $(hostname).
  test_train_dist: 2 ranks x b == single 2b (LoRA 1e-5, metrics 1e-6) PASS for BOTH
  fsdp+deepspeed in sandbox (23s) - no host-only skip needed.
  Smaller REQUIRED_FILES_TWOLOCAL (toy: no exercism/loopgym/1.5B/cuda-kernels);
  ensure_venv_gpu factored so full run + two-node share the pinned .venv-gpu build.
  aws (study): network.tf VPC/IGW/subnet/route/SG (SSH 22 from var.admin_cidr
  ONLY + self all-protocol)/cluster PG/keypair; no NAT/LB/EIP/S3. Worker pinned
  10.0.0.10/11 (node0 = c10d master); user_data = shutdown FIRST (re-armed
  around the driver reboot), then idempotent clone + .venv-gpu (uv, pins from
  tools/versions.txt) + 0.5B model -> /work/.aws-ready. aws_launch.sh
  --dry-run = 3-config plan (single one-aws+fsdp, fsdp/deepspeed two-aws, 50
  steps, out runs/gpu/scale_<name>) + full preflight + estimate $2.02 <= $10
  (known answer: 2*3.25*(0.30+50*0.08/730+0.005)); real = apply -> wait
  .aws-ready -> 3 configs back to back (rank-0 rc decides; continue on fail)
  -> scp metrics.jsonl + `launch report` results/gpu/scale_aws_<name>.json per
  config (debuggable post-destroy) -> destroy+sweep trap on ANY exit.
  launch.py: +one-aws topology, --steps/--out overrides (train.py load_cfg
  coerces both). GOTCHA: sandbox has no aws CLI - stub via `import
  test_aws_guard; STUB_AWS` (regex on the source corrupts \n escapes); this
  shellcheck SC2329 false-positives a trap-invoked fn when a top-level exit
  follows the traps (disabled, reason inline). Evidence: .agent-evidence/aws-
  dryrun-clean.log, aws-makecheck.log.
- train.py backend=hf|vllm (in RESUME_CHECK, w/ trainer_device): vllm ->
  engine_args + rv.build_llm; sync_vllm(s, dir) saves adapter + sync_lora(f"step_{s}");
  restore re-syncs so the live engine uses the ckpt adapter; first generation is
  unlora (train before sync). check_vllm_hf.py: greedy n=1 both backends, common
  prefix, fp32 mean|Δ|<0.02 nats GATED (full-prefix too), bf16 reported; teardown =
  engine_core.shutdown + gc + empty_cache (0.35 mem util, 2048 ctx).
- `train.py`: toml+--k v; sample->gen->grade->grpo_step->jsonl->ckpt
  <out>^2/ckpt/<name>; --dist+--dry-run; exact resume; exit 1 on orphan ckpt;
  from_pretrained freezes ALL -> requires_grad lora_.
- `train.py`: max_difficulty caps train sampler (scale 1-9; 67 of 101 train <=3);
  eval_n>0 = that many val tasks/eval, stateless draw seed+off+s, row has tasks[].
  configs/grpo05_exercism.toml (0.5B r16, diff<=3, B4 G8, 384tok, lr2e-5, b0.02,
  40 steps, eval 20/n2, save 5, micro_batch 4). Probe: 1 step 524s, reward 0.25, tok-mean loss
  0.0957 vs seq-mean 0.0 (same completions) = length artifact, not a bug.
  tiny ctx 512 < exercism prompts -> tiny+exercism gen untestable.
- `grpo.py` group_advantages/kl_k3/policy_loss(denom=None); 7/7 known.
- `batch.py`: grpo_step = grad accumulation over micro_batch row chunks (token-mean:
  global mask denom; seq-mean: scale rows_c/n); logp_of = base Qwen2Model (no lm_head)
  + head weight + chunked kernel per row chunk (never [rows,T,V]); ref=disable_adapter.
  Merged w/ twonode: logp_of(model,batch,cfg,engine=None) (FSDP2 full_tensor() once;
  ZeRO-3 per-chunk GatheredParameters, lm_head never traversed by base fwd);
  grpo_step(...,loss_factor=1.0,engine=None): factor scales backward only; engine path
  = engine.backward/step + get_global_grad_norm; kl==0 -> ref skipped.
- `rollout.py` HFBackend(micro_batch=8): left pad, output_scores fp32, generate in row
  chunks w/ per-chunk seed (seed+7919*chunk); scores freed per chunk; top_k=0, rep=1.0.
- `rollout_vllm.py` (host): bf16 enable_lora r16; sync_lora id+1/step; 0.02-nat gate;
  build_llm() lazy (vLLM never imported in sandbox).
- grpomem DONE: grpo05 OOM'd at 24 G (full-vocab logits [32,~800,151936] fp32 + dlogits
  + rollout scores). micro_batch=8 key (TrainCfg, RESUME_CHECK, toml). 2-step run exit 0,
  peak RSS 25.1 GiB: .agent-evidence/grpomem-2step.log. grpo05 (mem: 24)->33 = peak GiB
  x1.3 rounded up. tests/test_grpomem.py 3/3: mb=1 grads == big batch 1e-5.
- fix-grpo05 DONE (attempt 3, 10-13): mb=8 54.9 GB, mb=4 41.3 GB at step 32
  (beer-song, T<=3859) > 39 cap -> micro_batch=2 in grpo05_exercism.toml.
  Width probe (deterministic sampler seed 1337, all 40 steps;
  grpo05fix_width_probe.py): max T = 3859 @ step 32, next 2465; beer prompt
  3475 reproduced = known answer. mb=2 beer probe (that row, OMP 8): parent
  28.71 GB < 39 -> (mem: 39) stays; ~11 h < 24 h timeout. Floor run (1-13)
  superseded: monotone in T + worst row measured. Retry: no ckpt (pre-mb2
  aside) -> --resume-if-any discards metrics, fresh from 1 (train.py:517-521,
  tested); dispatcher merges master (mb=2) into job branch (precedent 12f92b7).
  Evidence: grpo05fix-{repro,mb4,tail,beer}{,-mon}.log.
- gpurun: scripts/gpu_run.sh = Kevin's GPU window: 2 GPU preflight, --stop-agents,
  .venv-gpu, known tests on CUDA, check_vllm_hf <0.02-nats, 1.5B GRPO, 3 evals -> results/gpu/.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors, grads AVG, no 1/W.
  wrap_fsdp shards Qwen2ForCausalLM as the ROOT (children embed/layers/norm/lm_head first)
  so generate + train fwd + save() gather all reach a coordinating root.
- `ds.py` ZeRO-3: ds_zero3.json bf16 stage3; micro=b*group; CPU compatible_ops[deepspeed_shm_comm]=False.
  Pinned 0.19.7 aliases `zero` as an attribute (`from .runtime import zero`), NOT a
  registered submodule -> `from deepspeed.zero import X` fails; use `import deepspeed` +
  `deepspeed.zero.X` (rule 6 caught it; the test's DS_SKIP masked it as a sandbox gap).
  (pre-merge note: full-model fwd's post-hook re-released lm_head -> re-gather after fwd;
  the merged base-fwd design never touches lm_head, so plain per-chunk gather suffices.)
  save(): optimizer gather is a collective -> every rank joins,
  only rank0 persists; _dist_adapter_tensors names keys via get_peft_model_state_dict so
  the ckpt == single-process save_pretrained (drops the `default` adapter level).
  test_train_dist.py: 2 ranks x b == single 2b (LoRA 1e-5, metrics 1e-6) fsdp+deepspeed.
- Toy gates PASS: toybanana3/toynull3; banana 0.148 -> 1.000; null slope CI contains 0.
  `ds.py` ZeRO-3 bf16 stage3; micro=b*group; CPU compatible_ops[deepspeed_shm_comm]=False.
- Toy gates PASS: banana 0.148 -> 1.000; null slope CI contains 0.
- `tools/fetch_src.sh`: vllm/deepspeed/trl -> .tools/src/ (rule 6).

## AWS
- awsguard: deploy/aws/{terraform,iam,README} + scripts/aws_{sweep,launch}.sh;
  tests/test_aws_guard.py (15) stubs aws on PATH; sandbox never applies (rule 4).
- sweep: tag-based (Project=loopgym), state-independent: ec2 describe-* +
  cloudwatch/scheduler --tag-filters + resourcegroupstaggingapi catch-all;
  --kill = cancel spot -> terminate -> del alarm/schedule -> wait terminated
  (bounded 20x10s) -> del volumes/SGs -> re-check, fail loudly on leftovers.
- launch preflight (also under --dry-run): sweep clean (no double launch),
  caller Arn has role/loopgym-launch (short-lived creds only), G+VT spot quota
  <= 8 vCPU (paginated), estimate nodes*(max_hours+0.25)*(max_price+50*0.08/730
  +0.005) <= budget_usd <= $10. 2x3h@$0.30 -> $2.02; @$5 -> $32.57 refused.
  EXIT/INT/TERM trap: destroy (if state) -> sweep --check -> exit 1 + print
  `aws_sweep.sh --kill` if anything left. GOTCHA: terraform needs `-chdir=DIR`
  (equals form); tf state gitignored (*.tfstate). Makefile check gate ->
  deploy/aws/terraform (old deploy/aws/*.tf test would silently skip).
- .tf: nodes in {1,2}, max_hours 1..3 integer, instance_type g4dn.xlarge only,
  max_price <= 0.60, budget <= 10 (variable validation); shutdown-first
  user_data; one-time spot terminate-on-interruption; IMDSv2; gp3 delete_on_
  termination; scheduler at(+15m) terminate (own role); idle CPU<5% 6x300s ->
  arn:aws:automate terminate; budgets 50/80 actual + 100 forecast e-mail, 100
  actual -> deny-RunInstances on launch role (LAGGING backstop).

## Data
- `freeze.py`: v0 = 26 verified (draftlab 22/harness 4), train 13/test 13; `--check`
  exit 1 on drift; public draftlab/harness only. `data/verify.py`: 3x/side pristine
  worktrees -> verified.jsonl; tests=0 -> error.
- hidden checks: synthetic fixtures, temp root; assert completion claim; cross-check
  vs results/*.json (lazy discovery: absent at bad).
