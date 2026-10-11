# Notes

## Gotchas
- Bind mounts to Kevin's host (never `rm DIR`/find -delete): /work, .git,
  /home/agent, .venv, .tools, data, models, ckpt, runs; dispatcher is
  host-side.
- ruff I/EXE pinned, whole tree incl tests/hidden; known/ I001/F401. .venv: no pip (uv/ensurepip).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.

## Scheduling
- `(human-only[: P])`: tools/sched_guard.py --line <id> (0 ok/1 skip/2 unknown); tagged held.
- hostgate DONE (030ad80, kevin; line 137 [x]); humandispatch [x].
  Guard wired host-side; Kevin-only lines now hold until released.

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
  `eval/run.py`: --model/--adapter | --reference; sidecar resume. loopgym source
  (lgsource): reply = one ```diff unified-diff block (815-line median fix can't fit
  512 tokens as whole-file rewrite); ref = git diff bad..good -> 1.0 by construction,
  empty patch = bad state -> 0.0; frozen v0/checks, timeout 120. e2e 0.5B/2 tasks
  pass@1 0.0 (base range 0.0-0.25): .agent-evidence/lgsource-e2e2.log; smoke results/
  uncommitted (baselg2 supersedes).
- `tools/check_results.py`: baselines/benches/toy; PENDING != FAIL; out-of-range
  only via range-author ruling (malformed fails closed; rule 5).
  base05 ruling LANDED (64ecd1b, kevin, verbatim accept). basecheck DONE
  (atk13): `--only baselines` exit 0, 5/5 PASS, ruling cited verbatim;
  negative control exit 1; tests 11/11 (basecheck13-*.log).

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL.
  `lgbox.py`: RUNNERS["http"] iff LOOPGYM_LGBOX_URL.
  `splits.py`: sha1%100, 101/9/27,
  137/142 usable, val 9. `loopgym.py`: Task(prompt/parse/reference/stub)+grade over verified
  records; elided bad files (diff -U0 ±3, 4500-chr, <=6k tok); ```path blocks; grade = reply
  on bad worktree, pristine V._run_once (outside task.files -> 0); ref=1, stub=0, hidden=0.
  DiffTask (eval): ```diff reply, ref = git diff bad..good, frozen v0/checks.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]: chunked max/LSE bwd softmax (no
  [T,V]); torch,cpp,triton,cuda; cpp JIT (-O3 -fopenmp); triton iff CUDA or
  TRITON_INTERPRET=1, C<=1024; cuda host-only. bench_logprob.py -> bench_logprob.json, f64 1e-4.
  `_host_shim` wraps triton/cuda: known tests pass CPU tensors -> on a CUDA host they
  are moved .to("cuda") (differentiable, so f64 gradcheck still reaches the kernel);
  no-op in the sandbox. Host-only risk: triton/cuda on a real GPU untested here
  (gpu_run.sh step 5 surfaces it).

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: grpo_step (logp=token_logprobs; old=detach; ref=disable_adapter);
  step-0 β=0 ~0; no last_hidden_state in 5.x; kernel unbatched -> flatten.
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
  40 steps, eval 20/n2, save 5). Probe: 1 step 524s, reward 0.25, tok-mean loss
  0.0957 vs seq-mean 0.0 (same completions) = length artifact, not a bug.
  tiny ctx 512 < exercism prompts -> tiny+exercism gen untestable.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors, grads AVG, no 1/W.
- `ds.py` ZeRO-3: ds_zero3.json bf16 stage3; micro=b*group; CPU compatible_ops[deepspeed_shm_comm]=False.
- Toy gates PASS: toybanana3/toynull3; banana 0.148 -> 1.000; null slope CI contains 0.
- `tools/fetch_src.sh`: vllm/deepspeed/trl -> .tools/src/ (rule 6).

## Data
- `freeze.py`: v0 = 26 verified (draftlab 22/harness 4), sha1%100>=70
  train 13/test 13; data/loopgym/v0/ + results/loopgym_v0.json; `--check`
  exit 1 on drift; public draftlab/harness only.
- `data/verify.py`: 3x/side pristine worktrees -> verified.jsonl; tests=0 -> error.
- hidden checks: temp root + synthetic fixtures; assert completion claim;
  cross-check cited numbers vs results/*.json; GOTCHA: results/*.json absent
  at bad -> lazy discovery in tests (module level = collection error).
