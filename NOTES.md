# Notes

Next-run state; logs in .agent-evidence/.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: never `rm DIR`/find -delete.
- ruff I/EXE pinned; known/ I001/F401. .venv: no pip (uv/ensurepip).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar resume.
- `tools/check_results.py`: baselines/benches/toy; PENDING != FAIL; out-of-range
  passes only via range-author ruling (malformed fails closed). base05 0.0764
  [0.014,0.153] < 0.10 -> exit 1. HUMAN-ONLY: kevin commits
  results/rulings/base05_val.ruling.json (forms: state/tickets/kevinruling.md).
  atk18: absent all refs AND on disk (no remote); fresh verdict exit 1
  unchanged (kevinruling18-verdict.log). 18 dispatches; humandispatch stands;
  basecheck gated.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL.
  `lgbox.py`: RUNNERS["http"] iff LOOPGYM_LGBOX_URL.
  `toy.py`: banana (first 20 tok ci), coinflip (seeded). `splits.py`: sha1%100, 101/9/27,
  137/142 usable, val 9. `loopgym.py`: Task(prompt/parse/reference/stub)+grade over verified
  records (diff M/A text + hidden dir); elided bad-commit files (diff -U0 ±3, 4500-chr/file,
  <=6k tok); ```path-block parse; grade = reply on bad worktree, pristine V._run_once
  (outside task.files -> 0); ref=1, stub=0, hidden-test edit=0.

## lgbox
- `lgbox serve` (axum): /run (setpgid, RLIMIT, Q->503), /health, /metrics;
  FC FakeVmm-only. docker: Dockerfiles.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]: chunked max/LSE bwd softmax (no
  [T,V]); torch,cpp,triton,cuda; cpp JIT (-O3 -fopenmp); triton iff CUDA or
  TRITON_INTERPRET=1, C<=1024; cuda host-only. bench_logprob.py -> bench_logprob.json, f64 1e-4.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: Batch (mask [p-1,p+c-1)), build_batch; grpo_step (logp=token_logprobs;
  old=detach; ref=disable_adapter); step-0 β=0 ~0; no last_hidden_state in 5.x;
  kernel unbatched -> flatten.
- `rollout.py` HFBackend (left pad, output_scores fp32; T<=0 greedy). GOTCHA: top_k=0, rep=1.0.
- `rollout_vllm.py` (host): bf16 enable_lora r16; sync_lora LoRARequest(load_inplace)
  id+1/step; logprobs=0 -> sampler_logp; max_tokens explicit; 0.02-nat.
- `train.py`: TrainCfg toml+--k v; sample->gen->grade->grpo_step->jsonl->ckpt
  <out>^2/ckpt/<name>; --dist+--dry-run; exact resume; exit 1 on orphaned ckpt;
  from_pretrained freezes ALL -> requires_grad lora_.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors (.full_tensor()), grads AVG, no 1/W.
- `ds.py` ZeRO-3: ds_zero3.json bf16 stage3; micro=b*group; CPU compatible_ops[deepspeed_shm_comm]=False.
- Toy gates PASS: toybanana3/toynull3; toy_check exit 0; banana 0.148 -> 1.000;
  null slope CI contains 0.
- `tools/fetch_src.sh`: vllm/deepspeed/trl -> .tools/src/ (rule 6).

## Data
- `data/mine.py`: %B x blame keyword fixes -> (bad,good); <=200 lines/4 files.
- `data/verify.py`: 3x/side worktrees, pristine -> verified.jsonl; lazy import when file
  absent at one commit (tests=0 -> error).
- vb1/vb2 done 24/24 (checked; gate >= 30% passed). vb3: batch3 = 16 steps
  (cands 25-40, score order); 4/16 done: harness-bc2cdd8, draftlab-35a5d46-1e9f5d3,
  draftlab-bea1633-9b213ef, draftlab-7074368-07efa48 all verified (bad 0/3, good
  3/3); one per run; next = draftlab-8e65e54-ea27462; vb3 milestone still unchecked.
- hidden checks: temp root + synthetic fixtures (draftlab data/ gitignored).
  jaxlm: chi2 via stats.chi2.sf. NOTES.md cands: assert the completion claim, not
  bare numbers (bad-commit lines may hold cost estimates).
  harness cands: find ROOT from sys.path, fabricate a git project under ROOT/projects.
