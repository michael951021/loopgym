# Notes

Next-run state; evidence .agent-evidence/*.log.

## Foundations
- `make bootstrap` (py3.12, rust 1.99.0); `make check` = ruff+pytest+cargo+lints; `make known` strict.

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar resume. GOTCHA: splits.json
  values are dicts; loopgym src NotImplementedError.
- `tools/check_results.py`: baselines/benches/toy; PENDING != FAIL. BASECHECK: n=16 landed
  (landn16; job/basen16 output verified: 144-sidecar/model, first 36 rows = n=4, pass@1
  recomputed = JSON). base05 0.0764 [0.014,0.153] < 0.10 floor: rule-5 suspected bug, range
  mis-calibration not pipeline (6 causes ruled out; n=4 0.0833 also below); base15 0.3056 in
  range, 1.5B>0.5B, ref 1.0. Gated on range author ruling; never widen. basecheck5-verdict-n16.log.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip. `lgbox.py`: RUNNERS["http"] iff LOOPGYM_LGBOX_URL.
  `toy.py`: banana (first 20 tok ci), coinflip (seeded). `splits.py`: sha1%100, 101/9/27, 137/142.
  `loopgym.py`: Task(prompt/parse/reference/stub)+grade over verified records (diff all
  M/A text + hidden-check dir; 11 loaded, 7 excl D/binary); elided bad-commit files
  (diff -U0 ±3, 4500-chr/file, <=6k tok); ```path-block parse; grade = reply on bad
  worktree, pristine V._run_once (file outside task.files -> 0). Known answer 42 @95s
  (ref=1, stub=0, hidden-test edit=0).

## lgbox
- `lgbox serve` (axum): /run (setpgid, RLIMIT, Q->503), /health, /metrics; FC FakeVmm-only.
  docker: Dockerfiles (hadolint in check); no python in image -> no in-image grade.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]: chunked max/LSE + per-chunk bwd softmax
  (no [T,V]); torch,cpp,triton,cuda. cpp: csrc/logprob_cpu.cpp JIT (-O3 -fopenmp) f32/f64.
  triton: iff CUDA or TRITON_INTERPRET=1, C<=1024, masked cols -inf.
  cuda: host-only, csrc/logprob_cuda* + bootstrap_cuda.sh (.venv-nvcc).
- `tools/bench_logprob.py` -> results/bench_logprob.json; f64 1e-4, fused<1/8 naive; peak=chrome trace, triton T=1.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: Batch (mask [p-1,p+c-1)), build_batch, grpo_step (logp=token_logprobs;
  old=detach; ref=disable_adapter); step-0 β=0 ~0. GOTCHA: no last_hidden_state in 5.x;
  kernel unbatched -> flatten.
- `rollout.py` HFBackend (left pad, output_scores fp32; T<=0 greedy). GOTCHA: top_k=0, rep=1.0.
- `rollout_vllm.py` (host): bf16 enable_lora r16; sync_lora LoRARequest(load_inplace)
  id+1/step; logprobs=0 -> sampler_logp; max_tokens explicit; 14/14 ast (rule 6); 0.02-nat.
- `train.py`: TrainCfg toml+--k v; sample->gen->grade->grpo_step->jsonl->ckpt
  <out>^2/ckpt/<name>; --dist+--dry-run; exact resume; exit 1 on orphaned ckpt (resume
  past steps, no metrics). GOTCHA: from_pretrained freezes ALL -> requires_grad lora_.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors (.full_tensor()), grads AVG, no 1/W.
- `ds.py` ZeRO-3 (0.19.7, lazy): ds_zero3.json bf16 stage3; micro=b*group;
  CPU: compatible_ops[deepspeed_shm_comm]=False (no ninja).
- Toy gates: 60st r16 B4 G8 "Name a berry." Fresh toybanana3/toynull3 landed;
  toy_check exit 0 (toyok.log). Banana 0.148 (0-4) -> 1.000 (55-59): known answer
  <0.2 -> >0.8 PASS; slope 0.01055 [0.00702,0.01384].
- null slope -0.00042 [-0.00166,0.00078] contains 0: known answer PASS.
  results/toy.json committed (keys banana_early/late, *_slope_ci, null_slope_ci);
  check_results --only toy 3/3 PASS. Trainer gate passed; real GRPO may proceed.
- `tools/fetch_src.sh`: vllm/deepspeed/trl sources -> .tools/src/ (rule 6).

## Data
- `data/mine.py`: %B x blame keyword fixes -> (bad,good); <=200 lines/4 files + 3 seeds.
- `data/verify.py`: 3x/side worktrees, pristine -> verified.jsonl; lazy import when file
  absent at one commit (tests=0 -> error); 411f1c8 spec.py present -> top-level fine.
- vb1: 12/12. vb2: 5/12 (cands 13-17 verified); next 18.
- hidden checks: temp root + synthetic fixtures (draftlab data/ gitignored);
  jaxlm: chi2 via stats.chi2.sf (no chisquare); FixedDrafter mid-p token.
  harness: sched.py via runpy (name clash); empty logs/*.jsonl stops grep stdin;
  rmtree git fixture (worktree reused 3x/side).

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: never `rm DIR`, find -delete (EBUSY).
- ruff I/EXE pinned; known/ I001/F401. .venv: no pip (uv/ensurepip); polars 2.0.0.
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- p-l-f ref audit: timeout @30s (passes @60s).
- env.grade default timeout=30 < loopgym slowest check (~25s): pass cfg.timeout=120.
