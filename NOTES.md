# Notes

Next-run state; evidence .agent-evidence/*.log.

## Foundations
- `make bootstrap` (py3.12, rust 1.99.0); `make check` = ruff+pytest+cargo+lints; `make known` strict.

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar resume. GOTCHA: splits.json
  values are dicts; loopgym src NotImplemented.
- `tools/check_results.py`: baselines/benches/toy; PENDING != FAIL. BASECHECK blocked
  (rule 5): base05 0.0764 [0.014,0.153] (n=16) < 0.10 floor; 7 causes ruled out incl
  model identity (494M config, 4/4 probe); base15 in range, 1.5B>0.5B, ref 1.0;
  never widen. Ticket: rangerule.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip. `lgbox.py`: RUNNERS["http"] iff LOOPGYM_LGBOX_URL.
  `toy.py`: banana (first 20 tok ci), coinflip (seeded). `splits.py`: sha1%100, 101/9/27, 137/142.
  `loopgym.py`: Task(prompt/parse/reference/stub)+grade over verified records (diff all
  M/A text + hidden-check dir; 12 loaded, 9 excl D/binary); elided bad-commit files
  (diff -U0 ±3, 4500-chr/file, <=6k tok incl. the omitted-files note — unbounded it
  broke 6k on a wide diff); ```path-block parse; grade = reply on bad worktree,
  pristine V._run_once (outside task.files -> 0). ref=1, stub=0, hidden-test edit=0.
  `toy.py`: banana (first 20 tok ci), coinflip. `splits.py`: sha1%100, 137/142 usable, val 9.
- `loopgym.py`: Task+grade over verified records (diff M/A text + hidden dir; 11 loaded,
  7 excl); ```path parse; grade = reply on bad
  worktree, pristine V._run_once (outside task.files -> 0). Known answer 42
  (ref=1, stub=0, hidden-edit=0).

## lgbox
- `lgbox serve` (axum): /run (setpgid, RLIMIT, Q->503), /health, /metrics;
  FC FakeVmm-only. docker: Dockerfiles (hadolint in check).

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]: chunked max/LSE bwd softmax
  (no [T,V]); torch,cpp,triton,cuda. cpp JIT (-O3 -fopenmp); triton iff CUDA or
  TRITON_INTERPRET=1, C<=1024; cuda host-only.
- `tools/bench_logprob.py` -> results/bench_logprob.json; f64 1e-4, <1/8 naive.

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
- `ds.py` ZeRO-3 (0.19.7): ds_zero3.json bf16 stage3; micro=b*group; CPU:
  compatible_ops[deepspeed_shm_comm]=False.
- Toy gates PASS: toybanana3/toynull3; toy_check exit 0; banana 0.148 -> 1.000
  (<0.2->>0.8); null slope CI contains 0; check_results --only toy 3/3; real GRPO
  may proceed.
- `tools/fetch_src.sh`: vllm/deepspeed/trl -> .tools/src/ (rule 6).

## Data
- `data/mine.py`: %B x blame keyword fixes -> (bad,good); <=200 lines/4 files + 3 seeds.
- `data/verify.py`: 3x/side worktrees, pristine -> verified.jsonl; lazy import when file
  absent at one commit (tests=0 -> error).
- vb1: 12/12. vb2: 9/12; next 22 (draftlab-f67aa44-181ed9d).
- hidden checks: temp root + synthetic fixtures (draftlab data/ gitignored);
  jaxlm: chi2 via stats.chi2.sf (no chisquare); FixedDrafter mid-p token.
  harness: sched.py via runpy (name clash); empty logs/*.jsonl stops grep stdin;
  rmtree git fixture (worktree reused 3x/side). ab.py: setup_tree always commits
  (HEAD^ = base); binds 3-arg (exp,wt,task); fixtures under ROOT/projects.
  25ebe03: bad..good spans 5 commits -> 188-file diff (polyglot fixtures);
  _budgeted now budgets the omitted-files note.
  NOTES.md cands: assert completion claim, not bare numbers (pre-reg lines may
  contain e.g. "42 cells" cost estimate at bad commit).

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: never `rm DIR`, find -delete (EBUSY).
- ruff I/EXE pinned; known/ I001/F401. .venv: no pip (uv/ensurepip).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.
