# Notes

Next-run state; evidence .agent-evidence/*.log.

## Foundations
- `make bootstrap` (py3.12, rust 1.99.0); `make check` = ruff+pytest+cargo+lints; `make known` strict.

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; sidecar <out>.samples.jsonl resume.
  GOTCHA: splits.json values are dicts; loopgym src NotImplementedError.
- `tools/check_results.py` (--only/--results-dir): baselines/benches/toy; PENDING != FAIL.
  BASECHECK: base05 0.0833 < 0.10 floor blocks done (rule 5; 6 causes ruled out);
  n=16 rerun id basen16 unrun.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip. `lgbox.py`: RUNNERS["http"] iff LOOPGYM_LGBOX_URL.
  `toy.py`: banana (first 20 tok ci), coinflip (seeded). `splits.py`: sha1%100, 101/9/27, 137/142.
  `loopgym.py`: Task(prompt/parse/reference/stub)+grade. Task = verified record whose whole
  diff is M/A text with a hidden-check dir (10 loaded post-merge; 6 records excluded: D/binary).
  Prompt: ticket + bad-commit files elided at git diff -U0 hunks ±3, 4500-chr/file, <=6k tok
  (chars/3, candidate files first, rest by name). Parse: ```path blocks to next path fence
  (slash/ext/bare-known-name; inner ``` safe). Grade: apply reply to bad worktree, pristine
  V._run_once; reply file outside task.files -> 0 (anti-hack). Known answer 42 pass @95s
  (ref=1, stub=0, hidden-test edit=0; parse round-trip \n-tolerant).

## lgbox
- `lgbox serve` (axum): /run (setpgid, RLIMIT, Q->503), /health, /metrics; FC FakeVmm-only.
- docker: Dockerfiles (hadolint in check). GOTCHA: no python in image -> no in-image grade.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]: chunked max/LSE + per-chunk bwd softmax
  (no [T,V]); torch,cpp,triton,cuda.
  cpp: csrc/logprob_cpu.cpp JIT (.tools/build, -O3 -fopenmp), f32+f64.
  triton: iff CUDA or TRITON_INTERPRET=1; interpret cap -> C<=1024; masked cols -inf.
  cuda: host-only; csrc/logprob_cuda* + bootstrap_cuda.sh (.venv-nvcc); gate test_cuda_backward.
- `tools/bench_logprob.py` -> results/bench_logprob.json; f64 1e-4, fused<1/8 naive;
  peak=chrome trace, triton T=1.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: Batch (mask [p-1,p+c-1)), build_batch, grpo_step (logp=token_logprobs;
  old=detach; ref=disable_adapter); step-0 β=0 ~0. GOTCHA: no last_hidden_state in 5.x;
  kernel unbatched -> flatten.
- `rollout.py` HFBackend (left pad, output_scores fp32; T<=0 greedy). GOTCHA: top_k=0, rep=1.0.
- `rollout_vllm.py` (host): bf16 enable_lora r16; sync_lora LoRARequest(load_inplace)
  id+1/step; logprobs=0 -> sampler_logp; max_tokens explicit; 14/14 ast (rule 6); 0.02-nat.
- `train.py`: TrainCfg toml+--k v; sample->gen(seed+s)->grade->grpo_step->jsonl->
  ckpt <out>^2/ckpt/<name>; --dist+--dry-run; exact resume.
  GOTCHA: from_pretrained freezes ALL -> requires_grad lora_ before AdamW.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors (.full_tensor()), grads AVG, no 1/W.
- `ds.py` ZeRO-3 (0.19.7, lazy): ds_zero3.json bf16 stage3; micro=b*group;
  CPU: ops.__compatible_ops__[deepspeed_shm_comm]=False (no ninja).
- Toy gates: toy_{banana,null}.toml 60st r16 B4 G8. toyok: "Name a fruit." base 0.242
  broke early<0.2 precondition (run step-1 0.281); trainer innocent (probe == step 1).
  PROMPT "Name a berry." (base 0.109; pilot 10st 0.125->0.656). Stale runs ->
  toyok-stale-*; re-runs toybanana2/toynull2, then toyok writes results/toy.json.
- `tools/fetch_src.sh`: vllm/deepspeed/trl sources -> .tools/src/ (rule 6).

## Data
- data/raw/ = host export.
- `data/mine.py`: %B x blame keyword fixes -> (bad,good); <=200 lines/4 files + 3 seeds.
- `data/verify.py`: 3x/side worktrees, pristine -> verified.jsonl; lazy import when file
  absent at one commit (tests=0 -> error); 411f1c8 spec.py present -> top-level fine.
- vb1: 12/12. vb2: batch2.md, 3/12 (cands 13-15 verified); next 16.
- hidden checks: temp root + synthetic fixtures (draftlab data/ gitignored);
  jaxlm: chi2 via stats.chi2.sf (no chisquare); FixedDrafter mid-p token.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: never `rm DIR`, find -delete (EBUSY).
- ruff I/EXE pinned; known/ I001/F401.
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- NOTES.md: merges duplicate lines; condense.
- .venv: no pip (uv or ensurepip); polars 2.0.0.
- p-l-f ref audit: times out @30s (passes @60s); test split only.
- env.grade entry-point default timeout=30 < loopgym's slowest check (~25s chi2): future
  loopgym trainer/eval branch must pass cfg.timeout=120 (or V.TIMEOUT).
