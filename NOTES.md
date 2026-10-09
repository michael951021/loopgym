# Notes

State for next run; evidence: .agent-evidence/*.log.

## Foundations
- `make bootstrap`: .venv py3.12 torch+cpu, rust 1.99.0.
- `make check` = ruff + pytest + cargo + lints; `make known` = strict known.

## Eval
- `bootstrap_ci(values,groups)`; `pass_at_k` (Chen 2021), CI over tasks.
- `eval/run.py`: --model/--adapter | --reference; per-sample seed+idx*100+i; sidecar
  <out>.samples.jsonl resume; --reference val pass@1 1.000
  (evalcli-ref.log). GOTCHA: splits.json values are dicts; loopgym src NotImplementedError.

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151,643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL;
  reward 1 iff tests>0 no fail/err/skip (_xml_counts shared). `lgbox.py`:
  RUNNERS["http"] iff LOOPGYM_LGBOX_URL. `toy.py`: banana (first 20 tok ci), coinflip (seeded).
  `splits.py`: sha1%100, 101/9/27, 137/142.

## lgbox
- `lgbox serve` (axum): /run (setpgid, RLIMIT, tails, mpsc Q->503), /health, /metrics;
  Firecracker Vmm+VmPool, --dry-run, FakeVmm-only.
- docker: lgbox.Dockerfile (rust->distroless/cc:nonroot) + trainer.Dockerfile (cuda
  runtime, /app/.venv-gpu CUDA torch); .dockerignore; hadolint clean in make check.
  GOTCHA: distroless has no python -> process backend can't grade in-image (needs a python
  base or firecracker rootfs). Makefile hadolint find now matches *.Dockerfile.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]`: chunked online max/LSE, bwd per-chunk
  softmax (no [T,V]); impls torch,cpp,triton,cuda (cuda host-only).
  cpp: csrc/logprob_cpu.cpp JIT (.tools/build, -O3 -fopenmp), f32+f64; C++ bwd.
  triton: fwd 1 prog/row; bwd_h row, bwd_w V-tile; iff CUDA or TRITON_INTERPRET=1;
  interpret cap 1048576 -> C<=1024; masked cols -inf.
  cuda: csrc/logprob_cuda* + bootstrap_cuda.sh (.venv-nvcc, CUDA 13.2.86; cu12 lacks
  nvcc, 12.9 fails gcc15/glibc2.43); reg iff torch.cuda (host); gate
  test_cuda_backward.
- `tools/bench_logprob.py` -> results/bench_logprob.json; f64 1e-4 + fused<1/8 naive PASS;
  peak=chrome trace, triton T=1 interpret.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: Batch (mask [p-1,p+c-1)), build_batch, grpo_step (logp=token_logprobs;
  old=detach; ref=disable_adapter); step-0 β=0 ~0.
  GOTCHA: no last_hidden_state in 5.x; kernel unbatched -> flatten.
- `rollout.py` HFBackend (left pad, output_scores fp32; T<=0 greedy). GOTCHA:
  top_k=0, rep=1.0 (defaults leak). 
- `rollout_vllm.py` (host): bf16, enable_lora (r16); sync_lora -> LoRARequest(load_inplace)
  id+1/step; logprobs=0 -> sampler_logp; explicit max_tokens (def 16);
  14/14 ast (rule 6); 0.02-nat.
- `train.py`: TrainCfg (toml + --k v): sample->gen(seed+s)->grade->grpo_step->
  jsonl->ckpt <out>^2/ckpt/<name>; --dist {fsdp,deepspeed}+--dry-run; exact resume.
  GOTCHA: from_pretrained freezes ALL -> requires_grad=("lora_" in name) before AdamW.
- `dist.py` FSDP2: fully_shard, full_state_dict DTensors (.full_tensor()), grads AVG
  (W>1), no 1/W.
- `ds.py` ZeRO-3 (0.19.7, lazy): ds_zero3.json bf16 stage3, offload off; micro=b*group.
  CPU: ops.__compatible_ops__[deepspeed_shm_comm]=False (no ninja).
- Toy gates: toy_{banana,null}.toml (r16 B4 G8 24tok lr5e-5 β0 60st); step-1
  0.281>0.2, 60-st must confirm (rule 5).
- `tools/fetch_src.sh`: vllm/deepspeed/trl sources -> .tools/src/ (rule 6).

## Data
- data/raw/ = host export.
- `data/mine.py`: %B x blame keyword fixes -> (bad,good); <=200 lines/4 files + 3 seeds.
- `data/verify.py`: worktrees bad+good + pristine check, 3x/side -> verified.jsonl;
  checks may import worktree code but must COLLECT at both commits (import error ->
  "error"); Makefile ignores tests/hidden.
- `state/verify/batch1.md`: queue = candidates.jsonl score order; first unchecked step
  wins; vb1 needs all 12 verdicts. 8/12 verified (all bad 0/3, good 3/3). draftlab
  data/ gitignored -> hidden checks use a temp root + synthetic fixtures.

## Gotchas
- .venv/.tools/data/models/ckpt/runs are BIND MOUNTS: `find DIR -mindepth 1 -delete`,
  never `rm DIR` (EBUSY).
- ruff I/EXE pinned; known/ I001/F401 (rule 1).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork (tests /bin/sleep).
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- NOTES.md: merges duplicate lines; condense.
- .venv: pip (ensurepip) + polars 2.0.0; or `uv pip install`.
