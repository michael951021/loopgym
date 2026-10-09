# loopgym — RL post-training of a small coding model, on tasks made from bugs agents shipped

Train Qwen2.5-Coder (0.5B on CPU here, 1.5B on GPU on the host) with GRPO to fix code. Two task sources:
Exercism Python exercises (`data/raw/exercism-python`, fast to grade) and **loopgym**: defects this machine's own
agents shipped in draftlab and the harness, each turned into (repo at the bad commit, ticket text, a hidden check
that fails there and passes at the fix). The systems are the point: a Rust sandbox server that grades untrusted code,
a fused log-prob kernel (C++ / CUDA / Triton), FSDP and DeepSpeed training, vLLM rollouts with weight sync, and
host / AWS scripts for the GPU runs.

## HARD RULES
1. **Never edit `tests/known/`.** Kevin wrote those tests before the code so the code cannot grade itself. If one
   looks wrong, write why in NOTES.md and stop with RESULT: not done. Do not skip, xfail or loosen it, and do not
   add a second implementation just to pass it. `make known` fails when anyone but Kevin touched the directory.
2. **No GPU in the sandbox.** Everything here runs on CPU: torch CPU wheels, Triton under `TRITON_INTERPRET=1`,
   CUDA only compiled (`nvcc -c`), never run. GPU paths sit behind flags and in `scripts/gpu_run.sh`, which Kevin
   runs on the host. Cap threads: `OMP_NUM_THREADS=8` (other agents share the CPU).
3. **Long runs are compute jobs, never your session.** Training, evaluation sweeps and benchmarks are TODO lines with
   `(run: \`CMD\`)` that lib/jobs.py runs after your ticket. Your ticket writes the code, the config and a smoke
   test that finishes in **under 5 minutes**. A command that needs more than 10 minutes does not belong in a run.
4. **Never apply or deploy anything.** Terraform: `init -backend=false` + `validate` only; no AWS credentials, no
   `plan`. Docker is not available: write Dockerfiles and lint them with hadolint. `lgbox --backend firecracker`
   needs /dev/kvm: test it with the fake VMM only.
5. **Numbers are claims.** A metric ticket is done only when its known answers (below) hold on the real pipeline.
   A result outside its plausible range blocks "done": record it in NOTES.md as a suspected bug with the evidence
   and stop. Never explain an out-of-range number away ("different protocol", "expected noise") in a report.
6. **Check every third-party API you call** against the pinned source in `.tools/src/` (vLLM, DeepSpeed, TRL; see
   tools/versions.txt) with `rg -n`, not from memory. Pinned versions only.
7. Never Read image files (the model has no vision); describe figures from the data that made them.
8. `data/`, `models/`, `ckpt/`, `runs/`, `.tools/`, `.venv/` are gitignored and shared between agents. Commit
   `results/` (small JSON, < 2 MB each) and `configs/`.
9. BE SUCCINCT in code comments, notes and docs.

## Layout (target)
- `loopgym/env/` tasks + grading: `exercism.py`, `splits.py`, `toy.py` (known-answer rewards), `lgbox.py` (client),
  `loopgym.py` (mined tasks) · `loopgym/data/` mining (`mine.py`) and verification (`verify.py`) of loopgym tasks
- `loopgym/rl/` `grpo.py` (math) · `rollout.py` (HF backend) · `rollout_vllm.py` · `batch.py` (one GRPO step from
  completions) · `train.py` (loop, CLI, logging, checkpoints) · `dist.py` (FSDP2) · DeepSpeed config path
- `loopgym/kernels/` fused token log-probs: `torch`, `cpp` (csrc/logprob_cpu.cpp), `triton`, `cuda` (csrc/logprob_cuda.cu)
- `loopgym/eval/` `stats.py` (bootstrap CIs), `passk.py` · `loopgym/models.py`
- `lgbox/` Rust sandbox server (axum) · `configs/*.toml` · `scripts/gpu_run.sh` · `deploy/{aws,docker}`
- `tests/known/` Kevin's known-answer tests (read-only) · `tests/` everything else · `results/` · `REPORT.md`

## Commands
- `make bootstrap` · `make check` (ruff + pytest tests/ + cargo test + lints) · `make known` (strict known tests)
- Python: `.venv/bin/python -m loopgym.<module>` · Rust: `PATH=$PWD/.tools/cargo/bin:$PATH cargo test` in lgbox/
- Long output: `python3 /opt/agent/hooks/capture.py --log /work/.agent-evidence/NAME.log --timeout 600 -- CMD`

## Models
- `models/qwen2.5-coder-0.5b-instruct` (Qwen/Qwen2.5-Coder-0.5B-Instruct): CPU policy for every run here.
- `models/qwen2.5-coder-1.5b-instruct`: GPU policy (host) and the CPU baseline eval only.
- `loopgym.models.tiny_config()`: 2-layer, d=64, vocab-512 random Qwen2 for fast tests.
All three share Qwen's tokenizer (vocab 151936 for the real ones).

## Key definitions (the known tests pin these)
- **Group advantage**: rewards [B, G] (G completions per prompt) → (r − mean_row) / (std_row + 1e-6), `std` the
  sample std (Bessel, as TRL); `scale="none"` skips the division (Dr. GRPO). A constant group gives 0, not NaN.
- **k3 KL** per token: d = ref_logp − logp, k3 = exp(d) − d − 1 (≥ 0; its mean over samples from the policy is
  KL(policy ‖ ref)).
- **Policy loss**: ratio = exp(logp − old_logp); obj = min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) − β·k3;
  `agg="token-mean"`: loss = −Σ obj·mask / Σ mask; `agg="seq-mean"`: mean over sequences of each one's masked mean.
  old_logp and ref_logp are recomputed by the trainer's forward, never taken from the sampler.
- **Token log-probs**: logp[t] = log softmax(hidden[t] @ W.T)[target[t]], hidden = final hidden states after the
  last norm, W = lm_head weight; row t predicts token t+1, so targets = ids[1:] for hidden[:-1].
- **Reward** (exercism and loopgym): 1.0 iff the task's hidden tests, copied pristine into a fresh directory with
  only the solution file(s) taken from the model, all pass (pytest junit XML: tests > 0, no failures/errors/skips);
  else 0.0. Timeouts are 0.
- **Prompt and reply** (Exercism): system "You are an expert Python programmer. Reply with the complete solution
  file in one ```python code block and nothing else."; user = .docs/instructions.md (+ .docs/instructions.append.md)
  + "\n\nComplete this file (`<solution_file>`):\n```python\n<stub>\n```". Parsing takes the LAST ```python block
  (else the whole reply) as the solution file; a reply with no code scores 0. Same rules for loopgym with
  ```<path> blocks per file.
- **pass@k** (Chen et al. 2021): per task 1 − C(n−c, k)/C(n, k); mean over tasks; CI = bootstrap over **tasks**
  (`bootstrap_ci(values, groups=task)`), never over samples.
- **Splits**: by sha1(exercise or task id): train 70 / val 15 / test 15. Never train on val/test.

## Known answers (each metric ticket checks its own; reports state them next to the number)
- **GRPO math, CIs, rewards, log-probs**: `tests/known/` (hand cases, reward hacks score 0, real-model log-probs equal
  HF's own loss within 1e-3).
- **Rollout vs trainer**: the sampler's log-prob of each sampled token at temperature 1 equals the trainer's
  recomputed logp within 1e-4 on the CPU HF backend (same model, fp32). On vLLM (host) the mean |Δ| per token must
  be < 0.02 nats in fp32 / bf16 noted separately; larger means a weight-sync or sampling-param bug.
- **Planted effect / null control** (toy rewards, `loopgym/env/toy.py`): reward "says banana in its first 20 tokens"
  must rise from < 0.2 to > 0.8 mean within 60 steps (0.5B, LoRA r=16, lr 2e-5..1e-4); a coin-flip reward that ignores
  the text must show no trend (bootstrap CI of the slope over steps contains 0). If banana does not learn, the
  trainer has a bug: never run the real GRPO job before this passes.
- **Distributed = single process**: one optimizer step with FSDP2 on 2 ranks × batch b equals one single-process
  step on batch 2b: every parameter within 1e-5 (tiny_config, fp32, gloo).
- **Graders**: every usable exercise's reference solution (`.meta/example.py`) scores 1 and its stub 0, on both the
  local and the lgbox runner. A loopgym task's fix commit scores 1 and its bad commit 0, 3 runs out of 3.
- **Plausible ranges written before the runs** (outside = suspected bug, rule 5):
  - Qwen2.5-Coder-0.5B-Instruct exercism val pass@1 (T=0.8): 0.10–0.55; 1.5B: 0.25–0.75; 1.5B > 0.5B.
  - Base models on loopgym tasks pass@1: 0.0–0.25 (they are real bugs other agents missed).
  - Initial GRPO loss with on-policy samples: |loss| < 0.05 at step 0 when β = 0 (advantages have mean 0).
  - Fused log-prob peak extra memory: < 1/8 of the naive T×V×4 bytes; naive at T=2048, V=151936 ≈ 1.24 GB.
  - CPU GRPO step (0.5B, B=4, G=8, 256 new tokens, HF backend): 5–25 min. lgbox overhead per grade: < 50 ms over the
    local runner at concurrency 1.

## GPU and AWS (host only; agents only write and dry-run these)
- The model server holds ~21 GB of each 3090 while agents run, so GPU work needs the agents stopped:
  `scripts/gpu_run.sh` (Kevin) stops them, builds `.venv-gpu`, runs the known tests on CUDA/Triton, the vLLM-vs-HF
  check, then the 1.5B GRPO run (vLLM rollouts on GPU0, trainer on GPU1).
- **Budget: $10 total for cloud, ever.** Training happens on the host 3090s. Multi-node is rehearsed for free with two
  torchrun "nodes" on this host (`gpu_run.sh --two-node-local`). The AWS run is only a short scaling study: 2× spot
  g4dn.xlarge (T4), 0.5B, 3 configs × 50 steps (~3 node-hours, expected < $2), with auto-terminate after max_hours,
  a spot price cap, a $10 AWS Budget alert, and `aws_launch.sh` refusing any plan estimated over $10. Never add
  always-on resources (NAT gateway, load balancer, S3 lifecycle-less buckets, larger instances) to the Terraform.
