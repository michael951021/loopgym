# loopgym REPORT

## Question

Does GRPO post-training lift the pass@1 of Qwen2.5-Coder on coding tasks made
from real bugs agents shipped (loopgym) and from Exercism, with confidence
intervals, hack-proof grading, and every number recomputable from `results/`
by `tools/check_report.py`? What works, what does not, and what does a 1.5B
run on two 3090s show?

## Task sources

**Exercism Python.** 142 exercises, 137 usable (5 unusable: the reference
solution scores 0 on 3 — one of them a timeout — and the stub already scores
1 on 2). Splits by sha1(exercise id): 101/9/27 train/val/test. Reward 1.0 iff
the exercise's hidden tests, copied pristine into a fresh directory with only
the model's solution file, all pass (pytest junit XML); else 0.0.

**loopgym v0.** Tasks mined from fix commits in this machine's own draftlab
and harness repos: 141 candidates, 3/3 planted seeds recalled. Verification
rule: a candidate becomes a task only if its hidden pytest check fails 3/3
runs on the bad commit and passes 3/3 on the fix commit
(`loopgym.data.verify`). Freeze `data/loopgym/v0`: 26 tasks — 12/6/8
code/config/docs, 22/4 draftlab/harness — split 13/13 by sha1(id),
test_frac 0.50. Kill criterion (< 25 tasks → eval-only mode) outcome: met
(26 ≥ 25), so v0 is kept as a training set, not just an eval.

## Method

GRPO per CLAUDE.md: group advantages (r − mean)/(std + 1e-6), k3 KL, ratio
clipped at 0.2, token-mean aggregation. 0.5B CPU run (grpo05): LoRA
r16/α32 on q/k/v/o, batch 4, group 8, 40 steps, 384 new tokens, T=1.0,
lr 2e-5, β=0.02, seed 1337, local runner. Evals: pass@1 at T=0.8, 512 max new
tokens; CIs are bootstrap resamples over **tasks** (n_boot=2000, seed 0);
paired Δ over the 9 identical val tasks. Known-answer controls before any
real run: gold reference solutions, a planted "say banana" reward and a
coin-flip null reward (toy, 60 steps), and a deterministic hack screen that
regenerates completions at their eval seeds and re-grades them.

## Results

### Baselines (pass@1, 9 val tasks)

| model | pass@1 | 95% CI | note |
|---|---|---|---|
| 0.5B base (n=16) | 0.0764 | [0.014, 0.153] | below pre-run floor 0.10 — accepted by range-author ruling (see Corrections) |
| 1.5B base (n=16) | 0.3056 | [0.132, 0.486] | in range; 0.3056 > 0.0764 |
| reference (gold solutions) | 1.0000 | [1.000, 1.000] | known answer: gold scores 1 |
| 0.5B base, loopgym v0 test (13 tasks, n=4) | 0.0000 | [0.000, 0.000] | in range [0.0, 0.25] |
| 1.5B base, loopgym v0 test (13 tasks, n=4) | 0.0000 | [0.000, 0.000] | in range [0.0, 0.25] |

The base models solve no loopgym test task: these are real bugs other agents
shipped, as designed.

### Toy known answers (planted effect / null control)

| reward | early mean | late mean | slope CI over steps | verdict |
|---|---|---|---|---|
| banana (planted) | 0.1484 | 1.0000 | 0.0105 [0.0070, 0.0138] | learned (CI excludes 0) |
| coin-flip null | 0.4609 | 0.4750 | [-0.0017, 0.0008] | no trend (CI contains 0) |

The trainer's math is validated before the real run.

### CPU GRPO (grpo05: 0.5B, exercism train, LoRA)

- Val pass@1: grpo05 0.1111 [0.000, 0.222] vs base05 0.0764 [0.014, 0.153];
  paired Δ +0.0347 [-0.0625, +0.1250] → **null** (CI contains 0).
- Train reward slope -0.00311/step [-0.00556, -0.00002] (decline);
  KL first10 → last10: 1.59e-05 → 3.02e-04, max 4.06e-04; clip_frac 0.0;
  frac_groups_all_same 0.675 — 67.5% of groups carried zero advantage signal.
- Hack screen (20 regenerated completions): 20/20 sidecar reward match
  (determinism known answer), 3 flagged — `references_test_names` only, no
  reward-hack pattern.
- CPU step time: median 10.60 min, range [8.21, 21.93] min (pre-run range
  5–25 min). Config: batch 4, group 8, 40 steps.

### lgbox bench (200 grades per cell, reference solution)

| runner, concurrency | grades/s |
|---|---|
| local c1 | 6.08 |
| http c1 | 6.43 |
| local c16 | 41.93 |
| http c16 | 42.06 |

Latency p50 c1 local/http: 0.1645/0.1549 s; p95 c16: 0.499/0.455 s.
HTTP overhead per grade at c=1: -9.1 ms (known answer: < 50 ms). Every one of
the 1200 grades scored reward 1.

### Fused log-prob kernel bench (CPU, fp32, V=151936, T=2048)

| impl | median (s) | peak extra (B) | vs naive peak |
|---|---|---|---|
| naive | 1.2203 | 2489344008 | — |
| torch (chunked) | 1.5045 | 201369608 | 8.1% (< 1/8) |
| cpp | 10.6494 | 180240 | 0.007% (< 1/8) |

All implementations within 4.649e-05 of the f64 reference (tolerance 1e-4).
Triton ran under `TRITON_INTERPRET=1` (T=1 overhead cells only) and CUDA
compiled host-only: real-GPU timings land with the host GPU run.

## What failed, and why

- **grpo05 is a null result.** 40 steps × 4 prompts with 67.5% of groups
  carrying zero advantage gave no detectable val lift (paired Δ CI contains
  0) and train reward declined. The pipeline is not broken (KL bounded, no
  clipping, hack screen clean, toy controls pass) — this scale simply does
  not move the metric. The 1.5B GPU run below asks the same question with a
  3× larger model and 2.5× the steps.
- **base05 val 0.0764 sat below the pre-run floor 0.10.** Ten pipeline causes
  were ruled out (atk1-10) and the range author ruled it accepted; the range
  itself was never widened (Corrections).
- **5 of 142 Exercism exercises are unusable** (broken reference or stub
  grading) and were dropped before any split.
- **loopgym v0 test at n=4 shows 0.0000 for both base models** — in range,
  but too little data to say anything beyond "base models do not fix these
  bugs"; a post-GRPO loopgym eval needs the GPU run.
- **Triton and CUDA kernel paths were untested on real GPUs** in the sandbox
  (interpret / compile-only, per rule 2). The GPU run's known tests now cover
  both, and the 1.5B run trained with the Triton kernel.

## GPU run: GRPO 1.5B on two 3090s

`scripts/gpu_run.sh`, configs/grpo15_gpu.toml: LoRA r16/α32 on q/k/v/o over an
fp32 base, batch 4, group 8, 100 steps, 384 new tokens, T=1.0, lr 2e-5,
β=0.02. vLLM generates on GPU0 (bf16, the LoRA re-synced every step); the
trainer runs on GPU1 with the Triton log-prob kernel. Gates before training:
CUDA and Triton known tests, then vLLM vs HF log-probs on the same tokens,
fp32 mean |Δ| 0.00016 nats over 512 tokens (gate < 0.02), so the rollout
engine's log-probs can stand in for the trainer's.

**Training.** 100/100 steps (batch 4, group 8, 100 steps); step time median 7.12 min, range
[4.06, 22.11] min; 13.3 h of training over two tries (the 12 h job timeout
cut the first at step 90, the second resumed from the step-90 checkpoint).
Train reward slope +0.00082/step [-0.00019, +0.00188]; first-20 mean 0.175 → last-20 mean 0.263.
KL first10 → last10: 3.50e-06 → 5.06e-04; clip_frac 0.0 throughout. Groups with all 8
rewards equal (zero advantage, no gradient): 0.693 of groups on average,
about the same as grpo05 (0.675). The in-run validation checks (8 val tasks,
a different 8 each time) read 0.250, 0.219, 0.266, 0.297, 0.391 at steps
20–100.

**Evals.** The job's own evals use n=1 per task (adapter at step 100):
Exercism val 0.2222 [0.000, 0.556], test 0.1852 [0.074, 0.333], loopgym test 0.0000 [0.000, 0.000]. n=1 is not
comparable to the n=16 baseline, so `evalgrpo15` re-scores the adapter with
base15's protocol (T=0.8, 512 tokens, same seeds):

| model | Exercism val (n=16) | loopgym test (n=4) |
|---|---|---|
| 1.5B base | 0.3056 [0.132, 0.486] | 0.0000 [0.000, 0.000] |
| 1.5B GRPO, 100 steps | 0.3542 [0.167, 0.556] | 0.0000 [0.000, 0.000] |

Paired over the 9 val tasks: Δ +0.0486 [+0.0000, +0.1042] → **null (the CI reaches 0, so the +0.05 is not a detected gain)**. Hack screen
(20 regenerated completions): 20/20 sidecar match.

**What it took to run.** Each fix is a commit with a test where the sandbox
allows one:

- vLLM's LoRA kernel asserts fp16/bf16, and the fp32 check engine was built
  with LoRA on; the check compares base-model log-probs, so it now builds
  without LoRA (79d33f0). It also passes one conversation per prompt, as both
  backends expect (b31b8f1).
- FlashInfer's sampler JIT-builds on first use, and the pip nvcc (cu13) is
  incompatible with its bundled headers; `VLLM_USE_FLASHINFER_SAMPLER=0` uses
  vLLM's PyTorch sampler (7888448).
- Triton and the C++/CUDA extension launch on the *current* device; the
  trainer's tensors live on cuda:1 while cuda:0 stays current, so the kernels
  now launch under `torch.cuda.device(h.device)`, with a 2-GPU test (fd1bbd9).
- The trainer ran out of memory on GPU1: an fp32 1.5B is 6 GB of weights, and
  saved activations for one 2k-token row filled the rest by step 7 even at
  micro_batch 1. Gradient checkpointing recomputes them in backward (gradients
  identical, `tests/test_grad_ckpt.py`) and leaves room for micro_batch 4
  (6a126b3, e27b21d).
- Resuming rebuilt the policy on the CPU and never moved it back to the
  trainer GPU, so the retry pushed every batch onto vLLM's GPU and ran out of
  memory (37fb800, `tests/test_resume_device.py`).

**Scaling study (two-node / AWS): PENDING.** `--two-node-local` and the AWS
spot run (deploy/aws, $10 cap) are written and tested offline but have not
run; there are no `results/gpu/scale_*.json` files.

## Corrections

- **base05 val 0.0764** is outside the pre-run range [0.10, 0.55]. Accepted by
  range-author ruling `results/rulings/base05_val.ruling.json` (kevin: "task-
  clustered CI [0.014, 0.153] straddles the 0.10 floor; ten pipeline causes
  ruled out (atk1-10)"). `check_results.py` and `check_report.py` apply the
  ruling verbatim and cite it here; the range was not widened.
- No other numbers required correction at the time of writing.
