# Notes

## Gotchas
- Bind mounts to Kevin's host (never `rm DIR`/find -delete): /work, .git,
  /home/agent, .venv, .tools, data, models, ckpt, runs; dispatcher is
  host-side (task arrives via AGENT_TASK env).
- ruff I/EXE pinned, whole tree incl tests/hidden; known/ I001/F401. .venv: no pip (uv/ensurepip).
- RLIMIT_NPROC=64 counts ALL uid-1000 -> transient Cannot fork.
- torch 2.14 FSDP2: fully_shard auto-inits env://; rank() pre-init 0.
- env.grade default timeout=30 < slowest loopgym check: pass cfg.timeout=120.

## Scheduling
- `(human-only[: RELPATH])`: hold open tagged lines; gate
  tools/sched_guard.py --line <id> (0 ok/1 skip/2 unknown); open tagged
  lines held.
- hostgate: atk4: host tip still 4cb686d (all refs scanned; main/origin
  older); no HUMAN_ONLY -> unapplied; fresh worktree RED->GREEN (real
  TODO.md), 4/4, 0/16; shipped tools/hostgate_apply.sh (tested). Kevin:
  run on host + restart dispatcher.

## Eval
- `pass_at_k` (Chen 2021) + `bootstrap_ci(values,groups)`: CI over tasks.
  `eval/run.py`: --model/--adapter | --reference; sidecar resume.
- `tools/check_results.py`: baselines/benches/toy; PENDING != FAIL; out-of-range
  only via range-author ruling (malformed fails closed; rule 5).
  HUMAN-ONLY (rule 5): kevin commits results/rulings/base05_val.ruling.json
  (paste: kevinruling.md Plan step 1, :58). attempt10 (w1a35) terminal wait:
  ruling absent (disk+all refs), gate exit 1, master unchanged since dcb59cf
  -> atk34 verification holds (paste->exit 0, 5/5 PASS; verdict exit 1,
  base05 0.0764 [0.014,0.153] <0.10; 16/16; w1a34-*.log). No agent-side
  step; per-attempt cost is status re-check only. Kevin: paste + hostgate (w2).

## Models
- `tiny_config` (2L/d64/V512); `tiny_tokenizer()` byte-BPE. GOTCHA: cfg vocab 151936 vs tok 151643.

## Env
- `exercism.py`: fresh tmpdir, pristine tests + ONLY solution, junit XML, pgroup SIGKILL.
  `lgbox.py`: RUNNERS["http"] iff LOOPGYM_LGBOX_URL.
  `splits.py`: sha1%100, 101/9/27,
  137/142 usable, val 9. `loopgym.py`: Task(prompt/parse/reference/stub)+grade over verified
  records; elided bad files (diff -U0 ±3, 4500-chr, <=6k tok); ```path blocks; grade = reply
  on bad worktree, pristine V._run_once (outside task.files -> 0); ref=1, stub=0, hidden=0.

## Kernel
- `kernels/token_logprobs(h,w,t,impl,chunk)->[T]: chunked max/LSE bwd softmax (no
  [T,V]); torch,cpp,triton,cuda; cpp JIT (-O3 -fopenmp); triton iff CUDA or
  TRITON_INTERPRET=1, C<=1024; cuda host-only. bench_logprob.py -> bench_logprob.json, f64 1e-4.

## RL
- `grpo.py` group_advantages/kl_k3/policy_loss; 7/7 known.
- `batch.py`: grpo_step (logp=token_logprobs; old=detach; ref=disable_adapter);
  step-0 β=0 ~0; no last_hidden_state in 5.x; kernel unbatched -> flatten.
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
- `freeze.py`: v0 = 26 verified tasks (draftlab 22/harness 4), sha1%100>=70
  train 13/test 13; data/loopgym/v0/ + results/loopgym_v0.json; `--check`
  exit 1 on drift; redaction: public draftlab/harness only.
- `data/verify.py`: 3x/side pristine worktrees -> verified.jsonl; tests=0 -> error.
- hidden checks: temp root + synthetic fixtures; assert completion claim;
  cross-check cited numbers vs results/*.json; GOTCHA: results/*.json absent
  at bad -> lazy discovery in tests (module level = collection error).
