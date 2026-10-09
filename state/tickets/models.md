# models — Model downloads

## Issue
Every later ticket (rollouts, kernels, GRPO, eval) loads the Qwen2.5-Coder 0.5B/1.5B weights and
tokenizer from `models/`. Nothing provides them, and unit tests need a tiny random Qwen2 config
that builds in milliseconds. No `loopgym/models.py` exists at all.

## Question
Does `loopgym/models.py` provide `fetch` (downloads both instruct models, safetensors + tokenizer
only), `info` (prints params/dtype/vocab, asserts the two tokenizers are identical), and
`tiny_config()` (2L/d64/4H/2KV/V512 Qwen2) — such that on a shared bind-mounted `models/`,
`python -m loopgym.models info` lists both at vocab 151936?
Refuting observation: info lists a model with a different vocab, the tokenizer assertion fails on
a clean fetch, or tiny_config is not a Qwen2Config with the pinned shapes.

## Evidence
- Observed: `models/` bind mount is empty; huggingface_hub 1.33.0 + transformers 5.19.0 + torch
  2.14.1+cpu in .venv; HF reachable (307 on config.json resolve).
- tests/known/known_paths.py:6 pins `models/qwen2.5-coder-0.5b-instruct`; conftest.py:16 says fetch
  is `python -m loopgym.models fetch`.
- Failing check (run first): `.venv/bin/python -m loopgym.models info` → ModuleNotFoundError /
  "No module named loopgym.models". Log: .agent-evidence/models-accept.log.
- Guess to verify: Qwen2.5-Coder safetensors dtype (expected fp32 storage) — read from the
  safetensors header, do not assume.

## Done when
On the real inputs: `.venv/bin/python -m loopgym.models fetch` exits 0 (both repos), then
`.venv/bin/python -m loopgym.models info` exits 0 and prints both models with vocab 151936 and an
explicit "tokenizers identical" assertion pass. `tiny_config()` returns Qwen2Config(2 layers,
hidden 64, 4 heads, 2 KV heads, vocab 512) — pinned by a new unit test in tests/. Known answers:
vocab 151936 (the model config's vocab_size = embedding dim, published); tokenizer identity via
byte-equal tokenizer files; param counts cross-checked against the safetensors header, not config
math. Plausible ranges: 0.5B model file ≈1 GB, 1.5B ≈3 GB; info must finish < 1 min (no full model
load — safetensors header only).

### Result (2026-10-09)
- Fetch: both repos, 407 s total (.agent-evidence/models-fetch.log). 0.5B single
  model.safetensors (954 MB), 1.5B single (3.09 GB).
- Info: `0.5b: params=494,032,768 dtypes=['BF16'] vocab=151936`; `1.5b: params=1,543,714,304
  dtypes=['BF16'] vocab=151936`; tokenizers identical; 2.3 s (.agent-evidence/models-info2.log).
- Correction to the plan's guess: weights are BF16, not fp32.
- Vocab subtlety: tokenizer length is 151,643; 151,936 is the config vocab_size (embedding dim).
  info asserts the latter — the number training and kernels use.
- Independent load check: HF from_pretrained loads the 0.5B; live param count 494,032,768 equals
  the header count exactly. make check green (7 tests incl. new tests/test_models.py).

## Stop when
Not doing: building/loading model objects in tests, GPU, any training. If HF is unreachable or a
download is corrupt after one retry, record it in NOTES.md and stop (the fetch command stays for
the next run). Two attempts without new evidence → stop.

## Plan
1. Run failing acceptance check; save log.
2. loopgym/models.py: fetch (snapshot_download, allow safetensors + tokenizer files only),
   info (safetensors headers for params/dtype, AutoTokenizer vocab, byte-compare tokenizer files),
   tiny_config, CLI.
3. Unit test tests/test_models.py (tiny_config shapes; info on a stub dir).
4. Run fetch (0.5B then 1.5B) under capture.py, then the acceptance command.
5. make check; commit; update NOTES.md.
