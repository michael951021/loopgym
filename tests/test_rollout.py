"""HF rollout backend on the real Qwen2.5-Coder-0.5B (fp32).

Known answer (CLAUDE.md): at temperature 1 the sampler's per-token log-prob equals a fresh
forward through kernels.token_logprobs within 1e-4. Greedy must equal model.generate;
left padding must not change a prompt's completion."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from loopgym import kernels
from loopgym.rl.rollout import HFBackend

MODEL = os.environ.get("LOOPGYM_MODEL", "models/qwen2.5-coder-0.5b-instruct")

# The policy model is a shared host asset (gitignored); skip on a fresh clone.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL), reason=f"no model at {MODEL} (fresh clone)"
)

PROMPTS = [
    [{"role": "user", "content": "Reply with one word: banana."}],
    [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function `fibs(n)` that returns the first n Fibonacci numbers as a list. Include a docstring and a `if __name__ == '__main__':` demo that prints fibs(10)."},
    ],
]


@pytest.fixture(scope="module")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def backend(model, tokenizer):
    return HFBackend(model, tokenizer)


def test_sampler_logp_matches_fresh_forward(backend, model, tokenizer):
    groups = backend.generate(PROMPTS, n=2, max_new_tokens=16, temperature=1.0, top_p=1.0, seed=1234)
    assert [len(g) for g in groups] == [2, 2]
    worst = 0.0
    for group, messages in zip(groups, PROMPTS):
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True).input_ids
        for c in group:
            assert c.prompt_ids == prompt_ids
            k = len(c.completion_ids)
            assert 1 <= k <= 16 and len(c.sampler_logp) == k
            ids = torch.tensor(prompt_ids + c.completion_ids).unsqueeze(0)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            # rows len(prompt)-1 .. len(prompt)+k-2 predict the k completion tokens
            h = out.hidden_states[-1][0, len(prompt_ids) - 1 : len(prompt_ids) + k - 1]
            ref = kernels.token_logprobs(h, model.get_output_embeddings().weight, torch.tensor(c.completion_ids))
            gap = max(abs(a - b) for a, b in zip(c.sampler_logp, ref.tolist()))
            worst = max(worst, gap)
    assert worst <= 1e-4, f"max |sampler_logp - fresh forward| = {worst:.2e} > 1e-4"


def test_greedy_path_equals_model_generate(backend, model, tokenizer):
    groups = backend.generate([PROMPTS[1]], n=1, max_new_tokens=16, temperature=0.0, top_p=1.0, seed=0)
    c = groups[0][0]
    prompt_ids = tokenizer.apply_chat_template(PROMPTS[1], tokenize=True, add_generation_prompt=True).input_ids
    with torch.no_grad():
        ref = model.generate(torch.tensor([prompt_ids]), max_new_tokens=16)
    ref = ref[0, len(prompt_ids):].tolist()
    eos = tokenizer.eos_token_id
    if eos in ref:
        ref = ref[: ref.index(eos) + 1]
    assert c.completion_ids == ref


def test_padding_does_not_change_completion(backend):
    alone = backend.generate([PROMPTS[0]], n=1, max_new_tokens=16, temperature=0.0, top_p=1.0, seed=0)
    batched = backend.generate([PROMPTS[1], PROMPTS[0]], n=1, max_new_tokens=16, temperature=0.0, top_p=1.0, seed=0)
    assert batched[1][0].completion_ids == alone[0][0].completion_ids
    assert batched[1][0].text == alone[0][0].text
