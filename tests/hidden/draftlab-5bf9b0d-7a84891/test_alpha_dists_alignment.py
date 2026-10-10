"""Hidden check, draftlab-5bf9b0d-7a84891.

dists must score ids[j] with row len(ctx)+j-1 (row i predicts token i+1); bad
slices row len(ctx)+j (off by one); empty ctx: position 0 invalid. Known answer:
rows equal softmax of raw HF logits at the correct row. Tiny HF Qwen2, CPU.
"""
import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from draftlab.eval.alpha import HFDrafter, JaxModel


def _sm(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny")
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=128)
    Qwen2ForCausalLM(cfg).save_pretrained(d, safe_serialization=True)
    return str(d)


def _gold(tiny, ctx, ids):
    m = Qwen2ForCausalLM.from_pretrained(tiny, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        L = m(input_ids=torch.tensor([ctx + ids])).logits[0].float().numpy()
    return np.stack([_sm(L[len(ctx) - 1 + j]) for j in range(len(ids))])


def test_hf_dists_alignment(tiny):
    ctx, ids = list(range(3, 9)), list(range(20, 28))
    q, valid = HFDrafter(tiny).dists(ctx, ids)
    assert valid.all()
    assert np.allclose(q, _gold(tiny, ctx, ids), atol=1e-5)


def test_hf_dists_empty_ctx(tiny):
    ids = list(range(4))
    q, valid = HFDrafter(tiny).dists([], ids)
    assert not valid[0], "position 0 has no context row; must be invalid"
    assert valid[1:].all()
    m = Qwen2ForCausalLM.from_pretrained(tiny, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        L = m(torch.tensor([ids])).logits[0].float().numpy()
    assert np.allclose(q[1:], np.stack([_sm(L[j - 1]) for j in range(1, 4)]), atol=1e-5)


def test_jax_model_alignment(tiny):
    ctx, ids = list(range(3, 9)), list(range(20, 28))
    q, valid = JaxModel.from_path(tiny, max_len=64).dists(ctx, ids)
    assert valid.all()
    assert np.allclose(q, _gold(tiny, ctx, ids), atol=1e-4)
