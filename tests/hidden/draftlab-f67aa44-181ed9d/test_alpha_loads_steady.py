"""Hidden check, draftlab-f67aa44-181ed9d.

alpha: --lora-adapter must load a PEFT adapter (bad: nonexistent
PeftModel.from_model); JaxLM.forward_pass must measure steady-state
cost: one JIT signature per run (step recompiles per (T, max_len)) and
causal attention keeps padded short-block logits equal to unpadded.
"""
import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from draftlab.eval.alpha import JaxLM, TorchLM


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=128)
    Qwen2ForCausalLM(cfg).save_pretrained(d, safe_serialization=True)
    return d


@pytest.fixture(scope="module")
def adapter_dir(model_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("lora")
    m = Qwen2ForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    get_peft_model(m, LoraConfig(r=2, target_modules=["q_proj"])).save_pretrained(d)
    return d


def test_torch_lm_lora_adapter_loads(model_dir, adapter_dir):
    lm = TorchLM.from_pretrained(model_dir, dtype=torch.float32, lora_adapter=adapter_dir)
    assert "default" in lm.model.active_adapters
    out, _ = lm.forward_pass([[1, 2, 3]])
    assert out[0].shape == (3, 64)


def test_jaxlm_single_signature_steady(model_dir):
    lm = JaxLM(model_dir)
    M, orig = lm.M, lm.M.prefill
    lens = []

    def spy(c, p, ids, ml, *a, **kw):
        lens.append(int(np.asarray(ids).shape[0]))
        return orig(c, p, ids, ml, *a, **kw)

    M.prefill = spy
    try:
        out, _ = lm.forward_pass([list(range(5)), list(range(5, 14))])
    finally:
        M.prefill = orig
    assert lens, "forward_pass made no prefill calls"
    assert len(set(lens)) == 1, f"timed pass recompiles per length: {lens}"
    ref, _ = orig(lm.cfg, lm.params, list(range(5)), 5)
    assert np.allclose(out[0], np.asarray(ref), atol=1e-5), "padding changed short-block logits"
