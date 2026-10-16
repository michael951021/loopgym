"""grad_ckpt recomputes activations in backward; the LoRA gradients must not change."""
import torch
from peft import LoraConfig, get_peft_model

from loopgym.rl import train as T


def _lora_grads(grad_ckpt: bool):
    cfg = T.TrainCfg(model="tiny", grad_ckpt=grad_ckpt)
    policy = get_peft_model(T.build_base(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    for n, p in policy.named_parameters():
        if "lora_B" in n:  # B starts at zero; make the adapter non-trivial
            torch.nn.init.normal_(p, std=0.02, generator=torch.Generator().manual_seed(0))
    policy.train()
    ids = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(1))
    policy(input_ids=ids, labels=ids).loss.backward()
    return {n: p.grad.clone() for n, p in policy.named_parameters() if p.requires_grad}


def test_grad_ckpt_gives_the_same_lora_gradients():
    a, b = _lora_grads(False), _lora_grads(True)
    assert a.keys() == b.keys() and a
    for n in a:
        assert torch.allclose(a[n], b[n], atol=1e-6), n
