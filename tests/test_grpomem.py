"""GRPO step memory fix: row micro-batching must equal one big batch.

Known answer: grpo_step with micro_batch=1 (gradient accumulation) gives the same
loss, stats and per-parameter gradients as the single big-batch step within 1e-5
(tiny_config, fp32, CPU), for token-mean and seq-mean, with and without the KL term.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2ForCausalLM

from loopgym.models import tiny_config
from loopgym.rl import batch as B
from loopgym.rl.grpo import policy_loss
from loopgym.rl.rollout import Completion

TOK = type("T", (), {"pad_token_id": 0})


def make_policy(seed=11):
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(tiny_config())
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    return get_peft_model(m, cfg)


def make_batch():
    g = torch.Generator().manual_seed(7)
    rows = []
    for i in range(6):
        p = torch.randint(1, 100, (4 + i % 3,), generator=g).tolist()
        c = torch.randint(1, 100, (3 + i % 4,), generator=g).tolist()
        rows.append((p, c))
    groups = [[Completion(prompt_ids=list(p), completion_ids=list(c), text="")
               for p, c in rows[0:3]],
              [Completion(prompt_ids=list(p), completion_ids=list(c), text="")
               for p, c in rows[3:6]]]
    return B.build_batch(groups, [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], TOK)


def run_once(micro_batch, kl, agg):
    policy = make_policy()
    batch = make_batch()
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
    cfg = B.StepCfg(kl_coef=kl, agg=agg, micro_batch=micro_batch)
    grads = {}
    real_step = opt.step

    def capture():
        for name, p in policy.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[name] = p.grad.detach().clone()
        real_step()

    opt.step = capture
    stats = B.grpo_step(policy, None, batch, opt, cfg)
    return stats, grads


def test_microbatch_grads_match_big_batch():
    for kl, agg in ((0.1, "token-mean"), (0.1, "seq-mean"), (0.0, "token-mean")):
        ref_stats, ref_grads = run_once(micro_batch=64, kl=kl, agg=agg)
        mb_stats, mb_grads = run_once(micro_batch=1, kl=kl, agg=agg)
        assert abs(ref_stats["loss"] - mb_stats["loss"]) < 1e-5, (kl, agg, ref_stats, mb_stats)
        for k in ("kl", "clip_frac", "ratio_mean", "tokens"):
            assert abs(ref_stats[k] - mb_stats[k]) < 1e-5, (k, kl, agg, ref_stats[k], mb_stats[k])
        assert set(ref_grads) == set(mb_grads)
        for name in ref_grads:
            assert torch.allclose(ref_grads[name], mb_grads[name], atol=1e-5, rtol=1e-5), name


def test_policy_loss_denom_is_global_token_mean():
    # chunked losses with the global mask denominator sum to the full-batch loss
    torch.manual_seed(0)
    n, t = 6, 5
    logp = torch.randn(n, t)
    old = logp + 0.01 * torch.randn(n, t)
    ref = logp + 0.02 * torch.randn(n, t)
    adv = torch.randn(n)
    mask = (torch.rand(n, t) > 0.3).float()
    full, _ = policy_loss(logp, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.1)
    total = mask.sum()
    parts = torch.zeros(())
    for i in range(0, n, 2):
        part, _ = policy_loss(logp[i:i + 2], old[i:i + 2], ref[i:i + 2], adv[i:i + 2], mask[i:i + 2],
                              clip_eps=0.2, kl_coef=0.1, denom=total)
        parts = parts + part
    assert torch.allclose(full, parts, atol=1e-6)


def test_logp_of_microbatch_equals_single_forward():
    policy = make_policy()
    batch = make_batch()
    a = B.logp_of(policy, batch, B.StepCfg(micro_batch=64))
    b = B.logp_of(policy, batch, B.StepCfg(micro_batch=2))
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
