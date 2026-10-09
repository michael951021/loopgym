"""One GRPO step from completions: mask correctness and step semantics on tiny_config (fp32, CPU)."""
import math
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2ForCausalLM

from loopgym.models import tiny_config
from loopgym.rl import batch as B
from loopgym.rl import grpo
from loopgym.rl.rollout import Completion

PAD = 0
TOK = SimpleNamespace(pad_token_id=PAD)


def comp(prompt, completion):
    return Completion(prompt_ids=list(prompt), completion_ids=list(completion), text="")


def make_policy(seed=0):
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(tiny_config())
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    policy = get_peft_model(m, cfg)
    policy.eval()
    return policy


def masked_mean_logp(model, b, row):
    """Mean log-prob of row's completion tokens via an independent forward + gather."""
    with torch.no_grad():
        logits = model(input_ids=b.input_ids, attention_mask=b.attention_mask).logits
    logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, b.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = b.target_mask[row].bool()
    return logp[row][m].mean().item(), int(m.sum())


def test_mask_covers_exactly_completion_tokens():
    # row layout: (prompt, completion) -> (p, c)
    rows = [((7, 8, 9, 10, 11, 12), (20, 21, 22)), ((7, 8, 9, 10, 11, 12), (30, 31, 32, 33)),
            ((41, 42, 43), (50, 51)), ((41, 42, 43), (60, 61, 62, 63, 64))]
    groups = [[comp(*rows[0]), comp(*rows[1])], [comp(*rows[2]), comp(*rows[3])]]
    rewards = [[0.0, 1.0], [1.0, 0.0]]
    b = B.build_batch(groups, rewards, TOK)

    width = max(len(p) + len(c) for p, c in rows)
    assert b.input_ids.shape == (4, width)
    assert b.target_mask.shape == (4, width - 1)
    assert b.advantages.shape == (4,)

    for i, (p, c) in enumerate(rows):
        plen, clen = len(p), len(c)
        assert b.input_ids[i, :plen + clen].tolist() == list(p) + list(c)
        assert b.input_ids[i, plen + clen:].tolist() == [PAD] * (width - plen - clen)
        exp = [0.0] * (plen - 1) + [1.0] * clen + [0.0] * (width - plen - clen)
        assert b.target_mask[i].tolist() == exp, f"row {i} mask off"
        assert b.attention_mask[i].tolist() == [1] * (plen + clen) + [0] * (width - plen - clen)

    assert torch.allclose(b.advantages, grpo.group_advantages(torch.tensor(rewards)).reshape(-1))

    # independent check: masked log-probs == log-likelihood of the completion tokens only
    model = make_policy(seed=1)
    for i, (_, c) in enumerate(rows):
        got, n = masked_mean_logp(model, b, i)
        with torch.no_grad():
            logits = model(input_ids=b.input_ids, attention_mask=b.attention_mask).logits
            logp = torch.log_softmax(logits[i, :], -1)
        a = len(rows[i][0]) - 1
        exp = logp[torch.arange(a, a + len(c)), torch.tensor(c)].mean().item()
        assert n == len(c)
        assert abs(got - exp) < 1e-5


def test_bad_reward_shape_raises():
    groups = [[comp((1, 2), (3,)), comp((1, 2), (4,))]]
    try:
        B.build_batch(groups, [[0.0, 1.0], [0.0]], TOK)
    except ValueError:
        return
    raise AssertionError("expected ValueError for reward/shape mismatch")


def test_on_policy_step0_loss_is_zero():
    # equal-length rows, per-group advantages sum to 0 -> token-mean loss must be ~0 at step 0
    p = (5, 6, 7, 8, 9, 10, 11, 12)
    groups = [[comp(p, (20, 21, 22)), comp(p, (30, 31, 32))], [comp(p, (40, 41, 42)), comp(p, (50, 51, 52))]]
    b = B.build_batch(groups, [[0.0, 1.0], [1.0, 0.0]], TOK)
    policy = make_policy(seed=2)
    opt = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-2)
    cfg = B.StepCfg(kl_coef=0.0)
    stats = B.grpo_step(policy, None, b, opt, cfg)
    assert abs(stats["loss"]) < 1e-6, stats
    assert stats["tokens"] == 4 * 3
    assert stats["clip_frac"] == 0.0
    assert abs(stats["ratio_mean"] - 1.0) < 1e-5
    assert stats["grad_norm"] > 0.0 and math.isfinite(stats["grad_norm"])


def test_step_moves_logp_in_advantage_direction_and_ref_disables_adapter():
    p = (5, 6, 7, 8, 9, 10, 11, 12)
    groups = [[comp(p, (20, 21, 22, 23)), comp(p, (30, 31, 32, 33))]]
    b = B.build_batch(groups, [[1.0, 0.0]], TOK)
    assert (b.advantages[0] > 0) and (b.advantages[1] < 0)
    policy = make_policy(seed=3)
    opt = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-2)
    cfg = B.StepCfg(kl_coef=0.0)

    pos_before, _ = masked_mean_logp(policy, b, 0)
    neg_before, _ = masked_mean_logp(policy, b, 1)
    stats = B.grpo_step(policy, None, b, opt, cfg)
    assert stats["grad_norm"] > 0.0 and math.isfinite(stats["loss"])
    pos_after, _ = masked_mean_logp(policy, b, 0)
    neg_after, _ = masked_mean_logp(policy, b, 1)
    assert pos_after > pos_before + 1e-4, (pos_before, pos_after)
    assert neg_after < neg_before - 1e-4, (neg_before, neg_after)

    # after a step the LoRA adapter is nonzero, so a ref forward with the adapter disabled
    # must differ from the policy's; k3 KL of the second step must be > 0
    stats2 = B.grpo_step(policy, None, b, opt, B.StepCfg(kl_coef=0.1))
    assert stats2["kl"] > 1e-6, stats2


def test_custom_ref_fn_is_used():
    p = (5, 6, 7, 8)
    groups = [[comp(p, (20, 21)), comp(p, (30, 31))]]
    b = B.build_batch(groups, [[1.0, 0.0]], TOK)
    policy = make_policy(seed=4)
    opt = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-2)
    calls = []

    cfg = B.StepCfg(kl_coef=0.1)

    def ref_fn(model, batch_):
        calls.append(1)
        with model.disable_adapter():
            return B.logp_of(model, batch_, cfg)

    stats = B.grpo_step(policy, ref_fn, b, opt, cfg)
    assert calls and stats["kl"] >= 0.0
