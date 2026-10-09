"""GRPO math (loopgym/rl/grpo.py). Definitions are in CLAUDE.md "Key definitions"."""
import math

import torch

from loopgym.rl import grpo


def test_group_advantages_hand_computed():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [3.0, 1.0, 2.0, 2.0]])
    a = grpo.group_advantages(r)                      # (r - mean) / (sample std + eps), per group (row)
    s = math.sqrt(1 / 3)                               # sample std of [1, 0, 0, 1]
    assert torch.allclose(a[0], torch.tensor([0.5, -0.5, -0.5, 0.5]) / s, atol=1e-4)
    assert torch.allclose(a[1], torch.tensor([1.0, -1.0, 0.0, 0.0]) / math.sqrt(2 / 3), atol=1e-4)
    assert torch.allclose(grpo.group_advantages(r, scale="none")[0], torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_constant_group_has_zero_advantage_not_nan():
    a = grpo.group_advantages(torch.ones(2, 8))
    assert torch.isfinite(a).all() and a.abs().max() == 0


def test_k3_is_zero_for_identical_policies_and_positive_otherwise():
    lp = torch.log(torch.tensor([0.2, 0.5, 0.3]))
    assert grpo.kl_k3(lp, lp).abs().max() < 1e-7
    assert (grpo.kl_k3(lp, lp.flip(0)) >= 0).all()


def test_k3_is_unbiased_for_kl():
    """Mean of k3 over samples x ~ p estimates KL(p || ref)."""
    g = torch.Generator().manual_seed(0)
    p, ref = torch.tensor([0.6, 0.3, 0.1]), torch.tensor([0.2, 0.3, 0.5])
    x = torch.multinomial(p, 200_000, replacement=True, generator=g)
    est = grpo.kl_k3(p.log()[x], ref.log()[x])
    kl = float((p * (p / ref).log()).sum())
    assert abs(float(est.mean()) - kl) < 4 * float(est.std()) / math.sqrt(len(x))


def test_policy_loss_on_policy_equals_minus_mean_advantage():
    logp = torch.log(torch.tensor([[0.5, 0.4, 0.9], [0.2, 0.7, 0.1]]))
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    adv = torch.tensor([2.0, -1.0])
    loss, _ = grpo.policy_loss(logp, logp.clone(), logp.clone(), adv, mask)
    assert abs(float(loss) - (-(2 * 2 + 3 * -1) / 5)) < 1e-6           # token-mean over the 5 unmasked tokens
    loss_seq, _ = grpo.policy_loss(logp, logp.clone(), logp.clone(), adv, mask, agg="seq-mean")
    assert abs(float(loss_seq) - (-(2 + -1) / 2)) < 1e-6


def test_clipped_token_gets_no_gradient():
    old = torch.log(torch.tensor([[0.5, 0.5]]))
    logp = torch.log(torch.tensor([[0.5 * 1.5, 0.5 * 1.1]])).requires_grad_()   # ratios 1.5 (clipped) and 1.1
    loss, stats = grpo.policy_loss(logp, old, old, torch.tensor([1.0]), torch.ones(1, 2), clip_eps=0.2)
    loss.backward()
    assert logp.grad[0, 0] == 0 and logp.grad[0, 1] != 0
    assert abs(float(stats["clip_frac"]) - 0.5) < 1e-6


def test_kl_term_pulls_toward_reference():
    ref = torch.log(torch.tensor([[0.5]]))
    logp = torch.log(torch.tensor([[0.8]])).requires_grad_()
    loss, _ = grpo.policy_loss(logp, logp.detach(), ref, torch.tensor([0.0]), torch.ones(1, 1), kl_coef=0.1)
    loss.backward()
    assert float(loss) > 0 and logp.grad[0, 0] > 0      # gradient descent lowers logp toward ref
