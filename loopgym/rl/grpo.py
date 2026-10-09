"""GRPO math. Definitions: CLAUDE.md "Key definitions"; pinned by tests/known/test_grpo.py."""
import torch


def group_advantages(rewards, scale="std", eps=1e-6):
    """rewards [B, G] -> per-group (r - mean) / (sample_std + eps); scale="none" skips the division."""
    mean = rewards.mean(dim=-1, keepdim=True)
    centered = rewards - mean
    if scale == "none":
        return centered
    std = rewards.std(dim=-1, keepdim=True, unbiased=True)  # Bessel, as TRL
    return centered / (std + eps)


def kl_k3(logp, ref_logp):
    """k3 per-token KL estimator: d = ref_logp - logp, k3 = exp(d) - d - 1 (>= 0)."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1


def policy_loss(logp, old_logp, ref_logp, adv, mask, clip_eps=0.2, kl_coef=0.0, agg="token-mean"):
    """GRPO/PPO objective.

    logp/old_logp/ref_logp [B, T] (old/ref from the trainer's forward, never the sampler);
    adv [B] (per prompt); mask [B, T]. Returns (loss, {"clip_frac", "kl", "ratio_mean"}).
    """
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)
    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    a = adv.float()
    obj = torch.min(ratio * a, clipped * a) - kl_coef * kl_k3(logp, ref_logp)
    if agg == "token-mean":
        loss = -(obj * mask).sum() / mask.sum()
    elif agg == "seq-mean":
        denom = mask.sum(dim=-1).clamp_min(1)
        loss = -((obj * mask).sum(dim=-1) / denom).mean()
    else:
        raise ValueError(f"unknown agg: {agg!r}")
    m = mask.sum()
    # effective clipping: min() picked the frozen branch (sign-aware, ties to PPO semantics)
    clipped_tok = (a > 0) & (ratio > 1.0 + clip_eps)
    clipped_tok |= (a < 0) & (ratio < 1.0 - clip_eps)
    stats = {
        "clip_frac": (clipped_tok * mask).sum() / m,
        "kl": (kl_k3(logp, ref_logp) * mask).sum() / m,
        "ratio_mean": (ratio * mask).sum() / m,
    }
    return loss, stats
