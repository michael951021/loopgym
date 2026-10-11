"""One GRPO step from completions: build the padded batch, then the step itself.

Row i of the batch is one completion (prompt + completion, right-padded). Column j of
`target_mask` is 1 iff column j predicts a completion token (position j+1), i.e. the mask
covers exactly the completion tokens and no prompt/pad token.
"""
from dataclasses import dataclass

import torch
from torch.nn.utils import clip_grad_norm_

from loopgym import kernels
from loopgym.rl import grpo

__all__ = ["Batch", "StepCfg", "build_batch", "logp_of", "grpo_step"]


@dataclass
class Batch:
    input_ids: torch.Tensor      # [N, T] right-padded prompt+completion ids
    attention_mask: torch.Tensor  # [N, T] 1 on real tokens
    target_mask: torch.Tensor     # [N, T-1] 1 where column predicts a completion token
    advantages: torch.Tensor      # [N] per completion, from grpo.group_advantages


@dataclass
class StepCfg:
    logprob_impl: str = "torch"
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    agg: str = "token-mean"
    max_grad_norm: float = 1.0
    chunk: int = 8192


def build_batch(groups, rewards, tokenizer, scale="std"):
    """groups: list[B] of list[G] of Completion (rollout output order); rewards: [B, G].

    Right-pads to the longest prompt+completion; returns a Batch whose rows follow the
    (b, g) order of `groups`."""
    adv = grpo.group_advantages(torch.as_tensor(rewards, dtype=torch.float32), scale=scale)
    if adv.dim() != 2 or len(groups) != adv.shape[0]:
        raise ValueError(f"rewards shape {tuple(adv.shape)} does not match {len(groups)} groups")
    rows = []
    for b, group in enumerate(groups):
        if len(group) != adv.shape[1]:
            raise ValueError(f"group {b} has {len(group)} completions, rewards row has {adv.shape[1]}")
        for c in group:
            rows.append((list(c.prompt_ids) + list(c.completion_ids), len(c.prompt_ids)))
    width = max(len(ids) for ids, _ in rows)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
    target_mask = torch.zeros(len(rows), width - 1, dtype=torch.float32)
    for i, (ids, p) in enumerate(rows):
        c = len(ids) - p
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        if c > 0:
            # completion tokens sit at positions p..p+c-1, predicted by columns p-1..p+c-2
            target_mask[i, p - 1 : p + c - 1] = 1.0
    return Batch(input_ids, attention_mask, target_mask, adv.reshape(-1))


def logp_of(model, batch, cfg):
    """Per-column log-prob of the next token under `model`, [N, T-1], via the chunked kernel."""
    dev = next(model.parameters()).device
    input_ids = batch.input_ids.to(dev)
    attention_mask = batch.attention_mask.to(dev)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden = out.hidden_states[-1][:, :-1]
    targets = input_ids[:, 1:]
    n, t = targets.shape
    w = model.get_output_embeddings().weight
    # The kernel indexes weight rows by global vocab id; FSDP2 shards the weight along
    # dim 0, so materialize the full tensor before it reaches the kernel.
    if isinstance(w, torch.distributed.tensor.DTensor):
        w = w.full_tensor()
    logp = kernels.token_logprobs(
        hidden.reshape(-1, hidden.size(-1)),
        w,
        targets.reshape(-1),
        impl=cfg.logprob_impl,
        chunk=cfg.chunk,
    )
    return logp.reshape(n, t)


def _gathered_logp(engine, batch, cfg):
    """logp_of on a ZeRO-3 engine: materialize the partitioned lm_head weight for the kernel
    (the engine's forward hooks release it again after the forward)."""
    from deepspeed.zero import GatheredParameters

    with GatheredParameters([engine.get_output_embeddings().weight]):
        return logp_of(engine, batch, cfg)


def grpo_step(policy, ref_fn, batch, optimizer, cfg, loss_factor=1.0, engine=None):
    """One GRPO optimizer step. `policy` is a PeftModel; `ref_fn` (policy, batch) -> [N, T-1]
    log-probs under the reference, or None to use the base model via policy.disable_adapter().
    old_logp = logp.detach() (on-policy, first epoch). Returns stats: loss, kl, clip_frac,
    ratio_mean, grad_norm (pre-clip), tokens (the loss is the per-rank local mean).

    `loss_factor` scales only the backward (world * N_local / N_global turns the AVG
    reduce-scatter of per-rank token-mean losses into one global token-mean step).
    `engine` (a DeepSpeed ZeRO-3 engine around policy): forward and the weight gather go
    through the engine, clip at cfg.max_grad_norm comes from the engine's gradient_clipping,
    and the reference forward is skipped when kl_coef == 0."""
    fwd = engine if engine is not None else policy
    fwd.train()
    logp = _gathered_logp(fwd, batch, cfg) if engine is not None else logp_of(fwd, batch, cfg)
    old_logp = logp.detach()
    if ref_fn is not None:
        ref_logp = ref_fn(policy, batch)
    elif engine is not None and cfg.kl_coef == 0.0:
        ref_logp = logp
    else:
        with policy.disable_adapter(), torch.no_grad():
            ref_logp = _gathered_logp(fwd, batch, cfg) if engine is not None else logp_of(policy, batch, cfg)
    dev = logp.device
    loss, stats = grpo.policy_loss(
        logp, old_logp, ref_logp, batch.advantages.to(dev), batch.target_mask.to(dev),
        clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef, agg=cfg.agg,
    )
    if engine is not None:
        engine.backward(loss * loss_factor)
        engine.step()
        gn = engine.get_global_grad_norm()
        grad_norm = 0.0 if gn is None else float(gn)
    else:
        (loss * loss_factor).backward()
        grad_norm = clip_grad_norm_([p for p in fwd.parameters() if p.requires_grad], cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "kl": float(stats["kl"].detach()),
        "clip_frac": float(stats["clip_frac"].detach()),
        "ratio_mean": float(stats["ratio_mean"].detach()),
        "grad_norm": float(grad_norm),
        "tokens": int(batch.target_mask.sum()),
    }
