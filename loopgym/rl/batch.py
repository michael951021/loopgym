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
    micro_batch: int = 8  # rows per policy/ref forward; step = gradient accumulation


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


def _lang_model(model):
    """Base Qwen2Model (final norm, no lm_head) so log-probs never form [N, T, vocab] logits."""
    m = model
    while hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return getattr(m, "model", m)


def _head_weight(model):
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return out.weight if out is not None else _lang_model(model).embed_tokens.weight


def _logp_rows(hidden, W, targets, cfg):
    logp = kernels.token_logprobs(
        hidden.reshape(-1, hidden.size(-1)),
        W,
        targets.reshape(-1),
        impl=cfg.logprob_impl,
        chunk=cfg.chunk,
    )
    return logp.reshape(targets.shape)


def logp_of(model, batch, cfg, engine=None):
    """Per-column log-prob of the next token under `model`, [N, T-1].

    Forwards the base model (lm_head skipped) in row chunks of cfg.micro_batch and
    takes log-probs via the chunked kernel, which never materializes [rows, T, vocab].
    FSDP2: the sharded head weight is materialized once per call with full_tensor()
    (a collective - every rank calls logp_of the same number of times). ZeRO-3
    (`engine`): the head is a (0,) shard the base forward never touches, so each chunk
    re-gathers it just for the kernel run; the gathered weight needs no grad (frozen
    lm_head, hidden independent of it)."""
    dev = next(model.parameters()).device
    base = _lang_model(model)
    W = _head_weight(model)
    if isinstance(W, torch.distributed.tensor.DTensor):
        W = W.full_tensor()
    head = None
    if engine is not None:
        # the pinned source exposes `zero` as an attribute alias (from .runtime import
        # zero), not a registered submodule, so `from deepspeed.zero import ...` fails
        import deepspeed
        head = engine.get_output_embeddings().weight
    n = batch.input_ids.shape[0]
    mb = max(1, min(int(cfg.micro_batch), n))
    out = []
    for i in range(0, n, mb):
        ids = batch.input_ids[i:i + mb].to(dev)
        am = batch.attention_mask[i:i + mb].to(dev)
        hidden = base(input_ids=ids, attention_mask=am).last_hidden_state[:, :-1]
        targets = ids[:, 1:]
        if engine is None:
            out.append(_logp_rows(hidden, W, targets, cfg))
        else:
            with deepspeed.zero.GatheredParameters([head]):
                out.append(_logp_rows(hidden, head, targets, cfg))
    return torch.cat(out, dim=0)


def _sub_batch(batch, i, j):
    return Batch(batch.input_ids[i:j], batch.attention_mask[i:j],
                 batch.target_mask[i:j], batch.advantages[i:j])


def _step_fini(policy, batch, optimizer, cfg, loss_val, sums, engine=None):
    tokens = int(batch.target_mask.sum())
    if engine is not None:
        engine.step()
        gn = engine.get_global_grad_norm()
        grad_norm = 0.0 if gn is None else float(gn)
    else:
        grad_norm = clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                    cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_val,
        "kl": sums["kl"] / tokens,
        "clip_frac": sums["clip_frac"] / tokens,
        "ratio_mean": sums["ratio_mean"] / tokens,
        "grad_norm": float(grad_norm),
        "tokens": tokens,
    }


def _ref_logp(policy, ref_fn, logp, sub, cfg, engine):
    if ref_fn is not None:
        return ref_fn(policy, sub)
    if cfg.kl_coef == 0.0:
        # no KL term: the ref forward is pure waste (and a full re-gather under ZeRO-3)
        return logp
    with policy.disable_adapter(), torch.no_grad():
        return logp_of(policy, sub, cfg, engine=engine)


def _step_single(policy, ref_fn, batch, optimizer, cfg, loss_factor=1.0, engine=None):
    dev = next(policy.parameters()).device
    logp = logp_of(policy, batch, cfg, engine=engine)
    ref_logp = _ref_logp(policy, ref_fn, logp, batch, cfg, engine)
    loss, stats = grpo.policy_loss(
        logp, logp.detach(), ref_logp, batch.advantages.to(dev), batch.target_mask.to(dev),
        clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef, agg=cfg.agg,
    )
    if engine is not None:
        engine.backward(loss * loss_factor)
    else:
        (loss * loss_factor).backward()
    m = float(batch.target_mask.sum())
    sums = {k: float(v.detach()) * m for k, v in stats.items()}
    return _step_fini(policy, batch, optimizer, cfg, float(loss.detach()), sums, engine)


def grpo_step(policy, ref_fn, batch, optimizer, cfg, loss_factor=1.0, engine=None):
    """One GRPO optimizer step. `policy` is a PeftModel; `ref_fn` (policy, batch) -> [N, T-1]
    log-probs under the reference, or None to use the base model via policy.disable_adapter().
    old_logp = logp.detach() (on-policy, first epoch). With micro_batch < N the policy and
    ref forwards run per row chunk with gradient accumulation: the step is one big-batch
    step (token-mean uses the global mask denominator; seq-mean weights chunks by rows).
    `loss_factor` scales only the backwards (world * N_local / N_global turns the AVG
    reduce of per-rank token-mean losses into one global token-mean step, FSDP2 and
    ZeRO-3 alike). `engine` (a DeepSpeed ZeRO-3 engine around policy): forwards and the
    head gather go through it, clipping comes from the engine's gradient_clipping, and
    the step/grad-norm come from the engine. Returns stats: loss, kl, clip_frac,
    ratio_mean, grad_norm (pre-clip), tokens."""
    policy.train()
    n = batch.input_ids.shape[0]
    mb = max(1, min(int(cfg.micro_batch), n))
    if mb >= n or float(batch.target_mask.sum()) == 0.0:
        return _step_single(policy, ref_fn, batch, optimizer, cfg, loss_factor, engine)
    dev = next(policy.parameters()).device
    total = float(batch.target_mask.sum())
    loss_val = 0.0
    sums = {"kl": 0.0, "clip_frac": 0.0, "ratio_mean": 0.0}
    for i in range(0, n, mb):
        sub = _sub_batch(batch, i, i + mb)
        m_c = float(sub.target_mask.sum())
        if m_c == 0.0:
            continue
        logp = logp_of(policy, sub, cfg, engine=engine)
        ref_logp = _ref_logp(policy, ref_fn, logp, sub, cfg, engine)
        loss_c, stats_c = grpo.policy_loss(
            logp, logp.detach(), ref_logp, sub.advantages.to(dev), sub.target_mask.to(dev),
            clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef, agg=cfg.agg,
            denom=total if cfg.agg == "token-mean" else None,
        )
        scale = 1.0 if cfg.agg == "token-mean" else sub.target_mask.shape[0] / n
        if engine is not None:
            engine.backward(loss_c * scale * loss_factor)
        else:
            (loss_c * scale * loss_factor).backward()
        loss_val += float(loss_c.detach()) * scale
        for k in sums:
            sums[k] += float(stats_c[k].detach()) * m_c
    return _step_fini(policy, batch, optimizer, cfg, loss_val, sums, engine)
