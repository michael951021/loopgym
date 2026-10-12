"""FSDP2 (fully_shard) wrappers: init, wrap, rank helpers, full state dict, global loss."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard


def backend_for() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_dist(backend: str | None = None) -> None:
    """Init the process group: torchrun's env:// when WORLD_SIZE>1, else a local world-1 group."""
    if dist.is_initialized():
        return
    backend = backend or backend_for()
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        import tempfile

        store = os.path.join(tempfile.gettempdir(), f"loopgym-fsdp-{os.getpid()}.store")
        dist.init_process_group(
            backend=backend, init_method=f"file://{store}", rank=0, world_size=1
        )
        return
    kwargs: dict = {}
    if backend == "nccl":
        kwargs["device_id"] = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=backend, **kwargs)


def wrap_fsdp(model, cfg=None, layers=None):
    """fully_shard the Qwen2 backbone bottom-up under a Qwen2ForCausalLM root; returns model.

    The drill from `model` (a PeftModel) reaches the Qwen2Model (`base`, owns embed_tokens,
    layers, norm); its parent Qwen2ForCausalLM (`for_causal`, owns lm_head) is the FSDP root.
    Each decoder layer plus embed_tokens / norm / lm_head are leaf units, and the root is
    sharded last (children before parent). The root forward must be reachable by BOTH
    `generate()` (HF runs Qwen2ForCausalLM.forward per step) and the training forward, so
    its params unshard on either path; the root is what lets save() gather the sharded
    state (`full_tensor` / `get_model_state_dict` require a coordinating root)."""
    init_dist(getattr(cfg, "backend", None))
    base = model
    while not hasattr(base, "layers"):
        base = base.model
    if layers is None:
        layers = base.layers
    # parent of `base` that owns lm_head = the Qwen2ForCausalLM root
    for_causal = next(m for m in model.modules()
                      if getattr(m, "lm_head", None) is not None
                      and getattr(m, "model", None) is base)
    if hasattr(base, "embed_tokens"):
        fully_shard(base.embed_tokens)
    for layer in layers:
        fully_shard(layer)
    if hasattr(base, "norm"):
        fully_shard(base.norm)
    fully_shard(for_causal.lm_head)
    fully_shard(for_causal)  # root last
    return model


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_rank0() -> bool:
    return rank() == 0


def full_state_dict(model) -> dict:
    """Collective (all ranks must call); only rank 0 should persist the result."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from torch.distributed.tensor import DTensor

    sd = get_model_state_dict(model)
    return {k: (v.full_tensor() if isinstance(v, DTensor) else v) for k, v in sd.items()}


def global_mean_loss(local_sum: torch.Tensor, local_count: float) -> torch.Tensor:
    """All-reduce (sum, count) so every rank reports the same batch-mean loss."""
    if not dist.is_initialized():
        return local_sum / local_count
    t = torch.stack([local_sum.detach().to(torch.float32), torch.tensor(float(local_count))])
    dist.all_reduce(t)
    return t[0] / t[1]


def global_token_count(local: float) -> float:
    """All-reduce the local token count; 1.0 when not distributed."""
    if not dist.is_initialized():
        return float(local)
    t = torch.tensor(float(local))
    dist.all_reduce(t)
    return float(t.item())
