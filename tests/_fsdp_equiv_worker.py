"""torchrun worker: one FSDP2 step on 2 ranks x b vs one single-process step on 2b (tiny_config, fp32)."""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopgym.models import tiny_config
from loopgym.rl.dist import full_state_dict, global_mean_loss, rank, wrap_fsdp

B_PER_RANK = 2
T = 16
SEED = 1234
LR = 1e-3
ATOL = 1e-5


def make_model():
    torch.manual_seed(SEED)
    m = Qwen2ForCausalLM(tiny_config())
    m.config.use_cache = False
    m.train()
    return m


def make_ids():
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randint(0, 512, (2 * B_PER_RANK, T), generator=g)


def loss_sum(model, ids):
    logits = model(input_ids=ids).logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="sum"
    )


def main():
    ids = make_ids()
    count = float(B_PER_RANK * (T - 1))

    model = make_model()
    wrap_fsdp(model)  # inits the process group; rank() is only valid after this
    r = rank()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    s_raw = loss_sum(model, ids[r * B_PER_RANK : (r + 1) * B_PER_RANK])
    (s_raw / count).backward()  # per-token mean; AVG reduce-scatter = full-batch mean grad
    opt.step()
    reported = global_mean_loss(s_raw, count)

    # Reference: same seed -> same init; two b-row forwards mirror the ranks' forwards
    # so only FSDP's reduction (not MKL's batch-shape quirks) is under test.
    ref = make_model()
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=LR)
    ref_s = loss_sum(ref, ids[:B_PER_RANK]) + loss_sum(ref, ids[B_PER_RANK:])
    (ref_s / (2 * count)).backward()
    ref_opt.step()
    ref_loss = ref_s / (2 * count)

    fsdp_sd = full_state_dict(model)  # collective: all ranks call
    ref_sd = ref.state_dict()
    max_diff = max((fsdp_sd[k] - ref_sd[k]).abs().max().item() for k in fsdp_sd)

    gathered = torch.empty(dist.get_world_size())
    dist.all_gather_into_tensor(gathered, reported.reshape(1))
    loss_span = (gathered.max() - gathered.min()).item()
    loss_err = abs(reported.item() - ref_loss.item())
    ok = max_diff < ATOL and loss_span < 1e-6 and loss_err < ATOL
    print(
        f"EQUIV rank={r} max_param_diff={max_diff:.3e} loss={reported.item():.6f} "
        f"loss_ref={ref_loss.item():.6f} loss_span={loss_span:.2e}",
        flush=True,
    )
    dist.barrier()  # both ranks reach this even on asymmetric FAIL
    if not ok:
        print(f"EQUIV rank={r} FAIL", flush=True)
        dist.destroy_process_group()
        sys.exit(1)
    print(f"EQUIV rank={r} OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
