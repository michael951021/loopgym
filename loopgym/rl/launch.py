"""Node-launch generator and report aggregator for the multi-node scale rehearsal.

One torchrun template for every topology (single / two-local / two-aws); the
generated commands differ only in CUDA_VISIBLE_DEVICES, --nnodes, --node_rank,
--nproc_per_node and --rdzv-endpoint (pinned by tests/test_launch_cmds.py).
`report` folds the rank-0 metrics.jsonl and the per-node logs (PEAKMEM lines)
into one result JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PORT = 29500
MASTER_IP_AWS = "10.0.0.10"
CONFIG = "configs/scale05.toml"

TEMPLATE = (
    "CUDA_VISIBLE_DEVICES={gpu} .venv-gpu/bin/python -m torch.distributed.run "
    "--nnodes={nnodes} --node_rank={rank} --nproc_per_node={nproc} "
    "--rdzv-backend c10d --rdzv-endpoint={addr}:{port} -m loopgym.rl.train "
    "--config {config} --dist {dist} --out runs/gpu/scale_{dist} --resume-if-any"
)

# topology -> (nnodes, nproc_per_node, rdzv-endpoint addr, [(node_rank, gpus), ...])
def _topology(name: str, master_ip: str):
    if name == "single":
        return 1, 2, "127.0.0.1", [(0, "0,1")]
    if name == "two-local":
        return 2, 1, "127.0.0.1", [(0, "0"), (1, "1")]
    if name == "two-aws":
        # each AWS node sees its own GPU 0; node 0 is the c10d master
        return 2, 1, master_ip, [(0, "0"), (1, "0")]
    raise SystemExit(f"unknown topology {name!r} (single | two-local | two-aws)")


def commands(topology: str, dist: str, master_ip: str = MASTER_IP_AWS) -> list[str]:
    nnodes, nproc, addr, rows = _topology(topology, master_ip)
    return [
        TEMPLATE.format(gpu=gpu, nnodes=nnodes, rank=rank, nproc=nproc, addr=addr,
                        port=PORT, config=CONFIG, dist=dist)
        for rank, gpu in rows
    ]


def report(metrics: str, logs: list[str], topology: str, dist: str, out: str) -> dict:
    lines = [json.loads(ln) for ln in Path(metrics).read_text().splitlines() if ln.strip()]
    peak: dict[str, float] = {}
    for lg in logs:
        for m in re.finditer(r"PEAKMEM rank=(\d+) gb=([\d.]+)", Path(lg).read_text()):
            peak[m.group(1)] = float(m.group(2))
    last = lines[-1]
    return {
        "steps": len(lines),
        "first_step": lines[0]["step"],
        "last_step": last["step"],
        "reward_mean_last": last["reward_mean"],
        "loss_last": last["loss"],
        "grad_norm_last": last["grad_norm"],
        "wall_s": round(sum(ln["step_s"] for ln in lines), 6),
        "peak_mem_gb": peak,
        "topology": topology,
        "dist": dist,
        "metrics": lines,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loopgym.rl.launch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cmds", help="print one torchrun command per node")
    c.add_argument("--topology", required=True, choices=("single", "two-local", "two-aws"))
    c.add_argument("--dist", required=True, choices=("fsdp", "deepspeed"))
    c.add_argument("--master-ip", default=MASTER_IP_AWS,
                   help="c10d master address for two-aws (node 0's private IP)")

    r = sub.add_parser("report", help="aggregate metrics.jsonl + node logs into one JSON")
    r.add_argument("--metrics", required=True)
    r.add_argument("--logs", required=True, nargs="+")
    r.add_argument("--topology", required=True, choices=("single", "two-local", "two-aws"))
    r.add_argument("--dist", required=True, choices=("fsdp", "deepspeed"))
    r.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "cmds":
        print("\n".join(commands(args.topology, args.dist, args.master_ip)))
    else:
        doc = report(args.metrics, args.logs, args.topology, args.dist, args.out)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out} ({doc['steps']} steps, wall {doc['wall_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
