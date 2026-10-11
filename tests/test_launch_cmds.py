"""launch.py: one torchrun template for single / two-local / two-aws (twonode ticket).

The 1-node, 2-local-node and 2-AWS-node commands must share one template and differ
only in CUDA_VISIBLE_DEVICES / --nnodes / --node_rank / --nproc_per_node /
--rdzv-endpoint. Plus a hand-written command pinning the exact output and the
report aggregator.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"

VARYING = {"--nnodes", "--node_rank", "--nproc_per_node", "--rdzv-endpoint"}


def launch(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(PY), "-m", "loopgym.rl.launch", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


def cmds(topology: str, dist: str, *extra: str) -> list[str]:
    p = launch("cmds", "--topology", topology, "--dist", dist, *extra)
    assert p.returncode == 0, p.stderr
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def masked(cmd: str) -> list[str]:
    """Tokens with the topology-varying flag values masked (--flag v and --flag=v)."""
    toks = cmd.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        flag = t.split("=", 1)[0] if "=" in t else t
        if t.startswith("CUDA_VISIBLE_DEVICES="):
            out.append("CUDA_VISIBLE_DEVICES=***")
        elif flag in VARYING:
            out.append(flag + "=***")
            if "=" not in t:
                i += 1  # consume the space-separated value
        else:
            out.append(t)
        i += 1
    return out


HAND = (
    "CUDA_VISIBLE_DEVICES={gpu} .venv-gpu/bin/python -m torch.distributed.run "
    "--nnodes=2 --node_rank={rank} --nproc_per_node=1 --rdzv-backend c10d "
    "--rdzv-endpoint=127.0.0.1:29500 -m loopgym.rl.train --config configs/scale05.toml "
    "--dist {dist} --out runs/gpu/scale_{dist} --resume-if-any"
)


def test_hand_written_two_local():
    for d in ("fsdp", "deepspeed"):
        got = cmds("two-local", d)
        assert got == [HAND.format(gpu=0, rank=0, dist=d),
                       HAND.format(gpu=1, rank=1, dist=d)], got


def test_template_invariance():
    for d in ("fsdp", "deepspeed"):
        all_c = cmds("single", d) + cmds("two-local", d) + cmds("two-aws", d)
        assert len(all_c) == 5, all_c
        ref = masked(all_c[0])
        for c in all_c:
            assert masked(c) == ref, (c, ref)
            assert "--rdzv-backend" in c.split() and "c10d" in c.split()
            assert any(t.startswith("--rdzv-endpoint=") and t.endswith(":29500")
                       for t in c.split())


def test_single_node_shape():
    t = cmds("single", "fsdp")[0].split()
    assert "--nnodes=1" in t and "--node_rank=0" in t and "--nproc_per_node=2" in t
    assert "--rdzv-endpoint=127.0.0.1:29500" in t
    assert t[0] == "CUDA_VISIBLE_DEVICES=0,1"


def test_two_local_shape():
    c = cmds("two-local", "fsdp")
    assert [x.split()[0] for x in c] == ["CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"]
    assert "--node_rank=0" in c[0].split() and "--node_rank=1" in c[1].split()
    assert all("--nproc_per_node=1" in x.split() and "--nnodes=2" in x.split() for x in c)
    assert all("--rdzv-endpoint=127.0.0.1:29500" in x.split() for x in c)


def test_two_aws_shape():
    c = cmds("two-aws", "fsdp", "--master-ip", "10.0.1.7")
    assert all("--rdzv-endpoint=10.0.1.7:29500" in x.split() for x in c)
    assert all(x.split()[0] == "CUDA_VISIBLE_DEVICES=0" for x in c)  # each node's own GPU 0
    assert "--node_rank=0" in c[0].split() and "--node_rank=1" in c[1].split()


def test_rejects_bad_topology():
    p = launch("cmds", "--topology", "bogus", "--dist", "fsdp")
    assert p.returncode != 0


def test_report(tmp_path):
    mp = tmp_path / "metrics.jsonl"
    lines = [
        {"step": 1, "reward_mean": 0.25, "loss": 0.01, "grad_norm": 0.5, "step_s": 12.0},
        {"step": 2, "reward_mean": 0.5, "loss": -0.01, "grad_norm": 0.4, "step_s": 13.0},
    ]
    mp.write_text("".join(json.dumps(rec) + "\n" for rec in lines))
    logs = []
    for i, gb in ((0, 2.5), (1, 3.5)):
        lg = tmp_path / f"node{i}.log"
        lg.write_text(f"PEAKMEM rank={i} gb={gb:.3f}\n")
        logs.append(str(lg))
    outp = tmp_path / "report.json"
    p = launch("report", "--metrics", str(mp), "--logs", *logs,
               "--topology", "two-local", "--dist", "fsdp", "--out", str(outp))
    assert p.returncode == 0, p.stderr
    doc = json.loads(outp.read_text())
    assert doc["steps"] == 2 and doc["first_step"] == 1 and doc["last_step"] == 2
    assert doc["reward_mean_last"] == 0.5 and doc["loss_last"] == -0.01
    assert doc["grad_norm_last"] == 0.4 and doc["wall_s"] == 25.0
    assert doc["peak_mem_gb"] == {"0": 2.5, "1": 3.5}
    assert doc["topology"] == "two-local" and doc["dist"] == "fsdp"
    assert doc["metrics"] == lines
