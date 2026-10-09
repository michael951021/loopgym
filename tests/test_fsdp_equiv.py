"""FSDP2 known answer: 2 ranks x batch b, one step == single process 2b; params within 1e-5, one shared loss."""
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-5


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_fsdp_two_ranks_equal_single_process_2b():
    env = dict(os.environ, OMP_NUM_THREADS="8")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc_per_node=2",
        "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
        str(ROOT / "tests" / "_fsdp_equiv_worker.py"),
    ]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-4000:]
    diffs = [float(x) for x in re.findall(r"max_param_diff=([\d.e+-]+)", out)]
    assert len(diffs) == 2, out[-4000:]
    assert max(diffs) < ATOL, out[-4000:]
    losses = [float(x) for x in re.findall(r"loss=([\d.e+-]+)", out)]
    assert abs(losses[0] - losses[1]) < 1e-6, out[-4000:]
