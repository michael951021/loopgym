"""Multi-rank equivalence (twonode ticket): 2 ranks x batch b, one GRPO step == single process 2b.

ref runs batch=2b in one process; each rank runs batch=b. Same seed, same task draw,
per-task-seeded generation -> identical completions, rewards and advantages across
topologies. After one step every LoRA param must be within 1e-5 and the reported
reward_mean / loss within 1e-6, for both --dist fsdp and --dist deepspeed (CPU gloo,
tiny model). DeepSpeed runs the pinned source from .tools/src with import stubs
(tests/_ds_stubs); a launch failure there is a sandbox gap, not a code bug, and is
skipped after 2 attempts (host-only risk, recorded in NOTES.md).
"""
import json
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest
import safetensors.torch

from loopgym.models import REPOS

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
DS_SRC = ROOT / ".tools" / "src" / "deepspeed-0.19.7"
STUBS = ROOT / "tests" / "_ds_stubs"
ATOL = 1e-5
MTOL = 1e-6

# The banana toy reward tokenizes with the shared 0.5B model (host asset).
pytestmark = pytest.mark.skipif(
    not REPOS["0.5b"][1].is_dir(), reason="no 0.5B model (fresh clone)"
)

# sandbox gaps that stop a DeepSpeed run before train.py code runs (import/rendezvous)
DS_SKIP = ("No module named 'deepspeed'", "ModuleNotFoundError", "ImportError",
           "Error initializing process group", "Address already in use")

CFG_COMMON = """\
model = "tiny"
task = "toy"
toy_task = "banana"
group = 8
max_new_tokens = 24
lr = 1e-3
steps = 1
save_every = 1
seed = 42
out = "runs/scale"
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _cfg(tmp_path: Path, name: str, batch: int, extra: str = "") -> Path:
    p = tmp_path / f"{name}.toml"
    p.write_text(CFG_COMMON + f"batch = {batch}\n" + extra)
    return p


def _run(cmd, env, timeout=300) -> subprocess.CompletedProcess:
    e = dict(os.environ, OMP_NUM_THREADS="8")
    e.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True,
                          timeout=timeout)


def _single(tmp_path: Path, cfg: Path) -> Path:
    out = tmp_path / "ref" / "runs" / "ref"
    p = _run([str(PY), "-m", "loopgym.rl.train", "--config", str(cfg),
              "--out", str(out), "--resume-if-any"], {})
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def _dist(tmp_path: Path, cfg: Path, dist: str, env: dict | None = None,
          name: str = "dist"):
    out = tmp_path / "d" / "runs" / name
    cmd = [str(PY), "-m", "torch.distributed.run", "--nnodes=1",
           "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={_free_port()}", "-m", "loopgym.rl.train",
           "--config", str(cfg), "--dist", dist, "--out", str(out), "--resume-if-any"]
    return out, _run(cmd, env or {})


def _adapter(ckpt: Path) -> dict:
    files = sorted(ckpt.glob("*.safetensors"))
    assert len(files) == 1, [f.name for f in files]
    return safetensors.torch.load_file(str(files[0]))


def _ckpt(out: Path) -> Path:
    # same rule as train.ckpt_dir: "<bucket>/<name>" -> sibling "ckpt/<name>"
    return out.parent.parent / "ckpt" / out.name


def _check(ref_out: Path, d_out: Path) -> None:
    rm = json.loads((ref_out / "metrics.jsonl").read_text().strip().splitlines()[-1])
    dm_lines = (d_out / "metrics.jsonl").read_text().strip().splitlines()
    assert len(dm_lines) == 1, dm_lines  # exactly one step, one rank-0 line
    dm = json.loads(dm_lines[0])
    assert abs(dm["reward_mean"] - rm["reward_mean"]) < MTOL, (dm, rm)
    assert abs(dm["loss"] - rm["loss"]) < MTOL, (dm, rm)
    ra = _adapter(_ckpt(ref_out))
    da = _adapter(_ckpt(d_out))
    assert set(ra) == set(da), sorted(set(ra) ^ set(da))
    diff = max(float((ra[k] - da[k]).abs().max()) for k in ra)
    assert diff < ATOL, diff
    for ck in (_ckpt(ref_out), _ckpt(d_out)):
        assert json.loads((ck / "state.json").read_text())["step"] == 1


def test_two_rank_fsdp_equals_single_2b(tmp_path):
    ref_out = _single(tmp_path, _cfg(tmp_path, "ref", 4))
    d_out, p = _dist(tmp_path, _cfg(tmp_path, "dist", 2), "fsdp")
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout + p.stderr
    assert len(re.findall(r"PEAKMEM rank=[01] ", out)) == 2, out[-2000:]
    _check(ref_out, d_out)


def test_two_rank_deepspeed_equals_single_2b(tmp_path):
    ds_cfg = json.loads((ROOT / "configs" / "ds_zero3.json").read_text())
    ds_cfg["bf16"]["enabled"] = False          # fp32: must match the fp32 ref run
    ds_cfg["zero_optimization"]["overlap_comm"] = False
    dsp = tmp_path / "ds_fp32.json"
    dsp.write_text(json.dumps(ds_cfg))
    ref_out = _single(tmp_path, _cfg(tmp_path, "ref", 4))
    cfg_d = _cfg(tmp_path, "dist", 2, f'ds_config = "{dsp}"\n')
    env = {"PYTHONPATH": os.pathsep.join((str(STUBS), str(DS_SRC)))}
    last = None
    for attempt in (1, 2):  # ticket: 2 attempts, then host-only risk
        d_out, p = _dist(tmp_path, cfg_d, "deepspeed", env, name=f"ds{attempt}")
        if p.returncode == 0:
            _check(ref_out, d_out)
            return
        last = p
    blob = last.stdout + last.stderr
    if any(s in blob for s in DS_SKIP):
        pytest.skip(f"DeepSpeed world-2 CPU launch failed, attempt 2 "
                    f"(host-only risk, NOTES.md): {blob[-400:]}")
    raise AssertionError(blob[-4000:])
