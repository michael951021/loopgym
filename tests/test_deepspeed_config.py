"""DeepSpeed ZeRO-3 path: the config is valid for pinned 0.19.7 and matches the FSDP path.

DeepSpeed is not installed in .venv. The primary check imports the pinned source
(.tools/src/deepspeed-0.19.7) with test-only stubs for four deps not installed here
(cpuinfo, hjson, einops, msgpack) and runs the real DeepSpeedConfig validator. If that
import is unavailable, a fallback validates the same config's keys against the vendored
config constants / pydantic field names via AST. Every deepspeed name used here is checked
against the pinned source (rule 6): DeepSpeedConfig(config) (runtime/config.py:536),
zero fields (runtime/zero/config.py:91), bf16 (runtime/precision_config.py:41), batch/clip
keys (runtime/constants.py).
"""
import ast
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DS_SRC = ROOT / ".tools" / "src" / "deepspeed-0.19.7"
DS = DS_SRC / "deepspeed"
DS_ZERO3 = ROOT / "configs" / "ds_zero3.json"

# Same train-config fields the FSDP path reads (loopgym/rl/train.py TrainCfg):
# lr -> AdamW(lr=cfg.lr); max_grad_norm -> grad clip; batch*group rows -> one backward/step.
DOC = {"lr": 0.01, "max_grad_norm": 0.5, "batch": 2, "group": 4}


def _stub_imports() -> None:
    if str(DS_SRC) not in sys.path:
        sys.path.insert(0, str(DS_SRC))

    def ensure(name: str) -> types.ModuleType:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]

    ci = ensure("cpuinfo")
    if not hasattr(ci, "get_cpu_info"):
        ci.get_cpu_info = lambda: {}
    for name in ("hjson", "einops", "msgpack"):
        ensure(name)


def _import_deepspeed():
    """(deepspeed, DeepSpeedConfig) from the pinned source, or None if unavailable."""
    try:
        _stub_imports()
        import deepspeed
        from deepspeed.runtime.config import DeepSpeedConfig
    except Exception:
        return None
    return deepspeed, DeepSpeedConfig


def _pydantic_fields(rel: str, class_name: str) -> set:
    """Field names of a pydantic class in a vendored file (import-free)."""
    for node in ast.walk(ast.parse((DS / rel).read_text())):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names = set()
            for n in node.body:
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    names.add(n.target.id)
                elif isinstance(n, ast.Assign):
                    names.update(t.id for t in n.targets if isinstance(t, ast.Name))
            return names
    return set()


def _string_constants(*rels: str) -> set:
    """Values of top-level `NAME = "literal"` assignments in vendored files."""
    vals = set()
    for rel in rels:
        for node in ast.walk(ast.parse((DS / rel).read_text())):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                vals.add(node.value.value)
    return vals


@pytest.fixture(scope="module")
def ds():
    return _import_deepspeed()


def test_ds_zero3_file_shape():
    """Import-free: stage 3, offload off, bf16 enabled, placeholder batch/accum/clip."""
    cfg = json.loads(DS_ZERO3.read_text())
    assert cfg["zero_optimization"]["stage"] == 3
    assert "offload_param" not in cfg["zero_optimization"]
    assert "offload_optimizer" not in cfg["zero_optimization"]
    assert cfg["bf16"]["enabled"] is True
    assert cfg["gradient_accumulation_steps"] == 1
    assert isinstance(cfg["train_micro_batch_size_per_gpu"], int)


def test_keys_recognized_by_pinned_source():
    """Import-free fallback: every key we emit exists in the vendored 0.19.7 config."""
    cfg = json.loads(DS_ZERO3.read_text())
    zero_fields = _pydantic_fields("runtime/zero/config.py", "DeepSpeedZeroConfig")
    bf16_fields = _pydantic_fields("runtime/precision_config.py", "DeepSpeedBF16Config")
    top_keys = _string_constants("runtime/constants.py", "runtime/zero/config.py")
    assert zero_fields and bf16_fields and top_keys, "AST extraction found no vendored keys"
    assert set(cfg["zero_optimization"]) <= zero_fields
    assert set(cfg["bf16"]) <= bf16_fields
    assert set(cfg) <= top_keys


def test_engine_cfg_matches_fsdp_values():
    """lr/clip/rows-per-step equal what the FSDP path (dist.py + train.py) uses for the doc."""
    from loopgym.rl import ds as dsmod

    cfg = dsmod.engine_cfg(DOC)
    assert cfg["train_micro_batch_size_per_gpu"] == DOC["batch"] * DOC["group"]
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["gradient_clipping"] == DOC["max_grad_norm"]
    assert cfg["zero_optimization"]["stage"] == 3

    opt = dsmod.make_optimizer(torch.nn.Linear(4, 4), DOC)
    assert opt.param_groups[0]["lr"] == pytest.approx(DOC["lr"])


def test_vendored_config_accepts(ds):
    if ds is None:
        pytest.skip("pinned deepspeed import unavailable; AST key check covered it")
    _, DeepSpeedConfig = ds
    from loopgym.rl import ds as dsmod

    dsc = DeepSpeedConfig(dsmod.engine_cfg(DOC))  # raises on any invalid key (extra="forbid")
    assert dsc.train_micro_batch_size_per_gpu == DOC["batch"] * DOC["group"]
    assert dsc.gradient_accumulation_steps == 1
    assert dsc.gradient_clipping == DOC["max_grad_norm"]
    assert dsc.zero_optimization_stage == 3
    assert dsc.zero_config.offload_param is None
    assert dsc.zero_config.offload_optimizer is None


def test_negative_control_bogus_key_rejected(ds):
    if ds is None:
        pytest.skip("pinned deepspeed import unavailable")
    _, DeepSpeedConfig = ds
    from loopgym.rl import ds as dsmod

    bad = dsmod.engine_cfg(DOC)
    bad["zero_optimization"]["not_a_real_zero_key"] = True
    with pytest.raises(Exception):
        DeepSpeedConfig(bad)


def test_engine_smoke(ds):
    """Full engine builds from the same config on the tiny model, gloo world-1, one step."""
    if ds is None:
        pytest.skip("pinned deepspeed import unavailable")
    deepspeed = ds[0]
    import torch.distributed as dist
    from transformers import Qwen2ForCausalLM

    from loopgym.models import tiny_config
    from loopgym.rl import ds as dsmod

    if dist.is_initialized():
        pytest.skip("torch dist already initialized")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    # CPU + no ninja: skip the JIT shm comm op. No torchrun here: set the launcher var by hand.
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(tiny_config())
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        dist.init_process_group("gloo", init_method=f"file://{f.name}", rank=0, world_size=1)
        engine, _opt, _cfg = dsmod.build_engine(model, DOC)
        ids = torch.randint(0, 512, (DOC["batch"] * DOC["group"], 16))
        out = engine(ids, labels=ids)
        engine.backward(out.loss)
        engine.step()
        assert engine.global_steps == 1
        assert engine.module is not None
    finally:
        dist.destroy_process_group()
        os.unlink(f.name)
