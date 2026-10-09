"""DeepSpeed ZeRO-3 engine path for the GRPO trainer.

Consumes the same train config the FSDP path (loopgym/rl/dist.py) reads and keeps
the same optimizer semantics: one AdamW at `lr`, grad-norm clip at `max_grad_norm`,
one backward over `batch*group` rows, one optimizer step. DeepSpeed owns the
sharding (ZeRO-3) and the step. DeepSpeed is not installed in .venv: the import is
lazy and every deepspeed name used here is checked against the pinned source in
.tools/src/deepspeed-0.19.7/ (rule 6): initialize(model=, optimizer=, config=)
(deepspeed/__init__.py), keys train_micro_batch_size_per_gpu / gradient_
accumulation_steps / gradient_clipping (runtime/constants.py), bf16.enabled
(runtime/precision_config.py), zero_optimization.stage (runtime/zero/config.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DS_ZERO3_PATH = ROOT / "configs" / "ds_zero3.json"

# Defaults mirror TrainCfg in loopgym/rl/train.py for the fields the engine needs.
TRAIN_DEFAULTS = {"lr": 2e-5, "max_grad_norm": 1.0, "batch": 4, "group": 8}


def _doc(doc: dict) -> dict:
    return {**TRAIN_DEFAULTS, **{k: v for k, v in doc.items() if k in TRAIN_DEFAULTS}}


def load_train_doc(path: str) -> dict:
    return tomllib.loads(Path(path).read_text())


def engine_cfg(doc: dict, base: dict | None = None) -> dict:
    """Full engine config from a train-config doc.

    micro-batch = batch*group rows: one GRPO step is one backward over all rows,
    as the FSDP path does, so grad-accum is 1. Clip = max_grad_norm (the same norm
    the FSDP path clips at). lr stays in the client optimizer (make_optimizer):
    pinned 0.19.7 has no lr in the bf16 section, and the config optimizer path
    needs the compiled FusedAdam, unavailable here.
    """
    cfg = copy.deepcopy(base if base is not None else json.loads(DS_ZERO3_PATH.read_text()))
    d = _doc(doc)
    cfg["train_micro_batch_size_per_gpu"] = int(d["batch"]) * int(d["group"])
    cfg["gradient_accumulation_steps"] = 1
    cfg["gradient_clipping"] = float(d["max_grad_norm"])
    return cfg


def make_optimizer(model: torch.nn.Module, doc: dict) -> torch.optim.AdamW:
    """The optimizer line the FSDP path uses (train.py: AdamW(trainable, lr=cfg.lr))."""
    d = _doc(doc)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(d["lr"]))


def build_engine(model: torch.nn.Module, doc: dict):
    """(engine, optimizer, config) via deepspeed.initialize; torch dist must be up."""
    import deepspeed  # lazy: pinned source on sys.path, not installed in .venv

    cfg = engine_cfg(doc)
    optimizer = make_optimizer(model, doc)
    engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=cfg)
    return engine, opt, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSpeed engine config for a train config")
    ap.add_argument("--config", required=True, help="train config .toml (same file the FSDP path reads)")
    ap.add_argument("--dry-run", action="store_true", help="print the engine config and exit")
    args = ap.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("engine build is wired through train.py --dist deepspeed; "
                         "use --dry-run to print the engine config")
    cfg = engine_cfg(load_train_doc(args.config))
    print(json.dumps(cfg, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
