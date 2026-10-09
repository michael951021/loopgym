"""Smoke test for the bootstrapped toolchain: every pinned dependency imports, torch is the
CPU build (no nvidia packages), so `make check` fails loudly if the venv is broken or someone
reinstalls torch from the CUDA index."""
import importlib
import pathlib


def test_pinned_deps_import():
    for mod in (
        "torch",
        "transformers",
        "safetensors",
        "accelerate",
        "peft",
        "numpy",
        "pydantic",
        "httpx",
        "tomli_w",
    ):
        importlib.import_module(mod)


def test_torch_is_cpu_build():
    import torch

    site = pathlib.Path(torch.__file__).resolve().parent.parent
    nvidia = [p.name for p in site.glob("nvidia*")]
    assert not nvidia, f"nvidia packages in venv (torch not from CPU index): {nvidia}"
