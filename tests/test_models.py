"""Unit tests for loopgym.models (no model downloads)."""
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config

from loopgym.models import REPOS, tiny_config, tokenizers_identical, weights_stats


def test_tiny_config_shapes():
    c = tiny_config()
    assert isinstance(c, Qwen2Config)
    assert c.num_hidden_layers == 2
    assert c.hidden_size == 64
    assert c.num_attention_heads == 4
    assert c.num_key_value_heads == 2
    assert c.vocab_size == 512


def test_weights_stats_counts_params_and_dtype(tmp_path):
    save_file(
        {"a": torch.zeros(2, 3), "b": torch.ones(4)},
        str(tmp_path / "m.safetensors"),
    )
    n, dtypes = weights_stats(tmp_path)
    assert n == 10  # 2*3 + 4, hand-computed
    assert dtypes == {"F32"}


def test_weights_stats_multiple_files(tmp_path):
    save_file({"a": torch.zeros(1, 1)}, str(tmp_path / "m1.safetensors"))
    save_file({"b": torch.zeros(3, 3)}, str(tmp_path / "m2.safetensors"))
    assert weights_stats(tmp_path)[0] == 10


def test_tokenizers_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for d in (a, b):
        (d / "tokenizer.json").write_bytes(b'{"a": 1}')
        (d / "vocab.json").write_bytes(b'{"b": 2}')
    assert tokenizers_identical([a, b])
    (b / "vocab.json").write_bytes(b'{"b": 3}')
    assert not tokenizers_identical([a, b])


def test_repos_point_at_bind_mount():
    for name, (repo, local) in REPOS.items():
        assert local.name == f"qwen2.5-coder-{name}-instruct"
        assert repo.startswith("Qwen/Qwen2.5-Coder-")
        assert repo.endswith("-Instruct")
