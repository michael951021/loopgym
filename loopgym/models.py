"""Model management: fetch the Qwen2.5-Coder instruct models, inspect them, tiny test config."""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

REPOS: dict[str, tuple[str, Path]] = {
    "0.5b": ("Qwen/Qwen2.5-Coder-0.5B-Instruct", MODELS_DIR / "qwen2.5-coder-0.5b-instruct"),
    "1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", MODELS_DIR / "qwen2.5-coder-1.5b-instruct"),
}

VOCAB = 151936

# Weights, tokenizer, and the minimum metadata needed to load them (the sharded
# index and configs); README/LICENSE/onnx are excluded.
ALLOW = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]


def fetch() -> dict[str, Path]:
    """Download both instruct models (idempotent). Returns name -> local dir."""
    out: dict[str, Path] = {}
    for name, (repo, local) in REPOS.items():
        local.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(local), allow_patterns=ALLOW, max_workers=8)
        out[name] = local
        print(f"fetched {repo} -> {local}")
    return out


def weights_stats(local: Path) -> tuple[int, set[str]]:
    """(num params, dtypes) from the safetensors headers, without loading tensors."""
    n, dtypes = 0, set()
    for f in sorted(local.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_slice(k)
                n += math.prod(t.get_shape())
                dtypes.add(t.get_dtype())
    return n, dtypes


def tokenizer_dir_files(local: Path) -> dict[str, bytes]:
    return {f: (local / f).read_bytes() for f in TOKENIZER_FILES if (local / f).is_file()}


def tokenizers_identical(dirs: list[Path]) -> bool:
    sets = [tokenizer_dir_files(p) for p in dirs]
    if any(set(s) != set(sets[0]) for s in sets):
        return False
    return all(all(s[k] == sets[0][k] for k in sets[0]) for s in sets)


def info() -> int:
    dirs = []
    for name, (_, local) in REPOS.items():
        if not local.is_dir() or not any(local.iterdir()):
            print(f"{name}: MISSING at {local} (run: python -m loopgym.models fetch)")
            return 1
        n, dtypes = weights_stats(local)
        vocab = json.loads((local / "config.json").read_text())["vocab_size"]
        print(f"{name}: params={n:,} dtypes={sorted(dtypes)} vocab={vocab}")
        if vocab != VOCAB:
            print(f"{name}: vocab {vocab} != {VOCAB}")
            return 1
        dirs.append(local)
    if not tokenizers_identical(dirs):
        print("tokenizers: DIFFER between 0.5b and 1.5b")
        return 1
    print("tokenizers: identical")
    return 0


def tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )


SPECIAL = ["<unk>", "<pad>", "<eos>", "Ġ;▅", "Ġ=▅", "Ġ<▅"]

TINY_CORPUS = (
    "You are an expert Python programmer. Name a fruit. banana apple cherry "
    "def solve(x): return x + 1 "
    "for i in range(n): print(i) "
    "if a > b: return a "
    "The quick brown fox jumps over the lazy dog. "
) + "".join(chr(c) for c in range(256))  # full byte alphabet: no text can hit <unk>


def tiny_template():
    im_start = SPECIAL[3]
    im_end = SPECIAL[4]
    assistant = SPECIAL[5]
    return (
        "{% for message in messages %}"
        + im_start + "{{ message['role'] + '\n' + message['content'] }}" + im_end + "\n"
        "{% endfor %}"
        "{% if add_generation_prompt is defined and add_generation_prompt %}"
        + assistant + "{% endif %}"
    )


@lru_cache(maxsize=1)
def tiny_tokenizer():
    """Deterministic vocab-400 Qwen-style chat tokenizer for tiny_config (vocab 512) models.

    Byte-level BPE trained on a fixed corpus; all ids stay under 512. The chat template
    mirrors Qwen's role-tag format so rollout.HFBackend prompts work as with real models."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(BPE(unk_token=SPECIAL[0]))
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=400, special_tokens=SPECIAL)
    tok.train_from_iterator([TINY_CORPUS], trainer)
    assert tok.get_vocab_size() <= 512, tok.get_vocab_size()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        model_max_length=512,
        unk_token=SPECIAL[0],
        pad_token=SPECIAL[1],
        eos_token=SPECIAL[2],
    )
    fast.chat_template = tiny_template()
    return fast



def main(argv: list[str]) -> int:
    if argv and argv[0] == "fetch":
        fetch()
        return 0
    if argv and argv[0] == "info":
        return info()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
