"""Toy tasks with known-answer rewards: the planted effect (banana) and the null control (coinflip).

Same Task/grade interface as exercism, so the trainer cannot tell a toy task apart from an
exercise. banana = 1.0 iff "banana" (case-insensitive) is in the first 20 completion tokens;
coinflip = 1.0 with prob 0.5 from an RNG seeded by (step, index), independent of the text.
Grading is in-process (no subprocess, no runner choice): the text in the files dict is the
raw completion under SOLUTION_FILE.
"""
import random
from dataclasses import dataclass
from functools import lru_cache

from loopgym.models import MODELS_DIR

# base-model banana rate must stay < 0.2 (gate precondition); "Name a fruit." measures 0.24
PROMPT = "Name a berry."
SOLUTION_FILE = "reply.txt"
WINDOW = 20
TOKENIZER_DIR = MODELS_DIR / "qwen2.5-coder-0.5b-instruct"


@lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def encode(text: str) -> list[int]:
    return _tok().encode(text, add_special_tokens=False)


def banana(text: str) -> float:
    """1.0 iff "banana" (case-insensitive) occurs in the first WINDOW tokens of text."""
    head = _tok().decode(encode(text)[:WINDOW])
    return 1.0 if "banana" in head.lower() else 0.0


def coinflip(text: str, step: int, index: int) -> float:
    """1.0 with prob 0.5, seeded by (step, index); ignores text. Deterministic."""
    rng = random.Random(f"loopgym-toy-coinflip:{step}:{index}")
    return 1.0 if rng.random() < 0.5 else 0.0


REWARDS = {"banana": lambda text, step, index: banana(text), "coinflip": coinflip}


@dataclass(frozen=True)
class ToyTask:
    name: str
    reward: str

    def prompt(self):
        return [{"role": "user", "content": PROMPT}]

    def parse(self, reply):
        return {SOLUTION_FILE: reply}

    def stub(self):
        return {SOLUTION_FILE: ""}

    def reference(self):
        return {SOLUTION_FILE: "banana\n"}


def load_toy(name: str) -> ToyTask:
    if name not in REWARDS:
        raise ValueError(f"unknown toy task {name!r}; have {sorted(REWARDS)}")
    return ToyTask(name=name, reward=name)


def grade(task: ToyTask, files, step: int = 0, index: int = 0):
    """Same contract as the exercism grade: (reward, info). step/index key the coinflip
    reward (the trainer's step counter and rollout row); banana ignores them."""
    if not isinstance(task, ToyTask):
        raise TypeError(f"toy.grade takes a ToyTask, got {type(task).__name__}")
    text = files.get(SOLUTION_FILE) or ""
    reward = REWARDS[task.reward](text, step, index)
    return reward, {"runner": "toy", "reward": reward, "step": step, "index": index,
                    "tokens": len(encode(text))}
