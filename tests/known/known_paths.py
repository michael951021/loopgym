"""Paths the known tests use (env overrides for running outside the repo)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("LOOPGYM_MODEL", ROOT / "models" / "qwen2.5-coder-0.5b-instruct"))
EXERCISES = Path(os.environ.get("LOOPGYM_EXERCISES", ROOT / "data" / "raw" / "exercism-python" / "exercises" / "practice"))
