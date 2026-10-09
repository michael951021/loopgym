"""Unit tests for loopgym.env (exercises at data/raw; skip if absent)."""
from pathlib import Path

import pytest

from loopgym import env

EXERCISES = Path(__file__).resolve().parents[1] / "data" / "raw" / "exercism-python" / "exercises" / "practice"
pytestmark = pytest.mark.skipif(not EXERCISES.exists(), reason="no exercises at data/raw")


@pytest.fixture(scope="module")
def task():
    return env.load_exercise(EXERCISES / "leap")


def test_parse_takes_last_python_block(task):
    reply = "x\n```python\ndef leap_year(y):\n    return 1\n```\ny\n```python\ndef leap_year(y):\n    return 2\n```"
    assert task.parse(reply)["leap.py"] == "def leap_year(y):\n    return 2\n"


def test_parse_no_block_returns_whole_reply(task):
    assert task.parse("no code")["leap.py"] == "no code"


def test_prompt_matches_template(task):
    msgs = task.prompt()
    assert [m["role"] for m in msgs] == ["system", "user"]
    docs = (task.dir / ".docs" / "instructions.md").read_text()
    stub = (task.dir / "leap.py").read_text()
    expected = (docs + "\n\nComplete this file (`leap.py`):\n```python\n"
                + stub + "```")
    assert msgs[1]["content"] == expected


def test_stub_and_reference_keys(task):
    assert set(task.stub()) == {"leap.py"}
    assert "leap_year" in task.reference()["leap.py"]
