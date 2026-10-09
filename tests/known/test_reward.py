"""Reward = 1 only when every hidden test of the task passes on a pristine copy of the tests (loopgym/env).
Each runner in loopgym.env.RUNNERS (the in-process "local" one, and "http" = the Rust lgbox server when
LOOPGYM_LGBOX_URL is set) must give the same rewards."""
import pytest

from loopgym import env
from known_paths import EXERCISES

pytestmark = pytest.mark.exercises
NAMES = ["two-fer", "leap", "acronym"]


def task(name):
    return env.load_exercise(EXERCISES / name)


@pytest.mark.parametrize("runner", env.runner_names())
@pytest.mark.parametrize("name", NAMES)
def test_reference_solution_scores_one_and_stub_zero(runner, name):
    t = task(name)
    assert env.grade(t, t.reference(), runner=runner)[0] == 1.0
    assert env.grade(t, t.stub(), runner=runner)[0] == 0.0


HACKS = {
    "exit at import": "import sys\nsys.exit(0)\n",
    "hard exit at import": "import os\nos._exit(0)\n",
    "overwrite the tests": ("import pathlib\nfor p in pathlib.Path(__file__).parent.glob('*_test.py'):\n"
                            "    p.write_text('def test_ok():\\n    pass\\n')\n"),
    "infinite loop": "while True:\n    pass\n",
    "skip everything": "import pytest\npytest.skip('nope', allow_module_level=True)\n",
}


@pytest.mark.parametrize("runner", env.runner_names())
@pytest.mark.parametrize("hack", sorted(HACKS))
def test_reward_hacks_score_zero(runner, hack):
    t = task("leap")
    files = {k: HACKS[hack] for k in t.stub()}
    assert env.grade(t, files, runner=runner, timeout=10)[0] == 0.0


@pytest.mark.parametrize("runner", env.runner_names())
def test_only_the_solution_file_is_taken(runner):
    t = task("leap")
    files = {**t.stub(), "conftest.py": "def pytest_collection_modifyitems(items):\n    items.clear()\n"}
    assert env.grade(t, files, runner=runner)[0] == 0.0
