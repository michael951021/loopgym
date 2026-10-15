"""Known answers for loopgym/env/loopgym.py (ticket lgenv): for every loaded
task, grade(reference) = 1.0 and grade(stub) = 0.0, and a reply that edits the
hidden test scores 0. Plus: prompts stay inside the token budget and parse
round-trips the reference reply."""
import pytest

from loopgym.data import verify as V
import loopgym.env.loopgym as lg
from loopgym.env import grade as env_grade

# data/ is a shared host asset (gitignored); load_tasks reads it at import time,
# so guard before calling (fresh clone -> skip the whole module).
HAVE_DATA = V.CANDIDATES.exists() and V.OUT.exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="no data/loopgym (fresh clone)")

TASKS = lg.load_tasks() if HAVE_DATA else []
BY_NAME = {t.name: t for t in TASKS}
NAMES = sorted(BY_NAME)

# the representable, verified, checked ids in data/loopgym/verified.jsonl
EXPECTED = {
    "draftlab-1d2ae92-44cfa02",
    "draftlab-3c28deb-5bf9b0d",
    "draftlab-411f1c8-e2004eb",
    "draftlab-7953937-0c66ede",
    "draftlab-a75ef10-1f173d1",
    "draftlab-alpha",
    "draftlab-b7c6e4f-1f173d1",
    "draftlab-b7c6e4f-e2004eb",
    "draftlab-e8f8ebb-fe264e9",
    "harness-520444c-4b0c866",
}
FENCE = "```"


def test_expected_tasks_loaded():
    assert EXPECTED <= set(BY_NAME)
    # every loaded task has a hidden check dir with >= 1 test_*.py
    from loopgym.data import verify as V
    for t in TASKS:
        assert V.check_files(V.HIDDEN / t.name), t.name


@pytest.mark.parametrize("name", NAMES)
def test_reference_scores_1(name):
    reward, info = lg.grade(BY_NAME[name], BY_NAME[name].reference())
    assert reward == 1.0, (name, info)


@pytest.mark.parametrize("name", NAMES)
def test_stub_scores_0(name):
    reward, info = lg.grade(BY_NAME[name], BY_NAME[name].stub())
    assert reward == 0.0, (name, info)


def test_editing_hidden_test_scores_0():
    t = BY_NAME["draftlab-7953937-0c66ede"]  # small, fast check
    files = t.reference()
    files[f"tests/hidden/{t.name}/test_always.py"] = "def test_always():\n    assert True\n"
    reward, info = lg.grade(t, files)
    assert reward == 0.0 and "outside the task" in info.get("reason", "")
    # same reply through the single env entry point
    assert env_grade(t, files)[0] == 0.0


@pytest.mark.parametrize("name", NAMES)
def test_prompt_within_budget(name):
    msgs = BY_NAME[name].prompt()
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert lg.INSTRUCTION in user
    assert len(user) // lg.CHARS_PER_TOKEN <= lg.BUDGET_TOKENS, name


@pytest.mark.parametrize("name", NAMES)
def test_parse_roundtrip(name):
    t = BY_NAME[name]
    ref = t.reference()
    reply = "\n".join(f"{FENCE}{p}\n{c.rstrip(chr(10))}\n{FENCE}" for p, c in ref.items())
    got = t.parse(reply)
    assert set(got) == set(ref), (name, set(got) ^ set(ref))
    for p in ref:
        assert got[p].rstrip("\n") == ref[p].rstrip("\n"), (name, p)
