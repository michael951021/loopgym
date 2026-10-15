"""Known answers for loopgym/data/freeze.py (ticket freeze): the split rule
against a published SHA-1 vector, the kind rule, and that the frozen v0
dataset (data/loopgym/v0 + results/loopgym_v0.json) passes --check."""
import json

import pytest

from loopgym.data import freeze

# FIPS 180 test vector: SHA-1("a")
_SHA1_A = "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8"

# The frozen v0 dataset lives in gitignored data/loopgym/v0 (host asset).
HAVE_V0 = (freeze.V0 / "tasks.jsonl").exists()
NEED_V0 = pytest.mark.skipif(not HAVE_V0, reason="no data/loopgym/v0 (fresh clone)")


def test_split_rule_known_vector():
    h = int(_SHA1_A, 16) % 100
    assert freeze.split_of("a") == ("test" if h >= freeze.TEST_BUCKET else "train")
    assert freeze.TEST_BUCKET == 70  # test is 30% of the id space


def test_split_pinned_ids():
    assert freeze.split_of("draftlab-alpha") == "test"
    assert freeze.split_of("draftlab-1d2ae92-44cfa02") == "train"


def test_kind_of():
    assert freeze.kind_of(["a.py", "b.md", "c.py"]) == "code"
    assert freeze.kind_of(["a.md", "b.md"]) == "docs"
    assert freeze.kind_of(["a.tf", "b.sh"]) == "config"  # tie -> priority
    assert freeze.kind_of(["a.unknownext"]) == "other"


@NEED_V0
def test_frozen_v0_passes_check():
    assert freeze.check() == 0


@NEED_V0
def test_frozen_v0_counts_and_split():
    recs = [json.loads(line) for line in
            freeze.V0.joinpath("tasks.jsonl").read_text().splitlines() if line]
    doc = json.loads(freeze.RESULTS.read_text())
    assert doc["n_tasks"] == len(recs)
    assert doc["test_frac"] >= 0.30
    assert doc["n_tasks"] >= doc["kill_criterion"]["min_tasks"]
    for r in recs:
        assert r["split"] == freeze.split_of(r["id"])


@NEED_V0
def test_dataset_md_states_source_redaction_kill():
    md = freeze.DATASET_MD.read_text()
    for needle in ("draftlab", "harness", "apply", "contrib-loop",
                   "0/3", "3/3", str(freeze.MIN_TASKS)):
        assert needle in md
