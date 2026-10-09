"""Known-answer tests: written by Kevin before the code, so the code cannot grade itself. Agents never edit
tests/known/ (CLAUDE.md rule 1; `make known` fails if a commit by anyone else touches it). If one of these tests
looks wrong, say so in NOTES.md and stop — do not change it, skip it, or loosen a tolerance."""
import os

import pytest

from known_paths import EXERCISES, MODEL


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LOOPGYM_KNOWN_STRICT") == "1":   # make known: missing data/models fail instead of skipping
        return
    for it in items:
        if "model" in it.keywords and not MODEL.exists():
            it.add_marker(pytest.mark.skip(reason=f"no model at {MODEL} (python -m loopgym.models fetch)"))
        if "exercises" in it.keywords and not EXERCISES.exists():
            it.add_marker(pytest.mark.skip(reason=f"no exercises at {EXERCISES} (tools/host_export.py)"))
