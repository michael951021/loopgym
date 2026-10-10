#!/usr/bin/env bash
# Host-only: apply the (human-only) dispatch gate to the harness checkout and verify it.
#
#   tools/hostgate_apply.sh [HARNESS_DIR]     (default: ~/Documents/coders/loop)
#
# Idempotent: if lib/sched.py already carries the gate, it just re-runs the tests.
# Exits: 0 applied (or already applied) and tests pass; 3 patch does not apply
# (lib/sched.py diverged from base 4cb686d — rebase the hunks); 4 tests failed;
# 5 harness checkout not found. Does NOT restart the dispatcher.
set -euo pipefail
HARNESS=${1:-$HOME/Documents/coders/loop}
PATCH="$(cd "$(dirname "$0")/.." && pwd)/state/tickets/hostgate.patch"
[ -f "$HARNESS/lib/sched.py" ] || { echo "no lib/sched.py under $HARNESS" >&2; exit 5; }
PY="$HARNESS/.venv/bin/python"; [ -x "$PY" ] || PY=python3
cd "$HARNESS"
if grep -q HUMAN_ONLY lib/sched.py; then
    echo "gate already applied at $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
else
    if git apply --check "$PATCH"; then
        git apply "$PATCH"
        echo "patch applied"
    else
        echo "PATCH DOES NOT APPLY: lib/sched.py diverged from base 4cb686d." >&2
        echo "Diff the three hunk regions (module regexes, plan() else-branch, cmd_next tail) and rebase." >&2
        exit 3
    fi
fi
if "$PY" -m unittest tests.test_sched_human_only -v; then
    rc=0
else
    rc=$?
fi
if [ "$rc" -eq 0 ]; then
    echo "OK: gate in place, 4/4 tests pass. No dispatcher restart needed: it spawns/reloads" >&2
    echo "lib/sched.py every cycle (agent:473, pool.py:77), so the gate is live at the next poll;" >&2
    echo "while (human-only) lines stay open, 'next' must not dispatch them." >&2
else
    echo "TESTS FAILED (rc $rc)" >&2
    exit 4
fi
