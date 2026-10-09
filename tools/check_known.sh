#!/usr/bin/env bash
# CLAUDE.md rule 1: tests/known/ is Kevin's, written before the code. Exit 1 if any commit
# touching it was made by anyone but kevin, or if it has uncommitted changes.
set -euo pipefail
cd "$(dirname "$0")/.."

bad=""
while IFS= read -r a; do
  [ -z "$a" ] && continue
  [ "${a,,}" = "kevin" ] || { bad="$a"; break; }
done < <(git log --no-merges --format=%an -- tests/known)
if [ -n "$bad" ]; then
  echo "FAIL: tests/known committed by non-kevin author: $bad" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- tests/known; then
  echo "FAIL: tests/known has uncommitted changes" >&2
  exit 1
fi
echo "OK: tests/known intact (kevin-only history, clean worktree)"
