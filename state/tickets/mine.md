# mine — Mine defect candidates

## Issue
TODO.md (id: mine): build `loopgym/data/mine.py` over `data/raw/repos/{draftlab,harness}.git`.
Fix commits paired with the commit that introduced the lines they delete; plus the 3 seeds in
`data/raw/ticket_review.json`. Output `data/loopgym/candidates.jsonl` + `results/mining.json`.

## Question
Which commits are fixes, which commit is "bad" for each fix, and which (bad, good) pairs are
small enough to become RL tasks — such that the 3 human-confirmed seed defects are also found
by the automatic rules?

## Evidence (measured on the raw repos, /tmp/proto*.py)
- 197+54 non-merge commits; keyword (fix|correct|wrong|bug|audit, full message incl. TODO: lines):
  draftlab 38, harness 9 fix commits.
- Pairing per introducing commit (a fix's deleted lines blame to several committers): floor ≥1 line
  → 147 (bad, good) pairs; ≥2 lines → 98. Majority-only pairing recalls only 1/3 seeds — rejected.
- Seed recall via auto rules (keyword + blame on the fix, no size gate): alpha via 0158d68f (1 line,
  also 7a848915/554, ca4403cb/150, 5bf9b0db/10); spec via ae371e3c (4, also 1e9f5d39/66); lora via
  11b1d163 (11). 3/3.
- Size filter: seed bads are 432–1755 lines and the lora fix commit is 1247 lines/5 files (31 source
  lines), so filtering fix-commit or bad-commit size kills recall (0–2/3) and caps candidates at
  27–31. The only reading where "keep diffs ≤ 200 lines / ≤ 4 files" coexists with ≥60 candidates
  and 3/3 recall: filter the candidate's defect diff = lines bad introduced that good reverted
  (this is also the per-candidate `lines`/`files` field). It drops 1 pair (554-line) in draftlab.
- Ticket text: from the bad commit's tree TODO.md — line matching `(id: <first subject
  token>)`, else the unique task line the subject copies (>=12-char prefix match, both
  directions); else a TODO: line in the bad message (harness has no TODO.md); else "".
  50/141 have ticket text; the rest are result/interrupted/renamed commits with none.

## Plan
1. `loopgym/data/mine.py`: scan `rev-list --all --no-merges`; keyword on full %B; parse
   `diff -U0 p F` deleted ranges; `blame -L s,e p`; per (bad, n≥1) candidate with defect
   files; keep n ≤ 200 and ≤ 4 files; dedupe (repo, bad, good), seed records win.
2. Add the 3 seeds unconditionally (their own bad/good, ticket from ticket_review.json; lines/
   files computed by the same blame). Ids: seeds as given, auto `{repo}-{bad7}-{good7}`.
3. Write candidates.jsonl sorted by (lines desc, id) [downstream takes top-N "by score"],
   results/mining.json (per repo: commits, kw fixes, pairs, dropped by lines/files, kept, seeds;
   recall detail), print per-seed rule + "N/3 seeds recalled"; exit 0.
4. tests/test_mine.py: keyword regex; hunk-range parsing; end-to-end on a tmp git repo; seed
   recall against the real repos.

## Done when
`.venv/bin/python -m loopgym.data.mine` exits 0, writes ≥ 60 candidates, prints 3/3 seeds
recalled with the catching rule; tests pass; ruff clean.

## Stop when
Recall < 3/3 under the rules above (report which seed and why), or candidate count < 60 —
record in NOTES.md and RESULT: not done. No hidden checks, no verify.py (next ticket).
