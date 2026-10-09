"""Mine defect candidates from draftlab/harness history.

A fix commit is a non-merge commit whose message (incl. its TODO: lines) says
fix/correct/wrong/bug/audit. Its bad commit is whoever introduced the lines the fix
deletes (git blame of those lines at the fix's parent). Each introducing commit
yields one candidate (bad, good=fix); lines/files count the defect diff = the bad's
lines the fix reverts, and candidates are kept at <= 200 lines / <= 4 files. The 3
seeds in data/raw/ticket_review.json are always added. Recall: the automatic rules
must pair each seed's bad with some fix commit.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOS = {n: ROOT / "data" / "raw" / "repos" / f"{n}.git" for n in ("draftlab", "harness")}
SEEDS = ROOT / "data" / "raw" / "ticket_review.json"
OUT_DIR = ROOT / "data" / "loopgym"
MAX_LINES, MAX_FILES = 200, 4
KW = re.compile(r"\b(?:fix(?:ed|es)?|correct(?:ed|s)?|wrong|bug(?:s|gy)?|audit(?:ed|s)?)\b", re.I)
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout


_RESOLVED = {}


def resolve(repo, ref):
    key = (str(repo), ref)
    if key not in _RESOLVED:
        _RESOLVED[key] = git(repo, "rev-list", "-n", "1", ref).strip()
    return _RESOLVED[key]


def parent(repo, c):
    par = git(repo, "rev-list", "--parents", "-n", "1", c).split()
    return par[1] if len(par) > 1 else EMPTY_TREE


def deleted_ranges(repo, c):
    """(file, start, count) of deleted line ranges, in the parent's coordinates."""
    diff = git(repo, "diff", "-U0", parent(repo, c), c)
    cur, ranges = None, []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            cur = line.rsplit(" b/", 1)[1]
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
            cnt = int(m.group(2)) if m.group(2) is not None else 1
            if cnt:
                ranges.append((cur, int(m.group(1)), cnt))
    return ranges


def blame_deleted(repo, c):
    """{introducing commit: [lines, {files}]} for every line c deletes."""
    p = parent(repo, c)
    short = {}
    for f, s, cnt in deleted_ranges(repo, c):
        try:
            blame = git(repo, "blame", "-L", f"{s},{s + cnt - 1}", p, "--", f)
        except subprocess.CalledProcessError:
            continue
        for line in blame.splitlines():
            h = line.split()[0].lstrip("^")  # ^ marks boundary commits
            if set(h) == {"0"}:
                continue
            try:
                full = resolve(repo, h)
            except subprocess.CalledProcessError:  # outside this history
                continue
            rec = short.setdefault(full, [0, set()])
            rec[0] += 1
            rec[1].add(f)
    return short


def ticket_for(repo, bad):
    """Ticket text that produced the bad commit: the TODO.md line in its tree, matched
    by (id: <first subject token>) or as the unique task line the subject copies, else
    a TODO: line in its message, else '' (result/interrupted commits have none)."""
    subj = git(repo, "log", "-1", "--format=%s", bad).strip()
    tid = subj.split(":", 1)[0].strip()
    try:
        lines = git(repo, "show", f"{bad}:TODO.md").splitlines()
    except subprocess.CalledProcessError:
        lines = []
    for line in lines:
        if f"(id: {tid})" in line:
            return line.strip()
    tail = subj.split(":", 1)[1].strip() if ":" in subj else subj
    cands = []
    for line in lines:
        m = re.match(r"-\s*\[[ x]\]\s*(.*)", line)
        if m and ((len(tail) >= 12 and m.group(1).startswith(tail))
                  or (len(m.group(1)) >= 12 and tail.startswith(m.group(1)))):
            cands.append(line)
    if len(cands) == 1:
        return cands[0].strip()
    for line in git(repo, "log", "-1", "--format=%B", bad).splitlines():
        if line.strip().startswith("TODO:"):
            return line.strip()
    return ""


def mine_repo(repo, name):
    """Auto candidates from one bare repo. Returns (records, stats, hits: bad ->
    [(fix[:8], lines), ...])."""
    commits = git(repo, "rev-list", "--all", "--no-merges").split()
    stats = Counter(commits=len(commits), kw_fixes=0, pairs=0, kept=0,
                    dropped_lines=0, dropped_files=0)
    records, hits = {}, {}
    for c in commits:
        msg = git(repo, "log", "-1", "--format=%B", c)
        if not KW.search(msg):
            continue
        stats["kw_fixes"] += 1
        good = c
        fix_msg = msg.splitlines()[0]
        for bad, (n, files) in blame_deleted(repo, c).items():
            stats["pairs"] += 1
            hits.setdefault(bad, []).append((c[:8], n))
            if n > MAX_LINES:
                stats["dropped_lines"] += 1
                continue
            if len(files) > MAX_FILES:
                stats["dropped_files"] += 1
                continue
            stats["kept"] += 1
            cid = f"{name}-{bad[:7]}-{good[:7]}"
            records[cid] = {
                "id": cid, "repo": name, "bad": bad, "good": good,
                "files": sorted(files), "fix_msg": fix_msg,
                "ticket": ticket_for(repo, bad), "lines": n, "seed": False,
            }
    return records, dict(stats), hits


def seed_records():
    """The 3 human-confirmed seeds, with lines/files measured the same way."""
    out = []
    for s in json.loads(SEEDS.read_text()):
        repo = REPOS[s["project"]]
        bad = resolve(repo, s["bad"])
        good = resolve(repo, s["good"])
        lines, files = 0, set()
        for b, (n, fs) in blame_deleted(repo, good).items():
            if b == bad:
                lines, files = n, fs
        out.append({
            "id": s["id"], "repo": s["project"], "bad": bad, "good": good,
            "files": sorted(files), "fix_msg": git(repo, "log", "-1", "--format=%s", good),
            "ticket": s["ticket"], "lines": lines, "seed": True,
            "defects": s.get("defects", []),
        })
    return out


def main():
    records = {}
    per_repo, recalls = {}, {}
    for name, repo in REPOS.items():
        recs, stats, hits = mine_repo(repo, name)
        per_repo[name] = stats
        recalls[name] = hits
        records.update(recs)
    for s in seed_records():
        auto = f"{s['repo']}-{s['bad'][:7]}-{s['good'][:7]}"
        per_repo[s["repo"]]["seed_and_auto"] = int(auto in records)
        records.pop(auto, None)  # seed record wins on (bad, good) collision
        records[s["id"]] = s
        per_repo[s["repo"]]["seeds"] = per_repo[s["repo"]].get("seeds", 0) + 1

    cands = sorted(records.values(), key=lambda r: (-r["lines"], r["id"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "candidates.jsonl").open("w") as f:
        for r in cands:
            f.write(json.dumps(r) + "\n")

    seed_detail = {}
    n_recalled = 0
    for s in seed_records():
        hits = []
        for hits_by_bad in recalls.values():
            hits.extend(hits_by_bad.get(s["bad"], []))
        found = bool(hits)
        n_recalled += found
        seed_detail[s["id"]] = {"recalled": found,
                                "rule": "keyword in fix message + blame of deleted lines"
                                if found else "",
                                "fixes": sorted(hits, key=lambda h: -h[1])}
    doc = {"repos": per_repo, "candidates": len(cands), "seeds": seed_detail,
           "recalled": f"{n_recalled}/3"}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "mining.json").write_text(json.dumps(doc, indent=1) + "\n")

    for name, st in per_repo.items():
        print(f"{name}: {st['commits']} commits, {st['kw_fixes']} keyword fixes, "
              f"{st['pairs']} pairs, kept {st['kept']} "
              f"(-{st['dropped_lines']} >{MAX_LINES} lines, "
              f"-{st['dropped_files']} >{MAX_FILES} files), "
              f"+{st.get('seeds', 0)} seeds")
    for s in seed_records():
        d = seed_detail[s["id"]]
        if d["recalled"]:
            via = ", ".join(f"{c} ({n} line{'s' if n != 1 else ''})" for c, n in d["fixes"])
            print(f"seed {s['id']}: recalled by keyword + blame, via {via}")
        else:
            print(f"seed {s['id']}: NOT recalled by automatic rules")
    print(f"{n_recalled}/3 seeds recalled; {len(cands)} candidates -> "
          f"data/loopgym/candidates.jsonl")
    ok = n_recalled == 3 and len(cands) >= 60
    if not ok:
        print("GATE FAILED: need 3/3 recall and >= 60 candidates", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
