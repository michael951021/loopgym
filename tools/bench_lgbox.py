#!/usr/bin/env python3
"""lgbox throughput bench.

Grades reference solutions from the val split (results/splits.json) with the local
runner and the lgbox http runner at concurrency 1/4/16; per-cell grades/s and
p50/p95 latency -> results/lgbox_bench.json.

Known answer: a reference solution scores reward 1.0 (audit fact in
results/env_audit.json); the script exits 1 if any grade does not.

Usage: .venv/bin/python tools/bench_lgbox.py [--quick] [--url URL]
  --quick  20 grades per cell instead of 200 (must stay under 2 min)
  --url    reuse a running lgbox server instead of spawning one
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loopgym.env.exercism import grade, load_exercise  # noqa: E402
import loopgym.env.lgbox  # noqa: E402,F401  (registers the "http" runner)

EXDIR = ROOT / "data" / "raw" / "exercism-python" / "exercises" / "practice"
CONCURRENCIES = (1, 4, 16)
SERVER_WORKERS = 16
SERVER_QUEUE = 64
TIMEOUT = 30


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def spawn_server():
    bin_path = ROOT / "lgbox" / "target" / "release" / "lgbox"
    if not bin_path.exists():
        sys.exit(f"lgbox binary missing: {bin_path} (build: cargo build --release in lgbox/)")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [str(bin_path), "serve", "--port", str(port),
         "--workers", str(SERVER_WORKERS), "--queue", str(SERVER_QUEUE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            return url, proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    sys.exit("lgbox server did not become healthy in 15 s")


def quantile(sorted_xs, q):
    return sorted_xs[max(0, math.ceil(q * len(sorted_xs)) - 1)]


def grade_one(task, runner):
    t0 = time.monotonic()
    reward, _info = grade(task, task.reference(), runner, timeout=TIMEOUT)
    return reward, time.monotonic() - t0


def run_cell(tasks, runner, n, conc):
    rewards, lats = [], []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(grade_one, tasks[i % len(tasks)], runner) for i in range(n)]
        for f in as_completed(futs):
            r, dt = f.result()
            rewards.append(r)
            lats.append(dt)
    wall = time.monotonic() - t0
    lats.sort()
    return {
        "runner": runner, "concurrency": conc, "n": n,
        "wall_s": round(wall, 3), "grades_per_s": round(n / wall, 3),
        "lat_mean_s": round(sum(lats) / n, 4),
        "lat_p50_s": round(quantile(lats, 0.50), 4),
        "lat_p95_s": round(quantile(lats, 0.95), 4),
        "n_reward_1": sum(1 for r in rewards if r == 1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 grades per cell")
    ap.add_argument("--url", help="reuse a running lgbox server")
    args = ap.parse_args()
    n = 20 if args.quick else 200

    splits = json.loads((ROOT / "results" / "splits.json").read_text())
    tasks = [load_exercise(EXDIR / name) for name in sorted(splits["val"])]
    print(f"val split: {len(tasks)} exercises, {n} grades/cell, "
          f"concurrency {CONCURRENCIES}")

    server = None
    if args.url:
        url = args.url
    else:
        url, server = spawn_server()
    os.environ["LOOPGYM_LGBOX_URL"] = url  # the http runner reads it per call
    print(f"lgbox at {url}")

    cells = []
    try:
        for runner in ("local", "http"):
            for conc in CONCURRENCIES:
                cell = run_cell(tasks, runner, n, conc)
                cells.append(cell)
                print(f"  {runner:6s} c={conc:<2d} "
                      f"{cell['grades_per_s']:7.2f} grades/s "
                      f"p50={cell['lat_p50_s']:.3f}s p95={cell['lat_p95_s']:.3f}s "
                      f"reward1={cell['n_reward_1']}/{n}")
    finally:
        if server is not None:
            server.kill()
            server.wait()

    all_r1 = all(c["n_reward_1"] == c["n"] for c in cells)
    c1 = {c["runner"]: c["lat_mean_s"] for c in cells if c["concurrency"] == 1}
    overhead = c1.get("http", float("nan")) - c1.get("local", float("nan"))
    checks = {
        # known answer: audit fact, ref solutions score 1.0 on both runners
        "ref_reward_1_everywhere": all_r1,
        # CLAUDE.md plausible range: lgbox overhead < 50 ms per grade at c1
        "http_overhead_c1_lt_50ms": {"pass": overhead < 0.05,
                                     "value_s": round(overhead, 4)},
    }
    doc = {"mode": "quick" if args.quick else "full", "n_per_cell": n,
           "concurrencies": list(CONCURRENCIES), "timeout_s": TIMEOUT,
           "server": {"workers": SERVER_WORKERS, "queue": SERVER_QUEUE, "url": url},
           "cells": cells, "checks": checks,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = ROOT / "results" / "lgbox_bench.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    for name, chk in checks.items():
        ok = chk if isinstance(chk, bool) else chk["pass"]
        print(f"check {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all_r1 else 1


if __name__ == "__main__":
    sys.exit(main())
