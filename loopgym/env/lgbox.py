"""lgbox (Rust sandbox server) client and the "http" grade runner.

grade(..., runner="http") sends the pristine test files plus the model's solution file
to POST /run, runs `python -m pytest ... --junitxml=r.xml` inside the sandbox, fetches
r.xml back, and scores it with the same function as the local runner (exercism.py).
runner_names() exposes "http" only when LOOPGYM_LGBOX_URL is set.
"""
import os
import sys
import time

import httpx

from .exercism import RUNNERS, _reward, _xml_counts

# Address-space cap for a run; generous enough for pytest on a plain exercise.
MEM_MB = 2048
# Max total seconds to wait out 503 "queue full" before giving up.
RETRY_BUDGET_S = 120.0


class LgboxError(RuntimeError):
    pass


class LgboxClient:
    def __init__(self, base_url, mem_mb=MEM_MB, retry_budget_s=RETRY_BUDGET_S,
                 transport=None):
        self.base_url = base_url.rstrip("/")
        self.mem_mb = mem_mb
        self.retry_budget_s = retry_budget_s
        self.transport = transport

    def run(self, files, cmd, timeout_s, outputs=()):
        """POST /run; retry 503 honoring Retry-After until retry_budget_s is spent."""
        payload = {"files": files, "cmd": cmd, "timeout_s": timeout_s,
                   "mem_mb": self.mem_mb, "outputs": list(outputs)}
        # The server holds the request until the run finishes: read timeout must cover it.
        to = httpx.Timeout(connect=5.0, read=float(timeout_s) + 30.0, write=10.0, pool=5.0)
        deadline = time.monotonic() + self.retry_budget_s
        with httpx.Client(base_url=self.base_url, timeout=to,
                          transport=self.transport) as c:
            while True:
                r = c.post("/run", json=payload)
                if r.status_code == 503:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise LgboxError(f"lgbox {self.base_url} queue full "
                                         f"({self.retry_budget_s:.0f}s retry budget spent)")
                    time.sleep(min(float(r.headers.get("Retry-After") or 1), left))
                    continue
                if r.status_code >= 400:
                    raise LgboxError(f"lgbox /run -> HTTP {r.status_code}: {r.text[:200]}")
                return r.json()


def client():
    url = os.environ.get("LOOPGYM_LGBOX_URL")
    if not url:
        raise LgboxError("LOOPGYM_LGBOX_URL is not set (start `lgbox serve` and set it)")
    return LgboxClient(url)


def _http(task, files, timeout=30):
    info = {"runner": "http", "timed_out": False, "returncode": None,
            "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    payload = {name: (task.dir / name).read_text() for name in task.test_files}
    sol = files.get(task.solution_file)
    if sol is not None:
        payload[task.solution_file] = sol
    out = client().run(payload,
                       [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--junitxml=r.xml"],
                       timeout, outputs=["r.xml"])
    info["timed_out"] = bool(out["timed_out"])
    info["returncode"] = out["exit_code"]
    xml = (out.get("outputs") or {}).get("r.xml")
    if xml:
        info.update(_xml_counts(xml))
    return _reward(info)


RUNNERS["http"] = _http
