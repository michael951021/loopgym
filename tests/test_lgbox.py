"""lgbox client + "http" runner: 503/Retry-After retries, payload shape, shared XML scoring
(no server needed: httpx.MockTransport). End-to-end is tools/lgbox_smoke.sh."""
import json
import sys
from pathlib import Path

import httpx
import pytest

from loopgym import env
from loopgym.env import lgbox
from loopgym.env.exercism import _reward, _xml_counts

EXERCISES = (Path(__file__).resolve().parents[1] / "data" / "raw" / "exercism-python"
             / "exercises" / "practice")

PASS_XML = ('<testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" '
            'skipped="0" time="0.1"/></testsuites>')
FAIL_XML = ('<testsuites><testsuite name="pytest" tests="3" failures="2" errors="0" '
            'skipped="0" time="0.1"/></testsuites>')


def _out(xml=None, exit_code=0, timed_out=False):
    outputs = {}
    if xml is not None:
        outputs["r.xml"] = xml
    return {"exit_code": exit_code, "timed_out": timed_out, "stdout_tail": "",
            "stderr_tail": "", "wall_ms": 1, "outputs": outputs}


def test_xml_counts_sums_suites():
    xml = ('<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0"/>'
           '<testsuite tests="3" failures="0" errors="1" skipped="2"/></testsuites>')
    assert _xml_counts(xml) == {"tests": 5, "failures": 1, "errors": 1, "skipped": 2}


def test_reward_rules():
    base = {"timed_out": False, "tests": 2, "failures": 0, "errors": 0, "skipped": 0}
    assert _reward(dict(base))[0] == 1.0
    for k, v in (("tests", 0), ("failures", 1), ("errors", 1), ("skipped", 1),
                 ("timed_out", True)):
        bad = dict(base, **{k: v})
        assert _reward(bad)[0] == 0.0


def test_runner_names_gates_http_on_env(monkeypatch):
    monkeypatch.delenv("LOOPGYM_LGBOX_URL", raising=False)
    assert env.runner_names() == ["local"]
    monkeypatch.setenv("LOOPGYM_LGBOX_URL", "http://127.0.0.1:1")
    assert env.runner_names() == ["http", "local"]


def test_client_retries_503_honoring_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(lgbox.time, "sleep", sleeps.append)
    seq = iter([httpx.Response(503, headers={"Retry-After": "2"}),
                httpx.Response(503, headers={"Retry-After": "1"}),
                httpx.Response(200, json=_out(PASS_XML))])
    transport = httpx.MockTransport(lambda r: next(seq))
    c = lgbox.LgboxClient("http://fake", transport=transport)
    out = c.run({"a.py": "pass"}, ["python", "-m", "pytest"], 30)
    assert out["outputs"]["r.xml"] == PASS_XML
    assert sleeps == [2.0, 1.0]  # slept the Retry-After value each 503


def test_client_gives_up_after_retry_budget():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(503, headers={"Retry-After": "0.05"}))
    c = lgbox.LgboxClient("http://fake", retry_budget_s=0.1, transport=transport)
    with pytest.raises(lgbox.LgboxError, match="queue full"):
        c.run({}, ["python"], 30)


def test_client_raises_on_400():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(400, json={"error": "bad path: ../x"}))
    c = lgbox.LgboxClient("http://fake", transport=transport)
    with pytest.raises(lgbox.LgboxError, match="HTTP 400"):
        c.run({"../x": "pass"}, ["python"], 30)


@pytest.mark.skipif(not EXERCISES.exists(), reason="no exercises at data/raw")
def test_http_runner_payload_and_scoring(monkeypatch):
    t = env.load_exercise(EXERCISES / "leap")
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_out(PASS_XML))

    client = lgbox.LgboxClient("http://fake", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(lgbox, "client", lambda: client)

    reward, info = env.grade(t, t.reference(), runner="http")
    assert reward == 1.0
    assert info["runner"] == "http" and info["tests"] == 3

    p = captured["payload"]
    assert set(p) == {"files", "cmd", "timeout_s", "mem_mb", "outputs"}
    # pristine test file(s) from disk + the solution file, nothing else
    assert p["files"] == {n: (t.dir / n).read_text() for n in t.test_files} | {
        t.solution_file: (t.dir / ".meta" / "example.py").read_text()}
    assert p["cmd"][0:3] == [sys.executable, "-m", "pytest"]
    assert p["cmd"][-1] == "--junitxml=r.xml"
    assert p["outputs"] == ["r.xml"] and p["timeout_s"] == 30

    # failing XML and timeouts score 0, same scorer as local
    monkeypatch.setattr(lgbox, "client", lambda: lgbox.LgboxClient(
        "http://fake", transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_out(FAIL_XML)))))
    assert env.grade(t, t.reference(), runner="http")[0] == 0.0
    monkeypatch.setattr(lgbox, "client", lambda: lgbox.LgboxClient(
        "http://fake", transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_out(timed_out=True)))))
    assert env.grade(t, t.stub(), runner="http")[0] == 0.0
    monkeypatch.setattr(lgbox, "client", lambda: lgbox.LgboxClient(
        "http://fake", transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_out(exit_code=1)))))  # no r.xml
    assert env.grade(t, t.stub(), runner="http")[0] == 0.0
