"""env.splits: sha1 bucketing, split completeness, and a real mini-audit."""
import hashlib
import json

from loopgym.env.splits import audit, bucket, split_names


def test_sha1_known_value_and_buckets():
    # published sha1("hello"); int(hex, 16) % 100 == 81 -> val
    assert hashlib.sha1(b"hello").hexdigest() == "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
    assert bucket("hello") == "val"
    assert bucket("leap") == "train"   # 41
    assert bucket("bob") == "test"     # 90


def test_split_disjoint_and_complete():
    names = [f"ex-{i}" for i in range(37)]
    out = split_names(names)
    flat = out["train"] + out["val"] + out["test"]
    assert sorted(flat) == sorted(names)
    assert len(flat) == len(set(flat))
    for n in flat:
        h = int(hashlib.sha1(n.encode()).hexdigest(), 16) % 100
        want = "train" if h < 70 else ("val" if h < 85 else "test")
        assert n in out[want]


def _mk_exercise(root, name, ref, stub):
    d = root / name
    (d / ".meta").mkdir(parents=True)
    (d / ".meta" / "config.json").write_text(json.dumps(
        {"files": {"solution": [f"{name}.py"], "test": [f"{name}_test.py"],
                   "example": [".meta/example.py"]}}))
    (d / ".meta" / "example.py").write_text(ref)
    (d / f"{name}.py").write_text(stub)
    (d / f"{name}_test.py").write_text(
        f"from {name} import f\n\ndef test_f():\n    assert f(2) == 4\n")
    return d


def test_mini_audit_real_grades(tmp_path):
    good_ref = "def f(x):\n    return x * 2\n"
    bad_ref = "def f(x):\n    return x + 1\n"
    stub = "def f(x):\n    return 0\n"
    _mk_exercise(tmp_path, "good", good_ref, stub)          # usable
    _mk_exercise(tmp_path, "badref", bad_ref, stub)         # ref fails
    _mk_exercise(tmp_path, "leaky", bad_ref,
                 "def f(x):\n    return x * 2\n")           # stub passes -> leak
    (tmp_path / "leaky" / "leaky_test.py").write_text(
        "def test_always():\n    assert True\n")
    recs = audit(tmp_path, workers=2)
    assert recs["good"]["usable"] is True
    assert recs["badref"]["usable"] is False
    assert "ref=0.0" in recs["badref"]["reason"]
    assert recs["leaky"]["usable"] is False
    assert "stub=1.0" in recs["leaky"]["reason"]
