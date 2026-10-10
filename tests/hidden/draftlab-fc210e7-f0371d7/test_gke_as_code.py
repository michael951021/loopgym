"""Hidden check, draftlab-fc210e7-f0371d7: deploy/terraform = GKE as code.
init -backend=false + validate exits 0; ticket shape: VPC+subnet (pods/services
secondary ranges), Workload Identity cluster, L4 GPU pool autoscale 0-2 w/
NO_SCHEDULE taint, Artifact Registry, WI service account."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
def root():
    for p in sys.path:
        d = Path(p or ".")
        if (d / "draftlab").is_dir() and (d / "tools" / "check.sh").exists():
            return d
    raise AssertionError("draftlab worktree not on sys.path")

TF = root() / "deploy" / "terraform"
_C = Path("/work/.tools/tfplugin-cache")
if _C.is_dir():
    os.environ["TF_PLUGIN_CACHE_DIR"] = str(_C)
def tf_bin():
    for c in (shutil.which("terraform"), "/work/.tools/bin/terraform"):
        if c and Path(c).is_file():
            return c
    raise AssertionError("terraform binary not found")

def hcl():
    assert TF.is_dir(), "deploy/terraform missing"
    return "\n".join(f.read_text() for f in sorted(TF.rglob("*.tf")))

def num(attr, text):
    m = re.search(attr + r"\s*=\s*(\d+|var\.(\w+))", text)
    assert m, attr
    if m.group(2):
        v = re.search(r'variable "' + m.group(2) + r'".*?default\s*=\s*([0-9.]+)', text, re.S)
        assert v, "default of " + m.group(2)
        return v.group(1)
    return m.group(1)

def test_init_validate():
    assert TF.is_dir(), "deploy/terraform missing"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tf"
        shutil.copytree(TF, p)
        for args in (["init", "-backend=false", "-input=false"], ["validate"]):
            r = subprocess.run([tf_bin()] + args, cwd=p, capture_output=True,
                               text=True, timeout=100)
            assert r.returncode == 0, (r.stdout + r.stderr)[-1200:]

def test_ticket_shape():
    t = hcl()
    for s in ('resource "google_compute_network"', 'resource "google_compute_subnetwork"',
              'resource "google_container_cluster"', 'resource "google_container_node_pool"',
              'resource "google_artifact_registry_repository"', 'resource "google_service_account"'):
        assert s in t, s
    assert t.count("secondary_ip_range") >= 2 and '"pods"' in t and '"services"' in t
    assert "workload_identity_config" in t and "nvidia-l4" in t and "NO_SCHEDULE" in t
    assert num("min_node_count", t) == "0" and num("max_node_count", t) == "2"
