"""Hidden check draftlab-7953937-d82b39e: the dl-report merge resolution left
`make check` green. Behaviour restored (the 3 reds):
(a) terraform gke module: every google_* resource reference resolves to a
    declared resource with a valid HCL name (SA label matched its refs);
(b) tools/score_engine_trajectory.py has no line over the ruff line-length;
(c) web measured-best-speedup KPI skips the none baseline arm; its registered
    range brackets the best non-baseline arm of the real sweep export."""
import json
import re
import sys
from pathlib import Path

WT = next((Path(p) for p in sys.path
           if p and (Path(p) / "deploy" / "terraform" / "gke").is_dir()
           and (Path(p) / "web" / "public" / "data" / "sweep.json").is_file()),
          Path.cwd())
GKE = WT / "deploy" / "terraform" / "gke"


def test_terraform_gke_refs_resolve():
    text = "\n".join(f.read_text() for f in sorted(GKE.glob("*.tf")))
    decl = set(re.findall(r'resource "(\w+)" "([^"]+)"', text))
    refs = set(re.findall(r"(google_[a-z_]+)\.([a-z0-9_]+)", text))
    assert refs, "no google_* resource references found in gke module"
    for t, n in refs:
        assert (t, n) in decl, f"unresolved reference {t}.{n} (no such resource)"
    for t, n in decl:
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", n), f"invalid HCL identifier {t}.{n}"


def test_score_engine_lines_within_ruff_limit():
    limit = 88
    m = re.search(r"line-length\s*=\s*(\d+)", (WT / "pyproject.toml").read_text())
    if m:
        limit = int(m.group(1))
    lines = (WT / "tools" / "score_engine_trajectory.py").read_text().splitlines()
    bad = [(i + 1, len(ln)) for i, ln in enumerate(lines) if len(ln) > limit]
    assert not bad, f"lines over {limit} chars (ruff E501): {bad[:3]}"


def test_measured_speedup_kpi_excludes_none_baseline():
    cells = json.loads(
        (WT / "web" / "public" / "data" / "sweep.json").read_text())["cells"]
    done = [c["speedup_vs_none"] for c in cells.values()
            if c["status"] == "done" and c.get("speedup_vs_none") is not None]
    mx_nb = max(c["speedup_vs_none"] for n, c in cells.items()
                if not n.startswith("none") and c["status"] == "done"
                and c.get("speedup_vs_none") is not None)
    assert max(done) > mx_nb, "none arm is the 1.0x reference; excluding it must change the max"
    page = (WT / "web" / "src" / "pages" / "Overview.tsx").read_text()
    i = page.index("measSp")
    seg = page[max(0, i - 300): i + 700]
    assert re.search(r"""['"]none['"]""", seg), \
        "measured-speedup tile must skip the 'none' baseline arm"
    test = (WT / "web" / "src" / "tests" / "overview.test.tsx").read_text()
    seg2 = test.split("measured best speedup", 1)[1].split("});", 1)[0]
    lo = float(re.search(r"toBeGreaterThan\(([\d.]+)\)", seg2).group(1))
    hi = float(re.search(r"toBeLessThan\(([\d.]+)\)", seg2).group(1))
    assert lo < mx_nb < hi, \
        f"registered range ({lo}, {hi}) does not bracket best non-baseline arm {mx_nb:.4f}"
