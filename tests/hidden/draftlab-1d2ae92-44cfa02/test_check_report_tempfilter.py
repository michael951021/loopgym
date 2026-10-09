"""Hidden check draftlab-1d2ae92-44cfa02: tools/check_report.py exits 0 at the fix commit
and computes the T=0 no-drafting baseline from T=0 rows only (bad commit pooled both
temps into none_ts[0.0], giving a wrong T=0 speedup). Synthetic root; CPU, seconds."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import polars as pl

WT = next(Path(p) for p in sys.path if p and (Path(p) / "tools" / "check_report.py").exists())

def run(root: Path) -> int:
    p = subprocess.run([sys.executable, str(root / "tools" / "check_report.py")], capture_output=True,
                       text=True, cwd=root, timeout=100)
    assert p.returncode in (0, 1), (p.stdout + p.stderr)[-1500:]
    return p.returncode

def test_checker_exit0_at_fix_commit():
    assert run(WT) == 0, "checker must exit 0 at the fix commit"

def build(root: Path) -> Path:
    (root / "tools").mkdir(parents=True)
    shutil.copy2(WT / "tools" / "check_report.py", root / "tools" / "check_report.py")
    fig = root / "results" / "fig"
    fig.mkdir(parents=True)
    for n in "abc":
        (fig / f"{n}.png").write_bytes(b"")
    # none baselines T0 4 tok/s vs T0.7 16 (pooled would be 6.4); REPORT states T=0-only:
    # 4.00 / 16.00, ngram k8 speedup 20/4 = 5.0000, T0.7 30/16 = 1.8750.
    rows = [("base", k, 0.0, s, s, 16, s, 16, 1.0) for k, s in ((1, 8), (2, 6), (4, 4), (8, 2))]
    rows += [("none", 1, 0.0, 0, 0, 16, 16, 16, 4.0), ("none", 1, 0.7, 0, 0, 16, 16, 16, 1.0),
             ("ngram", 1, 0.7, 3295, 10000, 160, 500, 16, 1.0),
             ("ngram", 8, 0.0, 6, 10, 400, 100, 16, 20.0), ("ngram", 8, 0.7, 5, 10, 300, 100, 16, 10.0),
             ("lora", 8, 0.0, 7, 10, 16, 1, 16, 1.0), ("lora", 8, 0.7, 65, 100, 16, 1, 16, 1.0),
             ("lora_dpo", 8, 0.0, 75, 100, 16, 1, 16, 1.0), ("lora_dpo", 8, 0.7, 7, 10, 16, 1, 16, 1.0),
             ("tiny", 1, 0.0, 5, 10, 16, 1, 16, 1.0)]
    pl.DataFrame(rows, schema=["drafter", "k", "temp", "n_acc", "n_draft", "n_emit", "n_steps", "n_gen",
                               "wall_s"]).write_parquet(root / "results" / "sweep.parquet")
    alphas = {"results/alpha_sweep_ngram.json": 0.3333, "results/alpha_sweep_base.json": 0.91,
              "results/alpha_sweep_lora.json": 0.62, "results/alpha_sweep_lora_dpo.json": 0.67,
              "results/alpha/ngram.json": 0.345, "results/alpha/tiny.json": 0.41,
              "results/alpha/lora.json": 0.63, "results/alpha/dpo.json": 0.68}
    for p, a in alphas.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(json.dumps({"overall": {"alpha": a}}))
    (root / "results" / "judge.json").write_text(json.dumps(
        {"spec_vs_drafter_alone": {"win_rate": 0.95}, "spec_vs_target": {"win_rate": 0.99}}))
    (root / "REPORT.md").write_text(
        "# Question: corpus method judge\nfailed on gpu\n"
        "0.3295 0.6000 0.7000 0.6500 0.7500 0.5000 4.00 16.00 5.0000 1.8750\n"
        "0.3333 0.9100 0.6200 0.6700 0.3450 0.4100 0.6300 0.6800 0.950 0.990 a.png b.png c.png")
    (root / "README.md").write_text(
        "```mermaid\ngraph TD\nA-->B\n```\nQuickstart. JAX PyTorch FSDP LoRA DPO Triton Ray Go\n"
        "Prometheus Grafana Kubernetes Helm Terraform React LLM-as-judge OpenAI")
    return root

def test_t0_baseline_uses_only_temp0_rows(tmp_path):
    assert run(build(tmp_path)) == 0, "checker rejected the T=0-only no-draft baseline"
