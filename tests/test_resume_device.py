"""A resumed policy must land on trainer_device like a fresh one does.

restore() rebuilt the policy on the CPU and never moved it, so the GPU run's retry
trained on the CPU and the kernel shim copied each batch onto cuda:0 (vLLM's GPU),
which ran out of memory. The sandbox has no GPU, so this spies on Module.to."""
import torch

from loopgym.rl import train as T

from test_train import NEED_MODEL, write_cfg


@NEED_MODEL
def test_restore_moves_policy_to_trainer_device(tmp_path, monkeypatch):
    cfg = T.load_cfg(str(write_cfg(tmp_path)), {"out": str(tmp_path / "run"), "trainer_device": "cpu"})
    run_ = T.Run(cfg)
    ck = T.ckpt_dir(cfg)
    run_.save(ck)
    fresh = T.Run(cfg)  # __init__ moves its own policy; only restore() is under test

    moved = []
    orig = torch.nn.Module.to

    def spy(self, *a, **k):
        if type(self).__name__.startswith("PeftModel"):
            moved.append(a[0] if a else k.get("device"))
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.nn.Module, "to", spy)
    fresh.restore(ck)
    assert "cpu" in moved, f"restored policy never moved to trainer_device: {moved}"
