"""VLLMBackend wired into the train loop: engine args, per-step LoRA sync,
resume re-sync, backend validation. Sandbox-safe: tiny model + a FakeLLM whose
token ids fit the tiny vocab (512), vLLM never imported."""
from pathlib import Path

import pytest

from loopgym.rl import rollout_vllm as rv
from loopgym.rl.train import Run, TrainCfg, ckpt_dir
from test_vllm_api import (FakeCompletion, FakeLLM, FakeLogprob, FakeLoRARequest,
                           FakeRequestOutput, FakeSamplingParams)


class TinyIdsLLM(FakeLLM):
    """FakeLLM, but ids stay inside the tiny model's vocab (512) so the trainer
    can embed them; prompt ids are fixed per prompt."""

    def generate(self, prompts, sampling_params, *, use_tqdm=True, lora_request=None):
        self.calls.append({"sp": sampling_params, "lora_request": lora_request})
        outs = []
        for p_idx, p in enumerate(prompts):
            comps = []
            for j in range(sampling_params.n):
                ids = [5 + p_idx * 10 + j * 3 + i for i in range(3)]
                lp = [{t: FakeLogprob(-(1.0 + 0.1 * i))} for i, t in enumerate(ids)]
                comps.append(FakeCompletion(j, ids, " ".join(map(str, ids)), lp))
            outs.append(FakeRequestOutput([1, 2, 3], comps))
        return outs


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(rv, "SamplingParams", FakeSamplingParams)
    monkeypatch.setattr(rv, "LoRARequest", FakeLoRARequest)
    built = {}

    def fake_build(args):
        built["args"] = args
        return TinyIdsLLM(None)

    monkeypatch.setattr(rv, "build_llm", fake_build)
    return built


def cfg(tmp_path, **kw):
    base = dict(model="tiny", task="toy", toy_task="banana", backend="vllm",
                batch=1, group=2, max_new_tokens=16, lr=0.01, steps=2,
                save_every=1, seed=42, out=str(tmp_path / "runs" / "vllm"))
    base.update(kw)
    return TrainCfg(**base)


def one_step(r):
    tasks = r.tasks()
    rec = r.run_step(r.step + 1, tasks, [t.prompt() for t in tasks])
    r.step = r.step + 1
    return rec


def test_backend_selection_and_engine_args(fakes, tmp_path):
    r = Run(cfg(tmp_path, lora_r=8))
    assert isinstance(r.backend, rv.VLLMBackend)
    a = fakes["args"]
    assert a["model"] == "tiny"
    assert a["enable_lora"] is True and a["max_loras"] == 1
    assert a["max_lora_rank"] == 8
    assert a["dtype"] == "bfloat16"


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown backend"):
        Run(cfg(tmp_path, backend="weird"))


def test_first_generation_unlora_then_sync(fakes, tmp_path):
    r = Run(cfg(tmp_path))
    one_step(r)
    llm = r.backend.llm
    assert llm.calls[0]["lora_request"] is None  # trained before any sync
    adp = Path(r.cfg.out) / "vllm_adapter"
    assert (adp / "adapter_config.json").is_file()
    assert (adp / "adapter_model.safetensors").is_file()
    req = r.backend._lora
    assert req.lora_name == "step_1" and req.lora_path == str(adp)
    assert req.load_inplace is True


def test_next_generation_uses_previous_step_lora(fakes, tmp_path):
    r = Run(cfg(tmp_path))
    one_step(r)
    one_step(r)
    llm = r.backend.llm
    req1 = llm.calls[1]["lora_request"]
    assert req1.lora_name == "step_1" and req1.lora_int_id == 1
    req2 = r.backend._lora
    assert req2.lora_name == "step_2" and req2.lora_int_id == 2


def test_resume_resyncs_lora(fakes, tmp_path):
    c = cfg(tmp_path)
    r = Run(c)
    one_step(r)
    r.save(ckpt_dir(c))
    r2 = Run(c)
    r2.restore(ckpt_dir(c))
    assert r2.backend._lora.lora_name == "step_1"
    one_step(r2)
    assert r2.backend.llm.calls[0]["lora_request"].lora_name == "step_1"
