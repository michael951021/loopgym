"""rollout_vllm.py vs the pinned vLLM (tools/versions.txt), checked with ast.

vLLM cannot run here (rule 2: no GPU, not installed in .venv). Rule 6: every
vLLM name/attribute/kwarg the backend uses must exist in the pinned source
.tools/src/vllm-<ver>/. Usage is extracted from rollout_vllm.py's own source
(engine_args keys, SamplingParams/LoRARequest kwargs, llm.generate kwargs) and
asserted against an ast parse of the pinned tree. Backend logic runs against a
FakeLLM; --dry-run must print the engine args without importing vLLM."""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
VLLM_DIR = next((ROOT / ".tools" / "src").glob("vllm-*"))
SRC = (ROOT / "loopgym" / "rl" / "rollout_vllm.py").read_text()
BACKEND_TREE = ast.parse(SRC, filename="rollout_vllm.py")

sys.path.insert(0, str(ROOT))
from loopgym.rl import rollout_vllm as rv  # noqa: E402


# --- ast helpers over the pinned vLLM source --------------------------------

def _tree(relpath: str) -> ast.Module:
    p = VLLM_DIR / relpath
    return ast.parse(p.read_text(), filename=str(p))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not in pinned vLLM {VLLM_DIR.name}")


def _fields(cls: ast.ClassDef) -> set:
    """Class-level field names (dataclass / msgspec.Struct assignments)."""
    names = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    return names


def _init(cls: ast.ClassDef) -> ast.FunctionDef:
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            return stmt
    raise AssertionError(f"{cls.name} has no __init__")


def _params(func: ast.FunctionDef) -> tuple[set, bool]:
    a = func.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    return names, a.kwarg is not None


def _self_attrs(func: ast.FunctionDef) -> set:
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                names.add(t.attr)
    return names


# --- what rollout_vllm.py uses, extracted from its own source ---------------

def _backend_engine_keys() -> set:
    for node in BACKEND_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "engine_args":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    return {k.value for k in sub.keys if isinstance(k, ast.Constant)}
    raise AssertionError("engine_args() not found in rollout_vllm.py")


def _backend_calls(name: str) -> list[ast.Call]:
    return [n for n in ast.walk(BACKEND_TREE)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _backend_llm_generate_kwargs() -> set:
    names = set()
    for n in ast.walk(BACKEND_TREE):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "llm"):
            names.update(kw.arg for kw in n.keywords if kw.arg is not None)
    return names


# --- API checks: backend usage must exist in the pinned version -------------

def test_top_level_exports():
    attrs = set()
    for node in _tree("vllm/__init__.py").body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == "MODULE_ATTRS" for t in node.targets):
                attrs = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert {"LLM", "SamplingParams"} <= attrs, attrs


def test_engine_args_keys_exist():
    """Every LLM(**engine_args) kwarg is an LLM.__init__ param or an EngineArgs field."""
    llm_cls = _class(_tree("vllm/entrypoints/llm.py"), "LLM")
    llm_params, forwards_kwargs = _params(_init(llm_cls))
    engine_fields = _fields(_class(_tree("vllm/engine/arg_utils.py"), "EngineArgs"))
    keys = _backend_engine_keys()
    assert forwards_kwargs, "LLM.__init__ must forward **kwargs to the engine"
    missing = {k for k in keys if k not in llm_params and k not in engine_fields}
    assert not missing, f"engine args not in pinned vLLM: {sorted(missing)}"
    assert {"enable_lora", "max_loras", "max_lora_rank"} <= engine_fields
    assert "model" in llm_params


def test_llm_generate_kwargs_exist():
    llm_cls = _class(_tree("vllm/entrypoints/llm.py"), "LLM")
    gen = None
    for stmt in llm_cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "generate":
            gen = stmt
    assert gen is not None
    params, _ = _params(gen)
    used = _backend_llm_generate_kwargs()
    assert used, "backend must call llm.generate(...)"
    missing = used - params
    assert not missing, f"llm.generate kwargs not in pinned vLLM: {sorted(missing)}"


def test_sampling_params_kwargs_exist():
    sp_fields = _fields(_class(_tree("vllm/sampling_params.py"), "SamplingParams"))
    calls = _backend_calls("SamplingParams")
    assert calls, "backend must construct SamplingParams(...)"
    used = set()
    for c in calls:
        kw = {k.arg for k in c.keywords if k.arg is not None}
        used |= kw
        # pinned 0.31.0 defaults max_tokens to 16, so the backend must set it
        assert "max_tokens" in kw, "max_tokens default in pinned vLLM is 16; pass it explicitly"
        assert "logprobs" in kw, "sampler_logp needs logprobs=0"
    missing = used - sp_fields
    assert not missing, f"SamplingParams kwargs not in pinned vLLM: {sorted(missing)}"


def test_lora_request_kwargs_exist():
    lr_fields = _fields(_class(_tree("vllm/lora/request.py"), "LoRARequest"))
    calls = _backend_calls("LoRARequest")
    assert calls, "backend must construct LoRARequest(...)"
    used = set()
    for c in calls:
        used |= {k.arg for k in c.keywords if k.arg is not None}
    assert {"lora_name", "lora_path"} <= used
    missing = used - lr_fields
    assert not missing, f"LoRARequest kwargs not in pinned vLLM: {sorted(missing)}"


def test_output_attribute_access_exists():
    """Attributes the backend reads off vLLM result objects exist in the pinned source."""
    co = _fields(_class(_tree("vllm/outputs.py"), "CompletionOutput"))
    assert {"index", "token_ids", "text", "logprobs"} <= co, co
    ro = _self_attrs(_init(_class(_tree("vllm/outputs.py"), "RequestOutput")))
    assert {"prompt_token_ids", "outputs"} <= ro, ro
    lp = _fields(_class(_tree("vllm/logprobs.py"), "Logprob"))
    assert "logprob" in lp, lp


# --- FakeLLM unit tests -------------------------------------------------------

class FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob
        self.rank = 1
        self.decoded_token = None


class FakeCompletion:
    def __init__(self, index, token_ids, text, logprobs):
        self.index = index
        self.token_ids = token_ids
        self.text = text
        self.logprobs = logprobs
        self.finish_reason = "stop"


class FakeRequestOutput:
    def __init__(self, prompt_token_ids, outputs):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = True


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=None, add_generation_prompt=None):
        assert tokenize is False
        return "CHAT:" + "|".join(m["content"] for m in messages)

    def encode(self, text):
        return [1 + ord(c) % 997 for c in text]


NFAKE = 3  # tokens every fake completion produces


class FakeLLM:
    """Deterministic: sample j of prompt p is tokens 1000+p*10000+j*100..+2, and
    the logprob of token i in sample j is -(1 + 0.25j + 0.125i) (prompt-blind)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def generate(self, prompts, sampling_params, *, use_tqdm=True, lora_request=None):
        self.calls.append({"prompts": prompts, "sp": sampling_params,
                           "use_tqdm": use_tqdm, "lora_request": lora_request})
        outs = []
        for p_idx, p in enumerate(prompts):
            comps = []
            for j in range(sampling_params.n):
                ids = [1000 + p_idx * 10000 + j * 100 + i for i in range(NFAKE)]
                lp = [{t: FakeLogprob(-(1.0 + 0.25 * j + 0.125 * i))} for i, t in enumerate(ids)]
                comps.append(FakeCompletion(j, ids, " ".join(map(str, ids)), lp))
            outs.append(FakeRequestOutput(self.tokenizer.encode(p), comps))
        return outs


class FakeSamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeLoRARequest:
    def __init__(self, **kw):
        self.__dict__.update(kw)


PROMPTS = [
    [{"role": "user", "content": "alpha"}],
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "beta"}],
]


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(rv, "SamplingParams", FakeSamplingParams)
    monkeypatch.setattr(rv, "LoRARequest", FakeLoRARequest)
    tok = FakeTokenizer()
    llm = FakeLLM(tok)
    return rv.VLLMBackend(llm, tok), llm, tok


def test_shape_order_and_ids(backend):
    be, llm, tok = backend
    groups = be.generate(PROMPTS, n=3, max_new_tokens=32, temperature=0.9, top_p=0.95, seed=7)
    assert [len(g) for g in groups] == [3, 3]
    for p_idx, (group, p) in enumerate(zip(groups, PROMPTS)):
        want_ids = tok.encode(tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True))
        for s, c in enumerate(group):
            assert c.prompt_ids == want_ids
            assert c.completion_ids == [1000 + p_idx * 10000 + s * 100 + i for i in range(NFAKE)]
            assert len(c.sampler_logp) == NFAKE
            assert c.text == " ".join(map(str, c.completion_ids))


def test_sampling_params_passthrough(backend):
    be, llm, tok = backend
    be.generate(PROMPTS, n=3, max_new_tokens=32, temperature=0.9, top_p=0.95, seed=7)
    call = llm.calls[0]
    sp = call["sp"]
    assert (sp.n, sp.temperature, sp.top_p, sp.max_tokens, sp.seed) == (3, 0.9, 0.95, 32, 7)
    assert sp.logprobs == 0
    # instruct-model defaults (top_k=20, rep=1.05) must not leak in
    assert sp.top_k == 0 and sp.repetition_penalty == 1.0 and sp.min_p == 0.0
    assert call["use_tqdm"] is False
    assert call["prompts"] == [
        tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in PROMPTS
    ]


def test_greedy_maps_to_zero_temperature(backend):
    be, llm, _ = backend
    be.generate(PROMPTS, n=1, max_new_tokens=8, temperature=0.0, top_p=0.9, seed=0)
    sp = llm.calls[0]["sp"]
    assert sp.temperature == 0.0 and sp.top_p == 1.0


def test_sampler_logp_values(backend):
    be, _, _ = backend
    groups = be.generate(PROMPTS, n=2, max_new_tokens=16, temperature=1.0, top_p=1.0, seed=1)
    for group in groups:
        for j in range(2):
            want = [-(1.0 + 0.25 * j + 0.125 * i) for i in range(NFAKE)]
            # every group's sample j carries sample j's logprobs
            assert group[j].sampler_logp == pytest.approx(want)


def test_lora_sync(backend, tmp_path):
    be, llm, _ = backend
    assert be.generate(PROMPTS, n=1, max_new_tokens=4, temperature=0.0, top_p=1.0, seed=0)
    assert llm.calls[0]["lora_request"] is None
    adp = tmp_path / "adp5"
    adp.mkdir()
    be.sync_lora(adp, "step_5")
    be.generate(PROMPTS, n=1, max_new_tokens=4, temperature=0.0, top_p=1.0, seed=1)
    req = llm.calls[1]["lora_request"]
    assert req.lora_name == "step_5" and req.lora_path == str(adp)
    assert req.lora_int_id >= 1 and req.load_inplace is True
    be.sync_lora(adp, "step_6")
    assert be._lora.lora_int_id == req.lora_int_id + 1


def test_engine_args_defaults():
    a = rv.engine_args("models/qwen2.5-coder-1.5b-instruct")
    assert a["model"] == "models/qwen2.5-coder-1.5b-instruct"
    assert a["enable_lora"] is True and a["max_loras"] == 1 and a["max_lora_rank"] == 16
    assert a["dtype"] == "bfloat16" and a["max_model_len"] > 0


def test_dry_run_main(backend, capsys):
    rc = rv.main(["--dry-run", "--model", "m", "--lora-path", "x", "--seed", "3"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["engine_args"]["model"] == "m"
    assert data["engine_args"]["seed"] == 3
    assert data["engine_args"]["enable_lora"] is True
    assert data["lora_request"]["lora_path"] == "x"


def test_dry_run_cli_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "loopgym.rl.rollout_vllm", "--dry-run", "--model", "m"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "engine_args" in proc.stdout
    assert json.loads(proc.stdout)["engine_args"]["max_loras"] == 1
