"""Site hook for the DeepSpeed CPU tests: disable the JIT shared-memory comm op.

Runs at interpreter startup for torchrun workers whose PYTHONPATH carries this
directory (the in-process sys.modules trick from tests/test_deepspeed_config.py
cannot reach a subprocess). The pinned DeepSpeed's build_shm_op
(.tools/src/deepspeed-0.19.7/deepspeed/comm/torch.py:47) checks this flag and
returns None, so the engine falls back to plain torch.distributed collectives.
"""
try:
    import deepspeed.ops as _dsops
    _dsops.__compatible_ops__["deepspeed_shm_comm"] = False
except Exception:
    pass
