"""Test stub: einops names DeepSpeed imports; unused on the CPU test path (loud if hit)."""


def rearrange(tensor, pattern, **kwargs):
    raise NotImplementedError("einops stub hit at runtime")


def reduce(tensor, pattern, reduction, **kwargs):
    raise NotImplementedError("einops stub hit at runtime")


def repeat(tensor, pattern, **kwargs):
    raise NotImplementedError("einops stub hit at runtime")
