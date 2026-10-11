"""Test stub: msgpack names DeepSpeed imports; unused on the CPU test path (loud if hit)."""


def packb(obj):
    raise NotImplementedError("msgpack stub hit at runtime")


def unpackb(buf):
    raise NotImplementedError("msgpack stub hit at runtime")


class Packer:
    def __init__(self, *a, **k):
        raise NotImplementedError("msgpack stub hit at runtime")


class Unpacker:
    def __init__(self, *a, **k):
        raise NotImplementedError("msgpack stub hit at runtime")
