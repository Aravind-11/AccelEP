"""Read a safetensors file into float32 arrays. No torch."""

import json
import struct

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = blob[start:end]
        shape = tuple(info["shape"])
        dtype = info["dtype"]
        if dtype == "F32":
            values = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif dtype == "F16":
            values = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        elif dtype == "BF16":
            bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            values = bits.view(np.float32).reshape(shape)
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        tensors[name] = np.ascontiguousarray(values, dtype=np.float32)
    return tensors
