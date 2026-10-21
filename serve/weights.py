"""Read a safetensors file into float32 arrays. No torch.

``Llama`` uploads one tensor and drops it before the next read. ``load_safetensors``
still materializes the whole file for callers that want the dict.
"""

import json
import struct

import numpy as np


class SafetensorsReader:
    """Header-only reader. ``get`` materializes one array and leaves the file unchanged."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            self._header = json.loads(handle.read(header_len))
            self._data = 8 + header_len

    def contains(self, name):
        return name in self._header and name != "__metadata__"

    def get(self, name):
        info = self._header[name]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as handle:
            handle.seek(self._data + start)
            raw = handle.read(end - start)
        return _decode(raw, tuple(info["shape"]), info["dtype"], name)


def _decode(raw, shape, dtype, name):
    if dtype == "F32":
        values = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    elif dtype == "BF16":
        # Same shift the loader has always used: bf16 bits moved into the float32 exponent.
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        values = bits.view(np.float32).reshape(shape)
    else:
        raise ValueError(f"unsupported dtype {dtype} for {name}")
    return np.ascontiguousarray(values, dtype=np.float32)


def load_safetensors(path):
    reader = SafetensorsReader(path)
    tensors = {}
    for name in reader._header:
        if name == "__metadata__":
            continue
        tensors[name] = reader.get(name)
    return tensors
