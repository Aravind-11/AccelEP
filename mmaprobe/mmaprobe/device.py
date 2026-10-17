"""Devices under test. A device maps (a[K], b[K], c) -> FP32 result bits.

SyntheticDevice has hidden arithmetic (a model instance, optionally with an
out-of-family quirk). It exists so the identification and verification
machinery can be validated without hardware. CudaDevice drives probe_mma.cu on
a real GPU.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .formats import FP32, Fmt, decode, FIN
from .model import Params, Prepared, evaluate

Vec = Tuple[Tuple[int, ...], Tuple[int, ...], int]


@dataclass(frozen=True)
class Config:
    name: str
    fmt: Fmt
    K: int
    arch: str  # nvcc -arch value
    mma: str   # PTX shape/type summary, informational


class Device:
    queries = 0

    def query(self, vecs: Sequence[Vec]) -> List[int]:
        raise NotImplementedError


def _quirk_ftz_output(bits: int) -> int:
    """Flush FP32 subnormal results to zero (an out-of-family behavior)."""
    if (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF):
        return bits & 0x80000000
    return bits


QUIRKS = {
    "ftz_output": _quirk_ftz_output,
}


class SyntheticDevice(Device):
    def __init__(self, fmt: Fmt, params: Params, quirk: Optional[str] = None,
                 product_flush_exp: Optional[int] = None):
        self.fmt = fmt
        self.params = params
        self.quirk = quirk
        self.product_flush_exp = product_flush_exp  # flush products below 2**this
        self.queries = 0

    def _one(self, v: Vec) -> int:
        p = Prepared(self.fmt, *v)
        if self.product_flush_exp is not None:
            lim = self.product_flush_exp
            p.prods = [None if (t is not None and t[2] + t[1].bit_length() - 1 < lim) else t
                       for t in p.prods]
        out = evaluate(p, self.params)
        if self.quirk:
            out = QUIRKS[self.quirk](out)
        return out

    def query(self, vecs: Sequence[Vec]) -> List[int]:
        self.queries += len(vecs)
        return [self._one(v) for v in vecs]


class CudaDevice(Device):
    """Runs probe_mma on a GPU. Wire format (little endian):

    in.bin : u32 n, then n records of [a: K elems][b: K elems][c: u32]
             (elements are 1 byte for fp8, 2 bytes for fp16/bf16)
    out.bin: n x u32 FP32 result bits, element (0,0) of D.
    """

    def __init__(self, binary: str, config: Config):
        self.binary = binary
        self.config = config
        self.queries = 0
        self.esize = 1 if config.fmt.width <= 8 else 2

    def _pack(self, vecs: Sequence[Vec]) -> bytes:
        out = bytearray(struct.pack("<I", len(vecs)))
        code = "B" if self.esize == 1 else "H"
        for a, b, c in vecs:
            out += struct.pack("<%d%s" % (len(a), code), *a)
            out += struct.pack("<%d%s" % (len(b), code), *b)
            out += struct.pack("<I", c)
        return bytes(out)

    def query(self, vecs: Sequence[Vec]) -> List[int]:
        if not vecs:
            return []
        self.queries += len(vecs)
        with tempfile.TemporaryDirectory() as d:
            fin, fout = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            with open(fin, "wb") as f:
                f.write(self._pack(vecs))
            subprocess.run([self.binary, self.config.name, fin, fout], check=True)
            raw = open(fout, "rb").read()
        n = len(vecs)
        if len(raw) != 4 * n:
            raise RuntimeError(f"probe returned {len(raw)} bytes, expected {4 * n}")
        return list(struct.unpack("<%dI" % n, raw))


CONFIGS = {
    "e4m3_k32": Config("e4m3_k32", None, 32, "sm_89", "mma.m16n8k32 e4m3 -> f32"),
    "f16_k8": Config("f16_k8", None, 8, "sm_75", "mma.m16n8k8 f16 -> f32"),
}


def _init_configs():
    from .formats import E4M3, FP16
    CONFIGS["e4m3_k32"] = Config("e4m3_k32", E4M3, 32, "sm_89", "mma.m16n8k32 e4m3 -> f32")
    CONFIGS["f16_k8"] = Config("f16_k8", FP16, 8, "sm_75", "mma.m16n8k8 f16 -> f32")


_init_configs()
