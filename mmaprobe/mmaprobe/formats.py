"""Number formats and exact (integer) encode/decode.

Everything here is exact integer arithmetic. A finite value is represented as
(sign, mant, exp2) meaning (-1)**sign * mant * 2**exp2 with mant a Python int.
"""
from __future__ import annotations

from dataclasses import dataclass

FIN, INF, NAN = 0, 1, 2


@dataclass(frozen=True)
class Fmt:
    name: str
    ebits: int
    mbits: int
    bias: int
    nan_style: str  # "ieee": all-ones exponent is inf/nan. "fn": no inf, one NaN code.

    @property
    def width(self) -> int:
        return 1 + self.ebits + self.mbits

    @property
    def emax_field(self) -> int:
        return (1 << self.ebits) - 1

    @property
    def emin(self) -> int:
        """Unbiased exponent of the smallest normal number."""
        return 1 - self.bias


E4M3 = Fmt("e4m3", 4, 3, 7, "fn")
E5M2 = Fmt("e5m2", 5, 2, 15, "ieee")
FP16 = Fmt("fp16", 5, 10, 15, "ieee")
BF16 = Fmt("bf16", 8, 7, 127, "ieee")
FP32 = Fmt("fp32", 8, 23, 127, "ieee")

BY_NAME = {f.name: f for f in (E4M3, E5M2, FP16, BF16, FP32)}

FP32_QNAN = 0x7FC00000
FP32_INF = 0x7F800000


def decode(fmt: Fmt, bits: int):
    """Return (FIN, sign, mant, exp2) | (INF, sign, 0, 0) | (NAN, 0, 0, 0)."""
    s = (bits >> (fmt.ebits + fmt.mbits)) & 1
    e = (bits >> fmt.mbits) & fmt.emax_field
    f = bits & ((1 << fmt.mbits) - 1)
    if fmt.nan_style == "ieee":
        if e == fmt.emax_field:
            return (NAN, 0, 0, 0) if f else (INF, s, 0, 0)
    else:
        if e == fmt.emax_field and f == (1 << fmt.mbits) - 1:
            return (NAN, 0, 0, 0)
    if e == 0:
        return (FIN, s, f, fmt.emin - fmt.mbits)
    return (FIN, s, (1 << fmt.mbits) | f, e - fmt.bias - fmt.mbits)


def round_to_fp32(sign: int, mag: int, exp2: int, mode: str) -> int:
    """Round (-1)**sign * mag * 2**exp2 to FP32 bits. mode: 'rz' or 'rne'.

    mag == 0 gives +0. Overflow gives inf (rne) or max finite (rz).
    """
    fmt = FP32
    if mag == 0:
        return 0
    top = mag.bit_length() - 1 + exp2  # exponent of the leading bit
    quantum = (top if top >= fmt.emin else fmt.emin) - fmt.mbits  # exponent of the LSB
    shift = quantum - exp2
    if shift <= 0:
        q = mag << (-shift)
    else:
        q = mag >> shift
        if mode == "rne":
            rem = mag & ((1 << shift) - 1)
            half = 1 << (shift - 1)
            if rem > half or (rem == half and (q & 1)):
                q += 1
        elif mode != "rz":
            raise ValueError(mode)
    if q == 0:
        return sign << 31
    if q >= (1 << (fmt.mbits + 1)):  # rounding carried out of the mantissa
        q >>= 1
        quantum += 1
    if q < (1 << fmt.mbits):
        efield, frac = 0, q
    else:
        efield = quantum + fmt.mbits + fmt.bias
        frac = q & ((1 << fmt.mbits) - 1)
        if q == (1 << fmt.mbits) and quantum == fmt.emin - fmt.mbits:
            efield = 1
    if efield >= fmt.emax_field:
        if mode == "rne":
            return (sign << 31) | FP32_INF
        return (sign << 31) | 0x7F7FFFFF
    return (sign << 31) | (efield << fmt.mbits) | frac


def canon_out(bits: int) -> int:
    """Canonical form for comparing FP32 outputs: all NaNs equal, -0 == +0."""
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF):
        return FP32_QNAN
    if bits == 0x80000000:
        return 0
    return bits


def is_zero_sign_diff(x: int, y: int) -> bool:
    return x != y and canon_out(x) == canon_out(y) and (x | y) & 0x7FFFFFFF == 0


def fp32_to_float(bits: int) -> float:
    import struct
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
