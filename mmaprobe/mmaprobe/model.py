"""Parameterized model of a tensor-core dot product D = sum_k a_k*b_k + C.

Family (all exact integer arithmetic, no host floating point):

  * Products a_k*b_k are exact.
  * Products are consumed in groups of G consecutive k. Within a group, the
    group's products plus one accumulator term are added in a single fused
    step: every term is aligned to a fixed-point grid whose LSB sits F bits
    below the FP32 ulp of the largest term (grid LSB = lmax - 23 - F, so F=0
    keeps exactly 24 significant bits of the largest term and F>0 keeps F
    guard bits), bits shifted off are dropped
    by truncation (align='trunc') or round-to-nearest-even (align='rne'), the
    integers are summed exactly, and the sum is rounded to FP32 (out='rz' or
    out='rne').
  * cmode='chain'  : the accumulator starts as C and is carried through every
                     group.
    cmode='c_last' : the accumulator starts at 0 and C is added in one extra
                     two-term step after the last group.

This is a hypothesis family, not a claim about any chip. Identification
(identify.py) keeps only members that match hardware bit for bit, and the
held-out test (heldout.py) checks whether the survivors generalize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .formats import FIN, INF, NAN, FP32, FP32_INF, FP32_QNAN, Fmt, decode, round_to_fp32

Term = Tuple[int, int, int]  # (sign, mant>0, exp2)


@dataclass(frozen=True)
class Params:
    G: int
    F: int
    align: str  # 'trunc' | 'rne'
    out: str    # 'rz' | 'rne'
    cmode: str  # 'chain' | 'c_last'

    def key(self) -> str:
        return f"G{self.G}_F{self.F}_{self.align}_{self.out}_{self.cmode}"


F_MAX = 31
F_MIN = -8


def model_space(K: int, f_max: int = F_MAX, f_min: int = F_MIN) -> List[Params]:
    gs = [g for g in (1, 2, 4, 8, 16, 32) if g <= K and K % g == 0]
    return [
        Params(g, f, al, ou, cm)
        for g in gs
        for f in range(f_min, f_max + 1)
        for al in ("trunc", "rne")
        for ou in ("rz", "rne")
        for cm in ("chain", "c_last")
    ]


class Prepared:
    """A query vector with operands decoded once, shared across all models."""

    __slots__ = ("special", "prods", "cterm", "all_zero_neg")

    def __init__(self, fmt: Fmt, a_bits: Sequence[int], b_bits: Sequence[int], c_bits: int):
        self.special: Optional[int] = None
        self.prods: List[Optional[Term]] = []
        self.cterm: Optional[Term] = None
        pos_inf = neg_inf = False
        nan = False
        for ab, bb in zip(a_bits, b_bits):
            ka, sa, ma, ea = decode(fmt, ab)
            kb, sb, mb, eb = decode(fmt, bb)
            if ka == NAN or kb == NAN:
                nan = True
                self.prods.append(None)
                continue
            if ka == INF or kb == INF:
                other_zero = (ka == FIN and ma == 0) or (kb == FIN and mb == 0)
                if other_zero:
                    nan = True
                elif sa ^ sb:
                    neg_inf = True
                else:
                    pos_inf = True
                self.prods.append(None)
                continue
            if ma == 0 or mb == 0:
                self.prods.append(None)
            else:
                self.prods.append((sa ^ sb, ma * mb, ea + eb))
        kc, sc, mc, ec = decode(FP32, c_bits)
        if kc == NAN:
            nan = True
        elif kc == INF:
            if sc:
                neg_inf = True
            else:
                pos_inf = True
        elif mc:
            self.cterm = (sc, mc, ec)
        if nan or (pos_inf and neg_inf):
            self.special = FP32_QNAN
        elif pos_inf:
            self.special = FP32_INF
        elif neg_inf:
            self.special = FP32_INF | 0x80000000
        self.all_zero_neg = False


def _step(terms: List[Term], F: int, align: str, out: str) -> int:
    """One fused multi-term add. Returns FP32 bits."""
    if not terms:
        return 0
    lmax = max(e + m.bit_length() - 1 for _, m, e in terms)
    g = lmax - 23 - F
    total = 0
    for s, m, e in terms:
        d = e - g
        if d >= 0:
            v = m << d
        else:
            sh = -d
            v = m >> sh
            if align == "rne":
                rem = m & ((1 << sh) - 1)
                half = 1 << (sh - 1)
                if rem > half or (rem == half and (v & 1)):
                    v += 1
        total += -v if s else v
    if total == 0:
        return 0
    return round_to_fp32(1 if total < 0 else 0, abs(total), g, out)


def _acc_term(bits: int) -> Optional[Term]:
    k, s, m, e = decode(FP32, bits)
    if k != FIN or m == 0:
        return None
    return (s, m, e)


def evaluate(p: Prepared, prm: Params) -> int:
    """FP32 result bits of the model on a prepared vector."""
    if p.special is not None:
        return p.special
    prods = p.prods
    K = len(prods)
    G = prm.G
    if prm.cmode == "chain":
        acc = p.cterm
        bits = 0
        for g0 in range(0, K, G):
            terms = [t for t in prods[g0:g0 + G] if t is not None]
            if acc is not None:
                terms.append(acc)
            bits = _step(terms, prm.F, prm.align, prm.out)
            if (bits & 0x7F800000) == 0x7F800000:
                return bits  # overflowed to inf; stays inf
            acc = _acc_term(bits)
        return bits
    acc_bits = 0
    for g0 in range(0, K, G):
        terms = [t for t in prods[g0:g0 + G] if t is not None]
        a = _acc_term(acc_bits)
        if a is not None:
            terms.append(a)
        acc_bits = _step(terms, prm.F, prm.align, prm.out)
        if (acc_bits & 0x7F800000) == 0x7F800000:
            return acc_bits
    terms = [t for t in (_acc_term(acc_bits), p.cterm) if t is not None]
    return _step(terms, prm.F, prm.align, prm.out)


def model_dot(fmt: Fmt, prm: Params, a_bits, b_bits, c_bits) -> int:
    return evaluate(Prepared(fmt, a_bits, b_bits, c_bits), prm)
