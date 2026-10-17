"""Structured input generators.

gen_candidate() builds vectors designed to make different members of the model
family disagree: widely spread product exponents (alignment width), exact
cancellation (accumulation order and grouping), terms straddling group
boundaries, and accumulators far above or below the products.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from .formats import Fmt, FIN, decode
from .device import Vec


def max_normal_exp(fmt: Fmt) -> int:
    top = fmt.emax_field if fmt.nan_style == "fn" else fmt.emax_field - 1
    return top - fmt.bias


def min_sub_exp(fmt: Fmt) -> int:
    """Leading-bit exponent of the smallest subnormal."""
    return fmt.emin - fmt.mbits + 1


def code_with_exp(fmt: Fmt, e: int, rng: random.Random, sign: int = None, pattern: str = None) -> int:
    """A finite nonzero code whose leading bit has unbiased exponent e (clipped)."""
    e = max(min_sub_exp(fmt), min(max_normal_exp(fmt), e))
    s = rng.getrandbits(1) if sign is None else sign
    if pattern is None:
        pattern = rng.choice(("rand", "rand", "ones", "pow2", "half"))
    mb = fmt.mbits
    if e < fmt.emin:  # subnormal
        k = e - (fmt.emin - mb)  # f has leading bit at position k
        low = (1 << k)
        if pattern == "pow2":
            f = low
        elif pattern == "ones":
            f = (1 << (k + 1)) - 1
        else:
            f = low | (rng.getrandbits(k) if k > 0 else 0)
        return (s << (fmt.ebits + mb)) | f
    if pattern == "pow2":
        frac = 0
    elif pattern == "ones":
        frac = (1 << mb) - 1
    elif pattern == "half":
        frac = 1 << (mb - 1)
    else:
        frac = rng.getrandbits(mb)
    efield = e + fmt.bias
    if fmt.nan_style == "fn" and efield == fmt.emax_field and frac == (1 << mb) - 1:
        frac -= 1
    return (s << (fmt.ebits + mb)) | (efield << mb) | frac


def pair_for_product_exp(fmt: Fmt, T: int, rng: random.Random, sa=None, sb=None):
    lo, hi = min_sub_exp(fmt), max_normal_exp(fmt)
    T = max(2 * lo, min(2 * hi, T))
    ea_lo, ea_hi = max(lo, T - hi), min(hi, T - lo)
    ea = rng.randint(ea_lo, ea_hi)
    eb = T - ea
    return code_with_exp(fmt, ea, rng, sa), code_with_exp(fmt, eb, rng, sb)


def fp32_with_exp(e: int, rng: random.Random, sign: int = None, pattern: str = None) -> int:
    e = max(-126, min(127, e))
    s = rng.getrandbits(1) if sign is None else sign
    pattern = pattern or rng.choice(("rand", "rand", "ones", "pow2"))
    frac = {"rand": rng.getrandbits(23), "ones": (1 << 23) - 1, "pow2": 0}[pattern]
    return (s << 31) | ((e + 127) << 23) | frac


def gen_candidate(fmt: Fmt, K: int, rng: random.Random, fspan: int = None) -> Vec:
    if fspan is None:
        fspan = rng.choice((12, 28, 48, 64, 90))
    lo, hi = min_sub_exp(fmt), max_normal_exp(fmt)
    a = [0] * K
    b = [0] * K
    mode = rng.choice(("spread", "spread", "cancel", "c_vs_prod", "dense", "boundary", "ladder", "ladder",
                       "dense_wide", "cancel_dense"))
    T = rng.randint(2 * lo, 2 * hi)
    c = 0

    def put(pos, T_term, sa=None):
        a[pos], b[pos] = pair_for_product_exp(fmt, T_term, rng, sa)

    if mode == "spread":
        n = rng.choice((2, 2, 3, 4, 6, min(K, 12)))
        n = min(n, K)
        for pos in rng.sample(range(K), n):
            put(pos, T - rng.randint(0, fspan))
        if rng.random() < 0.6:
            c = fp32_with_exp(T + rng.randint(-fspan, 3), rng)
    elif mode == "cancel":
        p, q = rng.sample(range(K), 2) if K > 1 else (0, 0)
        put(p, T)
        a[q], b[q] = a[p] ^ (1 << (fmt.width - 1)), b[p]  # exact negation of product p
        for pos in rng.sample(range(K), min(K, rng.choice((1, 2, 3)))):
            if a[pos] == 0:
                put(pos, T - rng.randint(1, fspan))
        if rng.random() < 0.4:
            c = fp32_with_exp(T - rng.randint(0, fspan), rng)
    elif mode == "c_vs_prod":
        ce = T + rng.randint(-6, 60)
        c = fp32_with_exp(ce, rng)
        for pos in rng.sample(range(K), min(K, rng.choice((1, 2, 4)))):
            put(pos, ce - rng.randint(-4, fspan))
    elif mode == "dense":
        w = rng.randint(0, 8)
        for pos in range(K):
            put(pos, T - rng.randint(0, w))
        if rng.random() < 0.7:
            c = fp32_with_exp(T + rng.randint(-w - 4, 6), rng)
    elif mode == "ladder":  # one big term, one or two small ones at a swept offset
        put(rng.randrange(K), T)
        for _ in range(rng.choice((1, 1, 2))):
            pos = rng.randrange(K)
            if a[pos] == 0:
                put(pos, T - rng.randint(0, 90))
        if rng.random() < 0.3:
            c = fp32_with_exp(T - rng.randint(0, 90), rng)
    elif mode == "dense_wide":
        w = rng.choice((10, 20, 30, 50, 70))
        for pos in range(K):
            put(pos, T - rng.randint(0, w))
        c = fp32_with_exp(T + rng.randint(-w, 6), rng) if rng.random() < 0.7 else 0
    elif mode == "cancel_dense":
        w = rng.choice((0, 2, 6, 12))
        order = list(range(K))
        rng.shuffle(order)
        half = K // 2
        for i in range(half):
            p, q = order[i], order[half + i]
            put(p, T - rng.randint(0, w))
            a[q], b[q] = a[p] ^ (1 << (fmt.width - 1)), b[p]
        for pos in rng.sample(range(K), rng.choice((1, 2))):
            put(pos, T - rng.randint(0, 90))
        if rng.random() < 0.5:
            c = fp32_with_exp(T - rng.randint(0, 90), rng)
    else:  # boundary: terms in adjacent positions to expose group size
        start = rng.randint(0, max(0, K - 2))
        for pos in range(start, min(K, start + rng.choice((2, 3, 5)))):
            put(pos, T - rng.randint(0, fspan // 2))
        c = fp32_with_exp(T - rng.randint(0, fspan // 2), rng) if rng.random() < 0.5 else 0
    return tuple(a), tuple(b), c
