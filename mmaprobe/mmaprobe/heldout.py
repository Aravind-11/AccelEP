"""Pre-registered held-out verification.

IMPORTANT: this file is hashed into the pre-registration record, so editing it
after the record is written invalidates the held-out run. Vectors are a pure
function of (stratum, seed, index); nothing here depends on the fitted model.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

from .formats import FIN, INF, NAN, Fmt, canon_out, decode
from .probes import code_with_exp, fp32_with_exp, max_normal_exp, min_sub_exp, pair_for_product_exp

Vec = Tuple[Tuple[int, ...], Tuple[int, ...], int]

STRATA = ("uniform", "matched", "cancel", "boundary", "special")


def _finite_code(fmt: Fmt, rng: random.Random) -> int:
    while True:
        x = rng.getrandbits(fmt.width)
        if decode(fmt, x)[0] == FIN:
            return x


def _finite_fp32(rng: random.Random) -> int:
    while True:
        x = rng.getrandbits(32)
        if (x & 0x7F800000) != 0x7F800000:
            return x


def _uniform(fmt, K, rng):
    return (tuple(_finite_code(fmt, rng) for _ in range(K)),
            tuple(_finite_code(fmt, rng) for _ in range(K)), _finite_fp32(rng))


def _matched(fmt, K, rng):
    lo, hi = min_sub_exp(fmt), max_normal_exp(fmt)
    T = rng.randint(lo + hi // 2, 2 * hi - 2)
    w = rng.choice((0, 1, 2, 4, 8, 16, 24))
    a, b = [], []
    for _ in range(K):
        x, y = pair_for_product_exp(fmt, T - rng.randint(0, w), rng)
        a.append(x)
        b.append(y)
    c = fp32_with_exp(T + rng.randint(-w - 6, 8), rng) if rng.random() < 0.85 else 0
    return tuple(a), tuple(b), c


def _cancel(fmt, K, rng):
    a, b, c = _matched(fmt, K, rng)
    a, b = list(a), list(b)
    half = K // 2
    order = list(range(K))
    rng.shuffle(order)
    for i in range(half):  # make pairs of exactly opposite products
        p, q = order[i], order[half + i]
        a[q], b[q] = a[p] ^ (1 << (fmt.width - 1)), b[p]
    for pos in rng.sample(range(K), rng.choice((0, 1, 2))):  # perturb so it is only nearly zero
        x, y = pair_for_product_exp(fmt, rng.randint(2 * min_sub_exp(fmt), 2 * max_normal_exp(fmt)), rng)
        a[pos], b[pos] = x, y
    return tuple(a), tuple(b), c


def _boundary(fmt, K, rng):
    ext_a = [0, 1 << (fmt.width - 1)]
    for e in (min_sub_exp(fmt), fmt.emin, 0, max_normal_exp(fmt)):
        ext_a += [code_with_exp(fmt, e, rng, 0, "pow2"), code_with_exp(fmt, e, rng, 1, "ones")]
    a = tuple(rng.choice(ext_a) if rng.random() < 0.5 else _finite_code(fmt, rng) for _ in range(K))
    b = tuple(rng.choice(ext_a) if rng.random() < 0.5 else _finite_code(fmt, rng) for _ in range(K))
    c_choices = [0, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x80000001, 0x7F7FFFFF,
                 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x00800001]
    r = rng.random()
    if r < 0.35:   # accumulator pass-through: all products zero, C at a boundary value
        a = tuple(0 for _ in range(K))
        return a, b, rng.choice(c_choices)
    if r < 0.5:    # overflow push: C near FP32 max, product of the same sign
        c = rng.choice((0x7F7FFFFF, 0xFF7FFFFF, 0x7F000000, 0xFF000000))
        big = pair_for_product_exp(fmt, 2 * max_normal_exp(fmt), rng, c >> 31, 0)
        a2, b2 = [0] * K, [0] * K
        a2[0], b2[0] = big
        return tuple(a2), tuple(b2), c
    return a, b, rng.choice(c_choices)


def _special(fmt, K, rng):
    a, b, c = _matched(fmt, K, rng)
    a, b = list(a), list(b)
    emax = fmt.emax_field
    nan_code = ((emax << fmt.mbits) | ((1 << fmt.mbits) - 1)) if fmt.nan_style == "fn" else \
        ((emax << fmt.mbits) | 1)
    inf_code = None if fmt.nan_style == "fn" else (emax << fmt.mbits)
    for _ in range(rng.choice((1, 1, 2))):
        pos = rng.randrange(K)
        kind = rng.choice(("nan", "inf", "inf", "c_special"))
        if kind == "nan" or (kind == "inf" and inf_code is None):
            (a if rng.random() < 0.5 else b)[pos] = nan_code
        elif kind == "inf":
            (a if rng.random() < 0.5 else b)[pos] = inf_code | (rng.getrandbits(1) << (fmt.width - 1))
        else:
            c = rng.choice((0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001))
    return tuple(a), tuple(b), c


_GENS: Dict[str, Callable] = {"uniform": _uniform, "matched": _matched, "cancel": _cancel,
                              "boundary": _boundary, "special": _special}


def generate(stratum: str, fmt: Fmt, K: int, seed: int, n: int) -> List[Vec]:
    """Deterministic: same (stratum, fmt, K, seed, n) always gives the same vectors."""
    rng = random.Random(f"{seed}/{stratum}/{fmt.name}/{K}")
    return [_GENS[stratum](fmt, K, rng) for _ in range(n)]


# ---- statistics ---------------------------------------------------------------------

def upper_bound(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper confidence bound on the mismatch probability.

    Exact for k == 0 (1 - (1-conf)**(1/n), about 3/n). Wilson score otherwise.
    """
    if n == 0:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    z = {0.95: 1.6449, 0.99: 2.3263}[conf]
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + rad) / denom)


# ---- shrinking ----------------------------------------------------------------------

def shrink(fmt: Fmt, K: int, vec: Vec, fails: Callable[[Vec], bool]) -> Vec:
    """Greedy delta-debugging of a failing vector to a smaller one that still fails."""
    a, b, c = list(vec[0]), list(vec[1]), vec[2]
    changed = True
    while changed:
        changed = False
        for i in range(K):
            if a[i] == 0 and b[i] == 0:
                continue
            sa, sb = a[i], b[i]
            a[i], b[i] = 0, 0
            if fails((tuple(a), tuple(b), c)):
                changed = True
            else:
                a[i], b[i] = sa, sb
        if c != 0 and fails((tuple(a), tuple(b), 0)):
            c, changed = 0, True
        frac_mask = (1 << fmt.mbits) - 1
        for i in range(K):
            for arr in (a, b):
                v = arr[i]
                if v and (v & frac_mask):
                    arr[i] = v & ~frac_mask
                    if decode(fmt, arr[i])[0] != FIN or decode(fmt, arr[i])[2] == 0 or \
                            not fails((tuple(a), tuple(b), c)):
                        arr[i] = v
                    else:
                        changed = True
        if c & 0x007FFFFF and fails((tuple(a), tuple(b), c & ~0x007FFFFF)):
            c, changed = c & ~0x007FFFFF, True
    return tuple(a), tuple(b), c
