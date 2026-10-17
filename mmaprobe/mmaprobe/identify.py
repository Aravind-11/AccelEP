"""Active model identification.

Instead of running a fixed probe battery and reading parameters off by hand,
keep every member of the model family alive and repeatedly pick the input that
splits the survivors best (minimum collision probability), query the device on
it, and discard every model whose prediction differs from the hardware bit for
bit. Termination is one of:

  * one behavioral class survives (identified),
  * several survive and no generated input can split them (equivalent under the
    generator; the held-out test adjudicates),
  * none survive (the hardware is outside the family; the last query is the
    falsifying witness).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .device import Device, Vec
from .formats import Fmt, canon_out
from .model import Params, Prepared, evaluate, model_space
from .probes import gen_candidate


@dataclass
class IdentifyResult:
    survivors: List[Params]
    classes: List[List[Params]]
    rounds: int
    queries: int
    empty: bool
    nondeterministic: bool
    witness: Optional[Tuple[Vec, int]]  # (vector, device bits) that eliminated the last model
    log: List[dict] = field(default_factory=list)
    seconds: float = 0.0


def behavior_signature(fmt: Fmt, prm: Params, prepared: Sequence[Prepared]) -> Tuple[int, ...]:
    return tuple(canon_out(evaluate(p, prm)) for p in prepared)


def validation_vectors(fmt: Fmt, K: int, rng: random.Random, n: int = 400) -> List[Vec]:
    return [gen_candidate(fmt, K, rng) for _ in range(n)]


def group_classes(fmt: Fmt, K: int, survivors: Sequence[Params], rng: random.Random,
                  n: int = 400) -> List[List[Params]]:
    prepared = [Prepared(fmt, *v) for v in validation_vectors(fmt, K, rng, n)]
    groups: Dict[Tuple[int, ...], List[Params]] = {}
    for m in survivors:
        groups.setdefault(behavior_signature(fmt, m, prepared), []).append(m)
    return list(groups.values())


def identify(device: Device, fmt: Fmt, K: int, rng: random.Random, *, pool: int = 48,
             sample: int = 300, max_rounds: int = 300, patience: int = 4,
             verbose: bool = False, space: Optional[List[Params]] = None,
             max_pool: int = 6000) -> IdentifyResult:
    """Greedy active elimination with class-aware search effort.

    Once <= `sample` models survive they are grouped into behavioral classes (identical on a
    fixed probe set). Scoring then evaluates class representatives, which is cheap enough to
    afford a pool that grows 3x each time the search stalls, up to `max_pool`; stalled rounds
    also sample several members per class so within-class differences can be found.
    """
    t0 = time.time()
    survivors = list(space) if space is not None else model_space(K)
    log: List[dict] = []
    quiet = 0
    rounds = 0
    nondet = False
    witness = None
    q_start = device.queries
    classes: Optional[List[List[Params]]] = None
    sig_prep = [Prepared(fmt, *gen_candidate(fmt, K, rng)) for _ in range(150)]

    def build_classes(models: Sequence[Params]) -> List[List[Params]]:
        groups: Dict[Tuple[int, ...], List[Params]] = {}
        for m in models:
            groups.setdefault(behavior_signature(fmt, m, sig_prep), []).append(m)
        return list(groups.values())

    while len(survivors) > 1 and rounds < max_rounds:
        if classes is None and len(survivors) <= sample:
            classes = build_classes(survivors)
        if classes is None:
            units = [(m, 1.0) for m in (survivors if len(survivors) <= sample
                                        else rng.sample(survivors, sample))]
            n_pool = pool
        else:
            units = []
            for c in classes:
                if quiet == 0 or len(c) == 1:
                    units.append((c[0], float(len(c))))
                else:
                    pick = c if len(c) <= 6 else rng.sample(c, 6)
                    units.extend((m, len(c) / len(pick)) for m in pick)
            n_pool = min(max_pool, 200 * (3 ** quiet))
        best = None
        for _ in range(n_pool):
            v = gen_candidate(fmt, K, rng)
            p = Prepared(fmt, *v)
            counts: Dict[int, float] = {}
            for m, w in units:
                o = canon_out(evaluate(p, m))
                counts[o] = counts.get(o, 0.0) + w
            score = sum(c * c for c in counts.values())
            if len(counts) > 1 and (best is None or score < best[0]):
                best = (score, v, p)
        if best is None:
            quiet += 1
            if quiet >= patience:
                break
            continue
        quiet = 0
        _, v, p = best
        o1, o2 = device.query([v, v])
        if canon_out(o1) != canon_out(o2):
            nondet = True
        before = len(survivors)
        target = canon_out(o1)
        if classes is not None:
            classes = [[m for m in c if canon_out(evaluate(p, m)) == target] for c in classes]
            classes = [c for c in classes if c]
            survivors = [m for c in classes for m in c]
        else:
            survivors = [m for m in survivors if canon_out(evaluate(p, m)) == target]
        rounds += 1
        log.append({"round": rounds, "before": before, "after": len(survivors),
                    "vec": v, "device_bits": o1})
        if verbose:
            print(f"  round {rounds:3d}: {before:5d} -> {len(survivors):5d}", flush=True)
        if not survivors:
            witness = (v, o1)
            break
    final = group_classes(fmt, K, survivors, rng) if survivors else []
    return IdentifyResult(survivors, final, rounds, device.queries - q_start,
                          not survivors, nondet, witness, log, time.time() - t0)


def baseline_random_fit(device: Device, fmt: Fmt, K: int, rng: random.Random, n_queries: int,
                        space: Optional[List[Params]] = None) -> List[Params]:
    """What a typical emulator author does: fit on unit-scale random inputs.

    Returns every model tied for best agreement (the caller scores the tie set).
    """
    from .probes import code_with_exp, fp32_with_exp
    vecs: List[Vec] = []
    for _ in range(n_queries):
        a = tuple(code_with_exp(fmt, rng.randint(-3, 1), rng) for _ in range(K))
        b = tuple(code_with_exp(fmt, rng.randint(-3, 1), rng) for _ in range(K))
        vecs.append((a, b, fp32_with_exp(rng.randint(-3, 3), rng)))
    outs = device.query(vecs)
    prepared = [Prepared(fmt, *v) for v in vecs]
    best, tied = -1, []
    for m in (space if space is not None else model_space(K)):
        agree = sum(canon_out(evaluate(p, m)) == canon_out(o) for p, o in zip(prepared, outs))
        if agree > best:
            best, tied = agree, [m]
        elif agree == best:
            tied.append(m)
    return tied
