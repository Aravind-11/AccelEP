"""End-to-end protocol: pre-register -> identify -> held-out verify -> report.

The ledger is a hash-chained JSON-lines file. It records, in order:
  prereg      sha256 of heldout.py, seed, per-stratum N, config
  identified  the frozen set of surviving model keys and their hash
  heldout     per-class mismatch counts, bounds, minimal counterexamples
The held-out step refuses to run unless the chain is intact, heldout.py still
matches the pre-registered hash, and 'identified' was written after 'prereg'.

Limit: a local ledger is self-attested. For the claim to mean anything to an
outsider, commit the prereg line to a public repo (or otherwise timestamp it)
BEFORE the device is first queried.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import heldout as H
from .device import CONFIGS, Config, Device
from .formats import canon_out, is_zero_sign_diff
from .identify import IdentifyResult, identify
from .model import Params, Prepared, evaluate


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


class Ledger:
    def __init__(self, path: str):
        self.path = path

    def entries(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, event: str, payload: dict) -> dict:
        es = self.entries()
        prev = es[-1]["hash"] if es else "0" * 64
        body = {"event": event, "time": time.time(), "prev": prev, "payload": payload}
        body["hash"] = sha256_bytes(json.dumps(body, sort_keys=True).encode())
        with open(self.path, "a") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries():
            h = e["hash"]
            body = {k: v for k, v in e.items() if k != "hash"}
            if body["prev"] != prev or sha256_bytes(json.dumps(body, sort_keys=True).encode()) != h:
                return False
            prev = h
        return True

    def find(self, event: str) -> Optional[dict]:
        for e in reversed(self.entries()):
            if e["event"] == event:
                return e
        return None


def heldout_source_path() -> str:
    return H.__file__.replace(".pyc", ".py")


def preregister(ledger: Ledger, config: Config, seed: int, n_per_stratum: int) -> dict:
    return ledger.append("prereg", {
        "config": config.name, "seed": seed, "n_per_stratum": n_per_stratum,
        "strata": list(H.STRATA), "heldout_py_sha256": sha256_file(heldout_source_path()),
    })


def record_identified(ledger: Ledger, res: IdentifyResult) -> dict:
    keys = sorted(m.key() for m in res.survivors)
    return ledger.append("identified", {
        "survivors": keys, "survivors_sha256": sha256_bytes("\n".join(keys).encode()),
        "empty": res.empty, "rounds": res.rounds, "queries": res.queries,
        "nondeterministic": res.nondeterministic,
    })


def _parse_key(key: str) -> Params:
    g, f, al, ou, cm = key.split("_", 4)
    return Params(int(g[1:]), int(f[1:]), al, ou, cm)


@dataclass
class ClassReport:
    representative: str
    members: int
    per_stratum: Dict[str, dict]
    total_mismatches: int
    total_n: int
    zero_sign_only: int
    upper95: float


def run_heldout(ledger: Ledger, device: Device, config: Config,
                classes: Sequence[Sequence[Params]], max_counterexamples: int = 3) -> dict:
    pre = ledger.find("prereg")
    ident = ledger.find("identified")
    if pre is None or ident is None:
        raise RuntimeError("ledger needs a prereg entry and an identified entry")
    if not ledger.verify_chain():
        raise RuntimeError("ledger hash chain is broken")
    if pre["time"] > ident["time"]:
        raise RuntimeError("prereg was recorded after identification")
    if pre["payload"]["heldout_py_sha256"] != sha256_file(heldout_source_path()):
        raise RuntimeError("heldout.py changed after pre-registration")
    seed, n = pre["payload"]["seed"], pre["payload"]["n_per_stratum"]
    fmt, K = config.fmt, config.K

    vec_by_stratum = {s: H.generate(s, fmt, K, seed, n) for s in H.STRATA}
    dev_by_stratum = {s: device.query(vec_by_stratum[s]) for s in H.STRATA}

    reports: List[ClassReport] = []
    counterexamples: List[dict] = []
    for cls in classes:
        rep = cls[0]
        per: Dict[str, dict] = {}
        tot_mis = tot_zs = 0
        for s in H.STRATA:
            mis = zs = 0
            first_bad = []
            for v, d in zip(vec_by_stratum[s], dev_by_stratum[s]):
                m = evaluate(Prepared(fmt, *v), rep)
                if canon_out(m) != canon_out(d):
                    mis += 1
                    if len(first_bad) < max_counterexamples:
                        first_bad.append((v, d, m))
                elif m != d:
                    zs += 1
            per[s] = {"n": len(vec_by_stratum[s]), "mismatches": mis, "zero_sign_only": zs,
                      "upper95": H.upper_bound(mis, len(vec_by_stratum[s]))}
            tot_mis += mis
            tot_zs += zs
            for v, d, m in first_bad:
                def fails(x, rep=rep):
                    return canon_out(evaluate(Prepared(fmt, *x), rep)) != canon_out(device.query([x])[0])
                small = H.shrink(fmt, K, v, fails)
                counterexamples.append({
                    "class": rep.key(), "stratum": s, "minimal_vec": small,
                    "device_bits": device.query([small])[0],
                    "model_bits": evaluate(Prepared(fmt, *small), rep)})
        total_n = sum(p["n"] for p in per.values())
        reports.append(ClassReport(rep.key(), len(cls), per, tot_mis, total_n, tot_zs,
                                   H.upper_bound(tot_mis, total_n)))
    out = {"reports": [r.__dict__ for r in reports], "counterexamples": counterexamples}
    ledger.append("heldout", {"summary": [(r.representative, r.total_mismatches, r.total_n)
                                          for r in reports]})
    return out
