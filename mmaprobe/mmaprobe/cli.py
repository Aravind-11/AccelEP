"""Command line: `python -m mmaprobe validate | run | selftest-layout`."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import List, Optional

from . import heldout as H
from .device import CONFIGS, CudaDevice, Device, SyntheticDevice
from .formats import canon_out
from .identify import baseline_random_fit, behavior_signature, identify, validation_vectors
from .model import F_MAX, F_MIN, Params, Prepared, evaluate, model_space
from .protocol import Ledger, preregister, record_identified, run_heldout


def _parse_params(s: str) -> Params:
    g, f, al, ou, cm = s.split("_", 4)
    return Params(int(g[1:]), int(f[1:]), al, ou, cm)


def _signature_vectors(cfg, seed: int, n: int = 500):
    rng = random.Random(seed)
    vs = validation_vectors(cfg.fmt, cfg.K, rng, n)
    for s in H.STRATA:
        vs += H.generate(s, cfg.fmt, cfg.K, 987654, n // 5)
    return [Prepared(cfg.fmt, *v) for v in vs]


def _contains_truth(cfg, hidden: Params, survivors, sigvecs) -> bool:
    ts = behavior_signature(cfg.fmt, hidden, sigvecs)
    return any(behavior_signature(cfg.fmt, m, sigvecs) == ts for m in survivors)


def _random_params(cfg, rng: random.Random) -> Params:
    gs = [g for g in (1, 2, 4, 8, 16, 32) if g <= cfg.K and cfg.K % g == 0]
    return Params(rng.choice(gs), rng.randint(F_MIN, F_MAX), rng.choice(("trunc", "rne")),
                  rng.choice(("rz", "rne")), rng.choice(("chain", "c_last")))


def _units(res):
    """Held-out units: every survivor when there are few, else one representative per class."""
    if len(res.survivors) <= 60:
        return [[m] for m in res.survivors]
    return res.classes


def validate(args) -> int:
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    t_all = time.time()
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        rng = random.Random(args.seed)
        sigvecs = _signature_vectors(cfg, args.seed + 1)
        hidden_list = [("random", _random_params(cfg, rng), None, None) for _ in range(args.n_random)]
        hidden_list += [("single-fused", Params(cfg.K, 31, "trunc", "rz", "chain"), None, None),
                        ("sequential", Params(1, 8, "rne", "rne", "chain"), None, None)]
        # out-of-family devices (the model family cannot express these)
        p = _random_params(cfg, rng)
        hidden_list += [("quirk:ftz_output", p, "ftz_output", None),
                        ("quirk:product_flush", p, None, -10 if cfg.fmt.name == "fp16" else -14)]
        for kind, prm, quirk, pf in hidden_list:
            dev = SyntheticDevice(cfg.fmt, prm, quirk, pf)
            ledger_path = os.path.join(out_dir, f"ledger_{cfg_name}_{len(rows)}.jsonl")
            if os.path.exists(ledger_path):
                os.remove(ledger_path)
            led = Ledger(ledger_path)
            preregister(led, cfg, args.seed + len(rows), args.n_heldout)
            t0 = time.time()
            res = identify(dev, cfg.fmt, cfg.K, random.Random(args.seed + 100 + len(rows)))
            row = {"config": cfg_name, "kind": kind, "hidden": prm.key(), "rounds": res.rounds,
                   "queries": res.queries, "survivors": len(res.survivors),
                   "classes": len(res.classes), "falsified": res.empty,
                   "id_seconds": round(res.seconds, 2)}
            in_family = quirk is None and pf is None
            if res.empty:
                row.update(contains_truth=False, heldout_mismatches=None, heldout_n=None, upper95=None)
                row["outcome"] = "FLAGGED (no model fits)" if not in_family else "FAIL (truth eliminated)"
            else:
                record_identified(led, res)
                ho = run_heldout(led, dev, cfg, _units(res), max_counterexamples=1)
                rep = ho["reports"]
                best = min(rep, key=lambda r: r["total_mismatches"])
                verified = [r["representative"] for r in rep if r["total_mismatches"] == 0]
                row["verified_models"] = len(verified)
                if in_family:
                    row["truth_verified"] = prm.key() in verified or (
                        len(verified) > 0 and any(
                            behavior_signature(cfg.fmt, _parse_params(k), sigvecs)
                            == behavior_signature(cfg.fmt, prm, sigvecs) for k in verified))
                row.update(contains_truth=_contains_truth(cfg, prm, res.survivors, sigvecs) if in_family else None,
                           heldout_mismatches=best["total_mismatches"], heldout_n=best["total_n"],
                           upper95=best["upper95"])
                if in_family:
                    ok = row["contains_truth"] and row["truth_verified"] and \
                        len({behavior_signature(cfg.fmt, _parse_params(k), sigvecs) for k in verified}) == 1
                    row["outcome"] = "PASS" if ok else "FAIL"
                else:
                    row["outcome"] = "FLAGGED (held-out mismatches)" if best["total_mismatches"] else \
                        "MISSED (quirk not exercised)"
                    if ho["counterexamples"]:
                        row["example_counterexample"] = ho["counterexamples"][0]
            # baseline: random unit-scale fit with the same query budget (>= 16)
            nq = max(16, res.queries)
            tied = baseline_random_fit(SyntheticDevice(cfg.fmt, prm, quirk, pf), cfg.fmt, cfg.K,
                                       random.Random(args.seed + 200 + len(rows)), nq)
            row["baseline_tied_models"] = len(tied)
            if in_family:
                ts = behavior_signature(cfg.fmt, prm, sigvecs)
                good = sum(behavior_signature(cfg.fmt, m, sigvecs) == ts for m in tied)
                row["baseline_expected_correct"] = round(good / len(tied), 4)
            rows.append(row)
            print(f"[{cfg_name}] {kind:20s} {prm.key():28s} rounds={res.rounds:2d} "
                  f"classes={row['classes']} -> {row['outcome']}  "
                  f"(baseline tie set {len(tied)}, expected-correct {row.get('baseline_expected_correct')})",
                  flush=True)
    inf = [r for r in rows if not r["kind"].startswith("quirk")]
    quirks = [r for r in rows if r["kind"].startswith("quirk")]
    summary = {
        "in_family_pass": sum(r["outcome"] == "PASS" for r in inf), "in_family_total": len(inf),
        "quirks_flagged": sum(r["outcome"].startswith("FLAGGED") for r in quirks),
        "quirks_total": len(quirks),
        "median_queries": sorted(r["queries"] for r in inf)[len(inf) // 2] if inf else None,
        "baseline_mean_expected_correct": round(
            sum(r["baseline_expected_correct"] for r in inf) / len(inf), 4) if inf else None,
        "seconds": round(time.time() - t_all, 1),
    }
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    return 0 if summary["in_family_pass"] == summary["in_family_total"] else 1


def run(args) -> int:
    cfg = CONFIGS[args.config]
    os.makedirs(args.out, exist_ok=True)
    led = Ledger(os.path.join(args.out, "ledger.jsonl"))
    if args.device == "cuda":
        for step in ([args.binary, "--info"], [args.binary, cfg.name, "--selftest"]):
            r = subprocess.run(step, capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            if r.returncode != 0:
                print("probe selftest failed; fix the fragment layout before trusting any result",
                      file=sys.stderr)
                return 2
            if "--info" in step:
                led.append("device_info", {"info": r.stdout.strip()})
        device: Device = CudaDevice(args.binary, cfg)
    else:
        device = SyntheticDevice(cfg.fmt, _parse_params(args.params))
    if led.find("prereg") is None:
        preregister(led, cfg, args.seed, args.n_heldout)
        print("pre-registered; commit", os.path.join(args.out, "ledger.jsonl"), "now if you want "
              "an outside reader to trust the order of events")
    print("identifying ...", flush=True)
    res = identify(device, cfg.fmt, cfg.K, random.Random(args.seed), verbose=True)
    record_identified(led, res)
    report = {"config": cfg.name, "rounds": res.rounds, "queries": res.queries,
              "falsified": res.empty, "nondeterministic": res.nondeterministic,
              "survivors": [m.key() for m in res.survivors],
              "classes": [[m.key() for m in c] for c in res.classes]}
    if res.nondeterministic:
        print("WARNING: identical queries returned different results; the hardware (or the "
              "launch) is not deterministic for this instruction.", file=sys.stderr)
    if res.empty:
        v, bits = res.witness
        report["witness"] = {"a": v[0], "b": v[1], "c": v[2], "device_bits": bits}
        print("NO MODEL IN THE FAMILY FITS THE HARDWARE. Witness saved in report.json.")
    else:
        ho = run_heldout(led, device, cfg, _units(res))
        report["heldout"] = ho
        for r in ho["reports"]:
            print(f"class {r['representative']} (+{r['members'] - 1} equivalent): "
                  f"{r['total_mismatches']} mismatches / {r['total_n']}  upper95={r['upper95']:.2e}")
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="mmaprobe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate the method on synthetic devices with hidden arithmetic")
    v.add_argument("--out", default="results")
    v.add_argument("--configs", nargs="+", default=["f16_k8", "e4m3_k32"])
    v.add_argument("--n-random", type=int, default=8)
    v.add_argument("--n-heldout", type=int, default=1500)
    v.add_argument("--seed", type=int, default=2026)
    r = sub.add_parser("run", help="pre-register, identify, held-out verify on a device")
    r.add_argument("--config", choices=sorted(CONFIGS), required=True)
    r.add_argument("--device", choices=["cuda", "synthetic"], default="cuda")
    r.add_argument("--binary", default="./probe_mma")
    r.add_argument("--params", default="G4_F3_trunc_rz_chain", help="synthetic device only")
    r.add_argument("--out", default="run")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--n-heldout", type=int, default=20000, help="vectors per stratum")
    args = ap.parse_args(argv)
    return {"validate": validate, "run": run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
