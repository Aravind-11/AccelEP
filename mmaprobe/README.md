# mmaprobe

Identify and verify the arithmetic of one tensor-core matrix instruction, bit for bit.

The question this answers: for a single `mma.sync` (accumulator bits read back before any later
rounding), is there an exact integer model of its adder, and does that model match the hardware
on inputs that were not used to build it?

## Status, stated plainly

| Piece | State |
| --- | --- |
| Exact softfloat (E4M3, E5M2, FP16, BF16, FP32), parameterized fused-adder model family | Done, unit-tested (17 tests) |
| Active identification (adaptive elimination over 1,920 hypotheses) | Done, validated on synthetic devices |
| Pre-registered held-out protocol, hash-chained ledger, shrinker, bounds | Done, unit-tested |
| CUDA probe (`cuda/probe_mma.cu`: `m16n8k32` E4M3 on sm_89, `m16n8k8` FP16 on sm_75) | Written. **Never compiled with nvcc and never run on a GPU.** Passes a host-side syntax check only |
| Result on real silicon | **None.** No GPU was available while building this |
| Comparison against published methods | **None.** See "What is not claimed" |

Run the layout and probe self-test on the GPU first. It exists because the fragment layouts
were written from the PTX documentation without hardware to check them.

## Layout

```text
mmaprobe/formats.py    exact decode / round-to-FP32 (integers only)
mmaprobe/model.py      model family: group size G, guard bits F, truncate|RNE alignment,
                       RZ|RNE output, C chained or added last
mmaprobe/probes.py     structured input generators used during identification
mmaprobe/identify.py   active elimination, plus the random-fit baseline
mmaprobe/heldout.py    held-out strata (generator is hashed at pre-registration), shrinker, bounds
mmaprobe/protocol.py   ledger, pre-registration, held-out run
mmaprobe/device.py     SyntheticDevice (hidden arithmetic) and CudaDevice
mmaprobe/layout.py     CPU mirror of the kernel's fragment layouts (consistency checks only)
cuda/probe_mma.cu      the probe; one warp, one mma, raw D[0][0] bits out
colab_run.sh           build and run on a Colab GPU
tests/test_all.py      unit tests
```

## Use

```bash
python3 tests/test_all.py                 # 17 tests, ~3 s
python3 -m mmaprobe validate              # synthetic validation, a few minutes

# on a GPU (Colab: Runtime > Change runtime type > GPU)
bash colab_run.sh
# or by hand
nvcc -O2 -arch=sm_89 -o probe_mma cuda/probe_mma.cu     # sm_75 for a T4
./probe_mma f16_k8 --selftest
python3 -m mmaprobe run --config e4m3_k32 --binary ./probe_mma --out run_e4m3
```

`e4m3_k32` needs nvcc 12.4 or newer and an sm_89+ GPU (L4, RTX 40xx, H100). On a T4, only
`f16_k8` runs, which is useful as a method check against published tensor-core results.

## Method

1. **Hypothesis family.** Products are exact. Each group of `G` consecutive products plus an
   accumulator term is aligned to a fixed-point grid whose LSB is `F` bits below the FP32 ulp of
   the largest term, bits shifted off are truncated or rounded-to-nearest-even, the integers are
   summed exactly, and the sum is rounded to FP32 (RZ or RNE). `C` is either chained through every
   group or added in a final step. That is 1,920 models for K=32.
2. **Active identification.** Keep every model alive. Generate candidate inputs (wide exponent
   spreads, exact cancellation, terms straddling group boundaries, accumulators far from the
   products), score each by how evenly it splits the survivors, query the hardware on the best one
   twice (determinism check), and drop every model whose prediction differs in any bit. If no model
   survives, the hardware is outside the family and the last query is the witness.
3. **Pre-registered held-out test.** Before identification, `heldout.py`'s SHA-256, the seed and the
   sample sizes go into a hash-chained ledger. Held-out vectors (uniform bit patterns, tightly
   matched exponents, near-cancellation, boundary values, NaN/Inf) are a pure function of the seed.
   The run refuses to start if the chain is broken, the file changed, or the order is wrong. Every
   surviving model is then checked; mismatches are shrunk to minimal counterexamples. Zero
   mismatches in N vectors gives a 95% upper bound of about 3/N on the mismatch rate.

## Validation on synthetic devices

A synthetic device runs a hidden model, so the answer is known. Numbers below are from
`python3 -m mmaprobe validate` (first run, seed 2026) and the same command with a fresh seed.

| Run | In-family devices fully resolved | Out-of-family devices flagged | Median hardware queries | Random-fit baseline: expected correct |
| --- | --- | --- | --- | --- |
| seed 2026 (tuned on) | 20 / 20 | 4 / 4 | 14 | 9.3% |
| seed 777 (fresh, no changes after) | 27 / 28 | 4 / 4 | 10 | 8.4% |

Queries count the duplicate determinism query, so 10 queries is 5 adaptive rounds. "Expected
correct" is the share of the baseline's tied best-fit models that behave like the hidden one.

The one fresh-seed miss (E4M3, hidden `G1_F8_rne_rne_chain`): 47 models survived identification and
all 47 had zero mismatches on 7,500 held-out vectors, including the hidden one. Identification did
not pick a single winner, and a different probe set separates them. The method reported the
ambiguity instead of a wrong answer, but it did not resolve it. Results are in
`results/validation.json` and `results_fresh/validation.json` (ledgers included).

What this does and does not show: it shows the identification loop recovers hidden members of
the family (27 of 28 fully on the fresh seed) with a median of 10 to 14 hardware queries, keeps the truth in every run, and that
the held-out step flags devices that behave outside the family. It does not show that any real
chip lies inside the family. The in-family devices were drawn from the same family the method
searches, so those results are a test of the machinery, not of the physics. The generator was
tuned once after the first run exposed early stopping on subtle models, so the fresh-seed run
is the out-of-sample number.

## What is not claimed

- No claim to beat any published method. The only comparison here is against a naive
  random-input fit (what an emulator author typically does), on synthetic devices.
  Prior work (for example Fasi et al. on tensor-core numerics, and FPRev on accumulation order)
  was not re-run or benchmarked; cite them by their own results.
- No claim about any real GPU until `colab_run.sh` has been run and its ledger inspected.
- The family is a hypothesis. Real units may do things it cannot express (stateful behavior,
  shape-dependent alignment, accumulator truncation on entry beyond `F >= -8`). The method reports
  that as "no model fits" with a witness.
- Zero signs and NaN payloads are compared loosely during identification; held-out reports
  zero-sign-only differences separately.
- The model and the held-out run are pure Python integers: fine for 10^5 vectors, too slow for
  10^8. A C++ port of the model is the obvious next step before a large run.
- The local ledger is self-attested. Commit the `prereg` line to a public repository before the
  device is first queried if an outside reader should trust the ordering.
