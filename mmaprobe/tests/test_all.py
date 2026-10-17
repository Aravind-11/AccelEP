import os
import random
import struct
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmaprobe import heldout as H
from mmaprobe import layout as L
from mmaprobe.device import CONFIGS, SyntheticDevice
from mmaprobe.formats import (BF16, E4M3, E5M2, FIN, FP16, FP32, canon_out, decode, round_to_fp32)
from mmaprobe.identify import baseline_random_fit, identify
from mmaprobe.model import Params, Prepared, evaluate, model_dot, model_space
from mmaprobe.probes import gen_candidate
from mmaprobe.protocol import Ledger, preregister, record_identified, run_heldout


class TestFormats(unittest.TestCase):
    def test_fp16_decode_matches_struct(self):
        for bits in range(1 << 16):
            k, s, m, e = decode(FP16, bits)
            if k != FIN:
                continue
            ref = struct.unpack("<e", struct.pack("<H", bits))[0]
            self.assertEqual(ref, (-1) ** s * m * 2.0 ** e)

    def test_e4m3_values(self):
        self.assertEqual(decode(E4M3, 0x38)[1:], (0, 8, -3))  # 1.0
        self.assertEqual(decode(E4M3, 0x7E)[1:], (0, 14, 5))  # 14 * 2**5 = 448
        self.assertEqual(decode(E4M3, 0x7F)[0], 2)            # NaN, no infinity in E4M3FN

    def test_round_fp32_rne_matches_struct(self):
        rng = random.Random(3)
        for _ in range(50000):
            mag = rng.getrandbits(rng.randint(1, 52)) or 1
            exp2 = rng.randint(-170, 100)
            s = rng.getrandbits(1)
            x = (-1) ** s * mag * 2.0 ** exp2
            try:
                ref = struct.unpack("<I", struct.pack("<f", x))[0]
            except OverflowError:
                continue
            self.assertEqual(canon_out(ref), canon_out(round_to_fp32(s, mag, exp2, "rne")))

    def test_round_rz_is_toward_zero(self):
        rng = random.Random(4)
        for _ in range(20000):
            mag = rng.getrandbits(rng.randint(25, 60)) | 1
            exp2 = rng.randint(-100, 50)
            rz = round_to_fp32(0, mag, exp2, "rz")
            self.assertLessEqual(Fraction(struct.unpack("<f", struct.pack("<I", rz))[0]),
                                 Fraction(mag) * Fraction(2) ** exp2)


class TestModel(unittest.TestCase):
    def test_wide_model_is_correctly_rounded_dot(self):
        """With a huge F, RNE alignment and one group, the model is the exact dot product."""
        rng = random.Random(5)
        prm = Params(8, 400, "rne", "rne", "chain")
        for _ in range(400):
            v = gen_candidate(FP16, 8, rng, fspan=20)
            exact = Fraction(0)
            ok = True
            for a, b in zip(v[0], v[1]):
                _, sa, ma, ea = decode(FP16, a)
                _, sb, mb, eb = decode(FP16, b)
                exact += (-1) ** (sa ^ sb) * ma * mb * Fraction(2) ** (ea + eb)
            _, sc, mc, ec = decode(FP32, v[2])
            exact += (-1) ** sc * mc * Fraction(2) ** ec
            num, den = exact.numerator, exact.denominator
            if num == 0:
                continue
            # den is a power of two: value = num * 2**-log2(den)
            e2 = -(den.bit_length() - 1)
            want = round_to_fp32(1 if num < 0 else 0, abs(num), e2, "rne")
            self.assertEqual(canon_out(want), canon_out(model_dot(FP16, prm, *v)))

    def test_truncation_drops_small_term(self):
        # 1.0 + 2**-30 with F=0 (24-bit grid): the small term is truncated away.
        one, tiny = 0x3C00, 0x0001  # fp16 1.0 and smallest subnormal 2**-24
        a = (one,) + (0,) * 7
        b = (one,) + (0,) * 7
        a2 = (one, tiny) + (0,) * 6
        b2 = (one, tiny) + (0,) * 6   # second product is 2**-48
        out = model_dot(FP16, Params(8, 0, "trunc", "rz", "chain"), a2, b2, 0)
        self.assertEqual(out, 0x3F800000)
        out_wide = model_dot(FP16, Params(8, 40, "trunc", "rz", "chain"), a2, b2, 0)
        self.assertEqual(out_wide, 0x3F800000)  # still below FP32 resolution after rounding

    def test_specials(self):
        nan16, inf16 = 0x7E00, 0x7C00
        prm = Params(8, 3, "trunc", "rz", "chain")
        self.assertEqual(canon_out(model_dot(FP16, prm, (nan16,) + (0,) * 7, (0x3C00,) + (0,) * 7, 0)),
                         canon_out(0x7FC00000))
        self.assertEqual(model_dot(FP16, prm, (inf16,) + (0,) * 7, (0x3C00,) + (0,) * 7, 0), 0x7F800000)
        self.assertEqual(canon_out(model_dot(FP16, prm, (inf16,) + (0,) * 7, (0,) * 8, 0)),
                         canon_out(0x7FC00000))  # inf * 0


class TestLayout(unittest.TestCase):
    def test_bijections(self):
        a = L.a_layout_e4m3_k32()
        self.assertEqual(len(set(a.values())), 16 * 32)
        b = L.b_layout_e4m3_k32()
        self.assertEqual(len(set(b.values())), 32 * 8)
        a2 = L.a_layout_f16_k8()
        self.assertEqual(len(set(a2.values())), 16 * 8)
        b2 = L.b_layout_f16_k8()
        self.assertEqual(len(set(b2.values())), 8 * 8)
        c = L.c_layout()
        self.assertEqual(len(set(c.values())), 16 * 8)

    def test_lane0_d0_is_c00_and_row0_col0_k_agree(self):
        self.assertEqual(L.c_layout()[(0, 0)], (0, 0))
        # every k of A row 0 and of B col 0 is held by the same lane in matching register slots
        a, b = L.a_layout_e4m3_k32(), L.b_layout_e4m3_k32()
        a_k = {col: key for key, (row, col) in a.items() if row == 0}
        b_k = {k: key for key, (k, col) in b.items() if col == 0}
        self.assertEqual(sorted(a_k), list(range(32)))
        self.assertEqual(sorted(b_k), list(range(32)))
        for k in range(32):
            self.assertEqual(a_k[k][0], b_k[k][0])  # same lane owns A[0][k] and B[k][0]


class TestIdentify(unittest.TestCase):
    def _check(self, cfg_name, prm, seed):
        cfg = CONFIGS[cfg_name]
        dev = SyntheticDevice(cfg.fmt, prm)
        res = identify(dev, cfg.fmt, cfg.K, random.Random(seed))
        self.assertFalse(res.empty)
        self.assertIn(prm, res.survivors)
        return res

    def test_recovers_hidden_params_f16(self):
        for i, prm in enumerate([Params(4, 3, "trunc", "rz", "chain"),
                                 Params(8, 0, "trunc", "rne", "chain"),
                                 Params(2, 7, "rne", "rz", "c_last")]):
            res = self._check("f16_k8", prm, 10 + i)
            self.assertEqual(len(res.classes), 1)

    def test_recovers_hidden_params_e4m3(self):
        res = self._check("e4m3_k32", Params(16, 2, "trunc", "rz", "chain"), 21)
        self.assertEqual(len(res.classes), 1)

    def test_out_of_family_device_is_detected(self):
        cfg = CONFIGS["f16_k8"]
        dev = SyntheticDevice(cfg.fmt, Params(4, 3, "trunc", "rz", "chain"), product_flush_exp=-10)
        res = identify(dev, cfg.fmt, cfg.K, random.Random(5))
        # the held-out protocol is what must catch it if identification did not:
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "ledger.jsonl"))
            preregister(led, cfg, 1, 600)
            if res.empty:
                return
            record_identified(led, res)
            out = run_heldout(led, dev, cfg, res.classes)
            self.assertGreater(out["reports"][0]["total_mismatches"], 0)

    def test_baseline_is_weaker(self):
        cfg = CONFIGS["f16_k8"]
        hidden = Params(4, 3, "trunc", "rz", "chain")
        dev = SyntheticDevice(cfg.fmt, hidden)
        tied = baseline_random_fit(dev, cfg.fmt, cfg.K, random.Random(1), 12)
        self.assertGreater(len(tied), 1)  # unit-scale random inputs cannot pin the model down


class TestHeldout(unittest.TestCase):
    def test_generation_is_deterministic(self):
        cfg = CONFIGS["f16_k8"]
        for s in H.STRATA:
            self.assertEqual(H.generate(s, cfg.fmt, cfg.K, 9, 50), H.generate(s, cfg.fmt, cfg.K, 9, 50))
            self.assertNotEqual(H.generate(s, cfg.fmt, cfg.K, 9, 50), H.generate(s, cfg.fmt, cfg.K, 10, 50))

    def test_upper_bound(self):
        self.assertAlmostEqual(H.upper_bound(0, 1000), 3.0 / 1000, delta=0.0002)
        self.assertGreater(H.upper_bound(5, 1000), 0.005)

    def test_protocol_guards(self):
        cfg = CONFIGS["f16_k8"]
        prm = Params(4, 3, "trunc", "rz", "chain")
        dev = SyntheticDevice(cfg.fmt, prm)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.jsonl")
            led = Ledger(path)
            with self.assertRaises(RuntimeError):
                run_heldout(led, dev, cfg, [[prm]])  # nothing pre-registered
            preregister(led, cfg, 7, 300)
            res = identify(dev, cfg.fmt, cfg.K, random.Random(2))
            record_identified(led, res)
            out = run_heldout(led, dev, cfg, res.classes)
            self.assertEqual(out["reports"][0]["total_mismatches"], 0)
            self.assertTrue(led.verify_chain())
            with open(path) as f:
                lines = f.read().splitlines()
            lines[0] = lines[0].replace('"seed": 7', '"seed": 8')
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            self.assertFalse(Ledger(path).verify_chain())

    def test_shrink_reaches_small_counterexample(self):
        cfg = CONFIGS["f16_k8"]
        truth = Params(4, 3, "trunc", "rz", "chain")
        wrong = Params(4, 3, "rne", "rz", "chain")
        rng = random.Random(0)
        for _ in range(500):
            v = gen_candidate(cfg.fmt, cfg.K, rng)
            if canon_out(model_dot(cfg.fmt, truth, *v)) != canon_out(model_dot(cfg.fmt, wrong, *v)):
                break
        fails = lambda x: canon_out(model_dot(cfg.fmt, truth, *x)) != canon_out(model_dot(cfg.fmt, wrong, *x))
        small = H.shrink(cfg.fmt, cfg.K, v, fails)
        self.assertTrue(fails(small))
        nz = lambda t: sum(1 for x, y in zip(t[0], t[1]) if x or y)
        self.assertLessEqual(nz(small), nz(v))


if __name__ == "__main__":
    unittest.main(verbosity=2)
