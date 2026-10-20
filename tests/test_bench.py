"""The MLP bench runs, matches NumPy, and does not report a speedup."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "mlp_bench.py"), "--warmup", "1", "--iters", "3"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    text = proc.stdout
    for banned in ("speedup", "ratio", "winner", "throughput", "tokens_per_s", "tokens/s"):
        if banned in text:
            raise SystemExit(f"bench JSON contains {banned}")
    report = json.loads(text)
    if report["bytes_to_device"] != 131072 or report["bytes_to_host"] != 32768:
        raise SystemExit(f"bench bytes {report}")
    if (report["matmul_ops"], report["kernel_matmuls"], report["relu_epilogues"], report["gemm_ops"]) != (2, 2, 1, 0):
        raise SystemExit(f"bench counters {report}")
    accel = np.load(report["accel_npy"])
    cpu = np.load(report["cpu_npy"])
    rng = np.random.default_rng(0)
    x = rng.integers(-2, 4, size=(128, 256)).astype(np.float32)
    w1 = rng.integers(-2, 4, size=(256, 256)).astype(np.float32)
    w2 = rng.integers(-2, 4, size=(256, 64)).astype(np.float32)
    want = np.maximum(x @ w1, 0) @ w2
    if not np.allclose(accel, want, rtol=1e-5, atol=1e-5) or not np.allclose(cpu, want, rtol=1e-5, atol=1e-5):
        raise SystemExit("bench npy mismatch")
    print("bench tests passed")


if __name__ == "__main__":
    main()
