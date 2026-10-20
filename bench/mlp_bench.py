"""Time one MatMul-Relu-MatMul on the CPU EP and on AccelEP.

The simulator can be slower than the CPU EP. This script records both medians
and does not compute a ratio.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

LIB = Path(__file__).resolve().parents[1] / "build" / "libaccel_ep.so"


def build(x, w1, w2):
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["R"]),
            helper.make_node("MatMul", ["R", "W2"], ["Y"]),
        ],
        "bench",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [x.shape[0], w2.shape[1]])],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return onnx.shape_inference.infer_shapes(model)


def session_options():
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def median_run(sess, feeds, warmup, iters):
    for _ in range(warmup):
        sess.run(None, feeds)
    samples = []
    last = None
    for _ in range(iters):
        started = time.perf_counter()
        last = sess.run(None, feeds)
        samples.append(time.perf_counter() - started)
    return last[0], float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=9)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    x = rng.integers(-2, 4, size=(128, 256)).astype(np.float32)
    w1 = rng.integers(-2, 4, size=(256, 256)).astype(np.float32)
    w2 = rng.integers(-2, 4, size=(256, 64)).astype(np.float32)
    want = np.maximum(x @ w1, 0) @ w2
    proto = build(x, w1, w2).SerializeToString()
    feeds = {"X": x}

    cpu = ort.InferenceSession(proto, session_options())
    cpu_out, cpu_median_s = median_run(cpu, feeds, args.warmup, args.iters)

    ort.register_execution_provider_library("AccelEP", str(LIB))
    try:
        devices = [device for device in ort.get_ep_devices() if device.ep_name == "AccelEP"]
        options = session_options()
        options.add_provider_for_devices(devices, {})
        accel = ort.InferenceSession(proto, options)
        import ctypes

        lib = ctypes.CDLL(str(LIB))
        lib.AccelEpResetStats.restype = None
        for name in (
            "AccelEpMatmulOps",
            "AccelEpGemmOps",
            "AccelEpKernelMatmuls",
            "AccelEpReluEpilogues",
            "AccelEpBytesToDevice",
            "AccelEpBytesToHost",
        ):
            getattr(lib, name).restype = ctypes.c_int64
        lib.AccelEpResetStats()
        measured = accel.run(None, feeds)[0]
        report = {
            "warmup": args.warmup,
            "iters": args.iters,
            "bytes_to_device": int(lib.AccelEpBytesToDevice()),
            "bytes_to_host": int(lib.AccelEpBytesToHost()),
            "matmul_ops": int(lib.AccelEpMatmulOps()),
            "gemm_ops": int(lib.AccelEpGemmOps()),
            "kernel_matmuls": int(lib.AccelEpKernelMatmuls()),
            "relu_epilogues": int(lib.AccelEpReluEpilogues()),
        }
        _timed, accel_median_s = median_run(accel, feeds, args.warmup, args.iters)
        report["accel_median_s"] = accel_median_s
        report["cpu_median_s"] = cpu_median_s
    finally:
        ort.unregister_execution_provider_library("AccelEP")

    if not np.allclose(measured, want, rtol=1e-5, atol=1e-5) or not np.allclose(cpu_out, want, rtol=1e-5, atol=1e-5):
        raise SystemExit("bench output mismatch")
    if report["bytes_to_device"] != 131072 or report["bytes_to_host"] != 32768:
        raise SystemExit(f"bench bytes {report['bytes_to_device']} {report['bytes_to_host']}")
    if (report["matmul_ops"], report["kernel_matmuls"], report["relu_epilogues"], report["gemm_ops"]) != (2, 2, 1, 0):
        raise SystemExit(f"bench counters {report}")
    accel_path = "/tmp/accel-bench-accel.npy"
    cpu_path = "/tmp/accel-bench-cpu.npy"
    np.save(accel_path, measured)
    np.save(cpu_path, cpu_out)
    report["accel_npy"] = accel_path
    report["cpu_npy"] = cpu_path
    print(json.dumps(report))


if __name__ == "__main__":
    main()
