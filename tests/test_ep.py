"""AccelEP runs MatMul, FusedMatMul, and Gemm. Every other op stays on CPU."""

import ctypes
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

LIB = Path(__file__).resolve().parents[1] / "build" / "libaccel_ep.so"
EP_NAME = "AccelEP"


def model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "accel", inputs, outputs, initializers)
    built = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return onnx.shape_inference.infer_shapes(built)


def contrib_model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "accel", inputs, outputs, initializers)
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )


def check(name, got, want):
    if not np.allclose(got, want, rtol=1e-5, atol=1e-5):
        raise SystemExit(f"{name} mismatch\n got {got}\n want {want}")


def main():
    lib = ctypes.CDLL(str(LIB))
    lib.AccelEpResetStats.restype = None
    for fn in (
        "AccelEpMatmulOps",
        "AccelEpGemmOps",
        "AccelEpFusedMatmulOps",
        "AccelEpKernelMatmuls",
        "AccelEpBytesToDevice",
        "AccelEpBytesToHost",
    ):
        getattr(lib, fn).restype = ctypes.c_int64

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w1 = rng.standard_normal((8, 16)).astype(np.float32)
    w2 = rng.standard_normal((16, 4)).astype(np.float32)
    hidden = x @ w1
    relu = np.maximum(hidden, 0)
    matmul_want = relu @ w2

    matmul_model = model(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["R"]),
            helper.make_node("MatMul", ["R", "W2"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )

    relu_x = rng.standard_normal((3, 5)).astype(np.float32)
    relu_model = model(
        [helper.make_node("Relu", ["X"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])],
        [],
    )

    gx = rng.standard_normal((2, 4)).astype(np.float32)
    gw = rng.standard_normal((3, 4)).astype(np.float32)
    gbias = rng.standard_normal((3,)).astype(np.float32)
    gv = rng.standard_normal((2, 3)).astype(np.float32)
    alpha = np.float32(1.5)
    beta = np.float32(0.5)
    gemm = alpha * (gx @ gw.T) + beta * gbias
    gemm_want = gemm + gv
    gemm_model = model(
        [
            helper.make_node(
                "Gemm",
                ["X", "W", "B"],
                ["G"],
                alpha=float(alpha),
                beta=float(beta),
                transA=0,
                transB=1,
            ),
            helper.make_node("Add", ["G", "V"], ["Y"]),
        ],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("V", TensorProto.FLOAT, [2, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(gw, "W"), numpy_helper.from_array(gbias, "B")],
    )

    cpu = ort.SessionOptions()
    cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu_matmul = ort.InferenceSession(matmul_model.SerializeToString(), cpu)
    check("cpu matmul", cpu_matmul.run(None, {"X": x})[0], matmul_want)

    ort.register_execution_provider_library(EP_NAME, str(LIB))
    try:
        devices = [device for device in ort.get_ep_devices() if device.ep_name == EP_NAME]
        if len(devices) != 1:
            raise SystemExit(f"expected one AccelEP device, found {[d.ep_name for d in ort.get_ep_devices()]}")

        def session(proto, optimize=False):
            options = ort.SessionOptions()
            if not optimize:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.add_provider_for_devices(devices, {})
            return ort.InferenceSession(proto.SerializeToString(), options)

        matmul_session = session(matmul_model)
        lib.AccelEpResetStats()
        got = matmul_session.run(None, {"X": x})[0]
        check("accel matmul+relu", got, matmul_want)
        matmul_ops = lib.AccelEpMatmulOps()
        gemm_ops = lib.AccelEpGemmOps()
        kernel_matmuls = lib.AccelEpKernelMatmuls()
        copied = lib.AccelEpBytesToDevice()
        # X and the Relu activation are copied. Both weights were uploaded at compile time.
        expect_copy = (x.size + relu.size) * x.itemsize
        if matmul_ops != 2 or gemm_ops != 0 or lib.AccelEpFusedMatmulOps() != 0 or kernel_matmuls != 2 or copied != expect_copy:
            raise SystemExit(
                "matmul offload stats "
                f"ops={matmul_ops} gemm={gemm_ops} kernels={kernel_matmuls} "
                f"copied={copied} expected_copy={expect_copy}"
            )

        relu_session = session(relu_model)
        lib.AccelEpResetStats()
        check("relu stays on cpu", relu_session.run(None, {"X": relu_x})[0], np.maximum(relu_x, 0))
        if (
            lib.AccelEpMatmulOps() != 0
            or lib.AccelEpGemmOps() != 0
            or lib.AccelEpFusedMatmulOps() != 0
            or lib.AccelEpBytesToDevice() != 0
        ):
            raise SystemExit("Relu was offloaded onto the accelerator")

        gemm_session = session(gemm_model)
        lib.AccelEpResetStats()
        check("accel gemm+add", gemm_session.run(None, {"X": gx, "V": gv})[0], gemm_want)
        gemm_copy = gx.size * gx.itemsize
        if (
            lib.AccelEpGemmOps() != 1
            or lib.AccelEpMatmulOps() != 0
            or lib.AccelEpFusedMatmulOps() != 0
            or lib.AccelEpKernelMatmuls() != 1
        ):
            raise SystemExit(
                "gemm offload stats "
                f"gemm={lib.AccelEpGemmOps()} matmul={lib.AccelEpMatmulOps()} "
                f"kernels={lib.AccelEpKernelMatmuls()}"
            )
        if lib.AccelEpBytesToDevice() != gemm_copy:
            raise SystemExit(
                f"gemm copied {lib.AccelEpBytesToDevice()} bytes, expected only X ({gemm_copy})"
            )

        fx = rng.standard_normal((4, 2)).astype(np.float32)
        fw = rng.standard_normal((4, 3)).astype(np.float32)
        fused_alpha = np.float32(1.5)
        fused_model = contrib_model(
            [
                helper.make_node(
                    "FusedMatMul",
                    ["X", "W"],
                    ["Y"],
                    domain="com.microsoft",
                    alpha=float(fused_alpha),
                    transA=1,
                    transB=0,
                ),
            ],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
            [numpy_helper.from_array(fw, "W")],
        )
        fused_want = fused_alpha * (fx.T @ fw)
        fused_session = session(fused_model)
        lib.AccelEpResetStats()
        check("fused matmul transA", fused_session.run(None, {"X": fx})[0], fused_want)
        fused_copy = fx.size * fx.itemsize
        if lib.AccelEpFusedMatmulOps() != 1 or lib.AccelEpMatmulOps() != 0 or lib.AccelEpGemmOps() != 0:
            raise SystemExit(
                "fused matmul offload stats "
                f"fused={lib.AccelEpFusedMatmulOps()} matmul={lib.AccelEpMatmulOps()} "
                f"gemm={lib.AccelEpGemmOps()}"
            )
        if lib.AccelEpKernelMatmuls() != 1 or lib.AccelEpBytesToDevice() != fused_copy:
            raise SystemExit(
                "fused matmul copy stats "
                f"kernels={lib.AccelEpKernelMatmuls()} copied={lib.AccelEpBytesToDevice()} "
                f"expected_copy={fused_copy}"
            )

        # Default optimizer leaves an authored FusedMatMul on this EP.
        fused_default = session(fused_model, optimize=True)
        lib.AccelEpResetStats()
        check("fused matmul default optimizer", fused_default.run(None, {"X": fx})[0], fused_want)
        if lib.AccelEpFusedMatmulOps() != 1 or lib.AccelEpBytesToDevice() != fused_copy:
            raise SystemExit(
                "default optimizer dropped FusedMatMul "
                f"fused={lib.AccelEpFusedMatmulOps()} matmul={lib.AccelEpMatmulOps()} "
                f"copied={lib.AccelEpBytesToDevice()}"
            )

        bx = rng.standard_normal((2, 1, 4, 3)).astype(np.float32)
        bw = rng.standard_normal((4, 5)).astype(np.float32)
        batch_alpha = np.float32(2)
        batch_model = contrib_model(
            [
                helper.make_node(
                    "FusedMatMul",
                    ["X", "W"],
                    ["H"],
                    domain="com.microsoft",
                    alpha=float(batch_alpha),
                    transA=1,
                ),
                helper.make_node("Relu", ["H"], ["Y"]),
            ],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 4, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 3, 5])],
            [numpy_helper.from_array(bw, "W")],
        )
        batch_want = np.maximum(batch_alpha * np.matmul(np.swapaxes(bx, -1, -2), bw), 0)
        batch_session = session(batch_model)
        lib.AccelEpResetStats()
        check("fused matmul broadcast+relu", batch_session.run(None, {"X": bx})[0], batch_want)
        batch_copy = bx.size * bx.itemsize
        # Two batches of X. W was uploaded at compile time.
        if (
            lib.AccelEpFusedMatmulOps() != 1
            or lib.AccelEpMatmulOps() != 0
            or lib.AccelEpKernelMatmuls() != 2
            or lib.AccelEpBytesToDevice() != batch_copy
        ):
            raise SystemExit(
                "broadcast fused matmul stats "
                f"fused={lib.AccelEpFusedMatmulOps()} matmul={lib.AccelEpMatmulOps()} "
                f"kernels={lib.AccelEpKernelMatmuls()} "
                f"copied={lib.AccelEpBytesToDevice()} expected_copy={batch_copy}"
            )

        # transBatchA moves axis 0 into the matrix-row slot. ORT's kernel accepts it only when both
        # inputs share a rank of at least 3: X (2, 3, 4) becomes logical (3, 2, 4). This EP leaves that on CPU.
        batch_x = rng.standard_normal((2, 3, 4)).astype(np.float32)
        batch_w = rng.standard_normal((3, 4, 5)).astype(np.float32)
        batch_transpose_model = contrib_model(
            [
                helper.make_node(
                    "FusedMatMul",
                    ["X", "W"],
                    ["Y"],
                    domain="com.microsoft",
                    transBatchA=1,
                ),
            ],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2, 5])],
            [numpy_helper.from_array(batch_w, "W")],
        )
        cpu_options = ort.SessionOptions()
        cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cpu_batch = ort.InferenceSession(batch_transpose_model.SerializeToString(), cpu_options)
        batch_transpose_session = session(batch_transpose_model)
        lib.AccelEpResetStats()
        check(
            "transBatch stays on cpu",
            batch_transpose_session.run(None, {"X": batch_x})[0],
            cpu_batch.run(None, {"X": batch_x})[0],
        )
        if lib.AccelEpFusedMatmulOps() != 0 or lib.AccelEpMatmulOps() != 0 or lib.AccelEpBytesToDevice() != 0:
            raise SystemExit("transBatch FusedMatMul was offloaded")

        tx = rng.standard_normal((4, 2)).astype(np.float32)
        tw = rng.standard_normal((4, 3)).astype(np.float32)
        transpose_model = model(
            [
                helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
                helper.make_node("MatMul", ["T", "W"], ["Y"]),
            ],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
            [numpy_helper.from_array(tw, "W")],
        )
        transpose_session = session(transpose_model, optimize=True)
        lib.AccelEpResetStats()
        check(
            "transpose+matmul default optimizer",
            transpose_session.run(None, {"X": tx})[0],
            tx.T @ tw,
        )
        accel_ops = lib.AccelEpMatmulOps() + lib.AccelEpFusedMatmulOps()
        transpose_copy = tx.size * tx.itemsize
        if accel_ops != 1 or lib.AccelEpGemmOps() != 0 or lib.AccelEpBytesToDevice() != transpose_copy:
            raise SystemExit(
                "transpose+matmul offload stats "
                f"matmul={lib.AccelEpMatmulOps()} fused={lib.AccelEpFusedMatmulOps()} "
                f"gemm={lib.AccelEpGemmOps()} copied={lib.AccelEpBytesToDevice()} "
                f"expected_copy={transpose_copy}"
            )
    finally:
        ort.unregister_execution_provider_library(EP_NAME)

    print("ep tests passed")
    print("matmul ops on accelerator: 2, relu left on CPU")
    print("gemm op on accelerator: 1, following add left on CPU")
    print("fused matmul op on accelerator: 1, transBatch left on CPU")


if __name__ == "__main__":
    main()
