"""AccelEP runs MatMul and Gemm. Every other op stays on CPU."""

import ctypes
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

LIB = Path(__file__).resolve().parents[1] / "build" / "libaccel_ep.so"
EP_NAME = "AccelEP"


def model(nodes, inputs, outputs, initializers, extra_opsets=()):
    graph = helper.make_graph(nodes, "accel", inputs, outputs, initializers)
    opsets = [helper.make_opsetid("", 17), *extra_opsets]
    built = helper.make_model(graph, opset_imports=opsets)
    return onnx.shape_inference.infer_shapes(built)


def check(name, got, want):
    if not np.allclose(got, want, rtol=1e-5, atol=1e-5):
        raise SystemExit(f"{name} mismatch\n got {got}\n want {want}")


def main():
    lib = ctypes.CDLL(str(LIB))
    lib.AccelEpResetStats.restype = None
    for fn in (
        "AccelEpMatmulOps",
        "AccelEpGemmOps",
        "AccelEpKernelMatmuls",
        "AccelEpReluEpilogues",
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

    # Default extended optimizer rewrites Transpose+MatMul into FusedMatMul.
    tx = rng.standard_normal((4, 2)).astype(np.float32)
    tw = rng.standard_normal((4, 3)).astype(np.float32)
    transpose_want = tx.T @ tw
    transpose_model = model(
        [
            helper.make_node("Transpose", ["X"], ["XT"], perm=[1, 0]),
            helper.make_node("MatMul", ["XT", "W"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(tw, "W")],
    )

    fx = rng.standard_normal((2, 4)).astype(np.float32)
    fw = rng.standard_normal((3, 4)).astype(np.float32)
    fused_alpha = np.float32(1.5)
    fused_want = fused_alpha * (fx @ fw.T)
    microsoft = helper.make_opsetid("com.microsoft", 1)
    fused_model = model(
        [
            helper.make_node(
                "FusedMatMul",
                ["X", "W"],
                ["Y"],
                domain="com.microsoft",
                transB=1,
                alpha=float(fused_alpha),
            )
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(fw, "W")],
        extra_opsets=(microsoft,),
    )

    # transBatchA permutes leading dimensions. That layout stays on the CPU EP.
    batch_a = rng.standard_normal((4, 2, 3, 5)).astype(np.float32)
    batch_b = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    batch_model = model(
        [helper.make_node("FusedMatMul", ["A", "B"], ["Y"], domain="com.microsoft", transBatchA=1)],
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 2, 3, 5]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3, 5, 6]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4, 6])],
        [],
        extra_opsets=(microsoft,),
    )

    cpu = ort.SessionOptions()
    cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu_matmul = ort.InferenceSession(matmul_model.SerializeToString(), cpu)
    check("cpu matmul", cpu_matmul.run(None, {"X": x})[0], matmul_want)

    optimized = ort.SessionOptions()
    optimized.optimized_model_filepath = "/tmp/accel-fused-matmul.onnx"
    ort.InferenceSession(transpose_model.SerializeToString(), optimized)
    fused_graph = onnx.load("/tmp/accel-fused-matmul.onnx")
    fused_nodes = [(node.domain, node.op_type) for node in fused_graph.graph.node]
    if fused_nodes != [("com.microsoft", "FusedMatMul")]:
        raise SystemExit(f"default optimizer did not produce FusedMatMul: {fused_nodes}")
    trans_a = {attr.name: attr.i for attr in fused_graph.graph.node[0].attribute}.get("transA")
    if trans_a != 1:
        raise SystemExit(f"FusedMatMul transA is {trans_a}, expected 1")

    batch_cpu = ort.SessionOptions()
    batch_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    batch_reference = ort.InferenceSession(batch_model.SerializeToString(), batch_cpu).run(
        None, {"A": batch_a, "B": batch_b}
    )[0]

    ort.register_execution_provider_library(EP_NAME, str(LIB))
    try:
        devices = [device for device in ort.get_ep_devices() if device.ep_name == EP_NAME]
        if len(devices) != 1:
            raise SystemExit(f"expected one AccelEP device, found {[d.ep_name for d in ort.get_ep_devices()]}")

        def session(proto, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL):
            options = ort.SessionOptions()
            if level is not None:
                options.graph_optimization_level = level
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
        copied_back = lib.AccelEpBytesToHost()
        relu_epilogues = lib.AccelEpReluEpilogues()
        # X is uploaded for the first matmul. The pre-Relu hidden stays on device and
        # the epilogue writes the activation. The next matmul copies that activation in.
        expect_copy = (x.size + relu.size) * x.itemsize
        expect_back = (relu.size + matmul_want.size) * x.itemsize
        if (
            matmul_ops != 2
            or gemm_ops != 0
            or kernel_matmuls != 2
            or relu_epilogues != 1
            or copied != expect_copy
            or copied_back != expect_back
        ):
            raise SystemExit(
                "matmul offload stats "
                f"ops={matmul_ops} gemm={gemm_ops} kernels={kernel_matmuls} "
                f"relu_epilogues={relu_epilogues} copied={copied} expected_copy={expect_copy} "
                f"copied_back={copied_back} expected_back={expect_back}"
            )

        relu_session = session(relu_model)
        lib.AccelEpResetStats()
        check("relu stays on cpu", relu_session.run(None, {"X": relu_x})[0], np.maximum(relu_x, 0))
        if (
            lib.AccelEpMatmulOps() != 0
            or lib.AccelEpGemmOps() != 0
            or lib.AccelEpReluEpilogues() != 0
            or lib.AccelEpBytesToDevice() != 0
        ):
            raise SystemExit("Relu was offloaded onto the accelerator")

        gemm_session = session(gemm_model)
        lib.AccelEpResetStats()
        check("accel gemm+add", gemm_session.run(None, {"X": gx, "V": gv})[0], gemm_want)
        gemm_copy = gx.size * gx.itemsize
        if lib.AccelEpGemmOps() != 1 or lib.AccelEpMatmulOps() != 0 or lib.AccelEpKernelMatmuls() != 1:
            raise SystemExit(
                "gemm offload stats "
                f"gemm={lib.AccelEpGemmOps()} matmul={lib.AccelEpMatmulOps()} "
                f"kernels={lib.AccelEpKernelMatmuls()}"
            )
        if lib.AccelEpBytesToDevice() != gemm_copy:
            raise SystemExit(
                f"gemm copied {lib.AccelEpBytesToDevice()} bytes, expected only X ({gemm_copy})"
            )
        if lib.AccelEpReluEpilogues() != 0:
            raise SystemExit("Add was folded into the Gemm epilogue")

        # Default session options leave the extended optimizer on.
        transpose_session = session(transpose_model, None)
        lib.AccelEpResetStats()
        check("fused transpose matmul", transpose_session.run(None, {"X": tx})[0], transpose_want)
        if lib.AccelEpMatmulOps() != 1 or lib.AccelEpGemmOps() != 0 or lib.AccelEpKernelMatmuls() != 1:
            raise SystemExit(
                "fused matmul offload stats "
                f"ops={lib.AccelEpMatmulOps()} gemm={lib.AccelEpGemmOps()} "
                f"kernels={lib.AccelEpKernelMatmuls()}"
            )
        if lib.AccelEpBytesToDevice() != tx.nbytes:
            raise SystemExit(
                f"fused matmul copied {lib.AccelEpBytesToDevice()} bytes, expected only X ({tx.nbytes})"
            )

        relu_transpose = model(
            [
                helper.make_node("Transpose", ["X"], ["XT"], perm=[1, 0]),
                helper.make_node("MatMul", ["XT", "W"], ["H"]),
                helper.make_node("Relu", ["H"], ["Y"]),
            ],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
            [numpy_helper.from_array(tw, "W")],
        )
        relu_transpose_session = session(relu_transpose, None)
        lib.AccelEpResetStats()
        check(
            "fused matmul relu",
            relu_transpose_session.run(None, {"X": tx})[0],
            np.maximum(transpose_want, 0),
        )
        if lib.AccelEpMatmulOps() != 1 or lib.AccelEpReluEpilogues() != 1 or lib.AccelEpBytesToDevice() != tx.nbytes:
            raise SystemExit(
                "fused matmul+relu stats "
                f"ops={lib.AccelEpMatmulOps()} relu={lib.AccelEpReluEpilogues()} "
                f"copied={lib.AccelEpBytesToDevice()} expected={tx.nbytes}"
            )

        fused_session = session(fused_model)
        lib.AccelEpResetStats()
        check("explicit fused matmul", fused_session.run(None, {"X": fx})[0], fused_want)
        if lib.AccelEpMatmulOps() != 1 or lib.AccelEpKernelMatmuls() != 1 or lib.AccelEpBytesToDevice() != fx.nbytes:
            raise SystemExit(
                "explicit fused matmul stats "
                f"ops={lib.AccelEpMatmulOps()} kernels={lib.AccelEpKernelMatmuls()} "
                f"copied={lib.AccelEpBytesToDevice()} expected={fx.nbytes}"
            )

        batch_session = session(batch_model)
        lib.AccelEpResetStats()
        check("transBatch stays on cpu", batch_session.run(None, {"A": batch_a, "B": batch_b})[0], batch_reference)
        if lib.AccelEpMatmulOps() != 0 or lib.AccelEpGemmOps() != 0 or lib.AccelEpBytesToDevice() != 0:
            raise SystemExit(
                "transBatch was offloaded "
                f"ops={lib.AccelEpMatmulOps()} gemm={lib.AccelEpGemmOps()} "
                f"copied={lib.AccelEpBytesToDevice()}"
            )
    finally:
        ort.unregister_execution_provider_library(EP_NAME)

    print("ep tests passed")
    print("matmul ops on accelerator: 2, relu epilogue kept the hidden activation on device")
    print("gemm op on accelerator: 1, following add left on CPU")
    print("fused matmul on accelerator: 1, copied only the activation")


if __name__ == "__main__":
    main()
