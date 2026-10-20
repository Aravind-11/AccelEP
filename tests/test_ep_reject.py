"""Graphs AccelEP must leave on the CPU, or that both sessions must reject."""

import ctypes
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

LIB = Path(__file__).resolve().parents[1] / "build" / "libaccel_ep.so"
EP_NAME = "AccelEP"
MICROSOFT = helper.make_opsetid("com.microsoft", 1)


def model(nodes, inputs, outputs, initializers, extra_opsets=(), opset=17, infer=True):
    graph = helper.make_graph(nodes, "reject", inputs, outputs, initializers)
    built = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset), *extra_opsets])
    if not infer:
        return built
    return onnx.shape_inference.infer_shapes(built)


def bind(lib):
    lib.AccelEpResetStats.restype = None
    for name in (
        "AccelEpMatmulOps",
        "AccelEpGemmOps",
        "AccelEpKernelMatmuls",
        "AccelEpReluEpilogues",
        "AccelEpGeluEpilogues",
        "AccelEpAddEpilogues",
        "AccelEpDeviceTransposes",
        "AccelEpBytesToDevice",
        "AccelEpBytesToHost",
    ):
        getattr(lib, name).restype = ctypes.c_int64


def counters(lib):
    return (
        lib.AccelEpMatmulOps(),
        lib.AccelEpGemmOps(),
        lib.AccelEpKernelMatmuls(),
        lib.AccelEpReluEpilogues(),
        lib.AccelEpGeluEpilogues(),
        lib.AccelEpAddEpilogues(),
        lib.AccelEpDeviceTransposes(),
        lib.AccelEpBytesToDevice(),
        lib.AccelEpBytesToHost(),
    )


def main():
    lib = ctypes.CDLL(str(LIB))
    bind(lib)
    rng = np.random.default_rng(1)
    ort.register_execution_provider_library(EP_NAME, str(LIB))
    try:
        devices = [device for device in ort.get_ep_devices() if device.ep_name == EP_NAME]
        if len(devices) != 1:
            raise SystemExit("AccelEP device missing")

        def cpu_session(proto, strict_shape=False):
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if strict_shape:
                options.add_session_config_entry("session.strict_shape_type_inference", "1")
            return ort.InferenceSession(proto.SerializeToString(), options)

        def ep_session(proto, strict_shape=False):
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if strict_shape:
                options.add_session_config_entry("session.strict_shape_type_inference", "1")
            options.add_provider_for_devices(devices, {})
            return ort.InferenceSession(proto.SerializeToString(), options)

        def runnable(name, proto, feeds):
            lib.AccelEpResetStats()
            try:
                cpu = cpu_session(proto)
            except Exception as ex:
                raise SystemExit(f"{name} cpu construct failed: {ex}") from ex
            try:
                ep = ep_session(proto)
            except Exception as ex:
                if "AccelEP" in str(ex):
                    raise SystemExit(f"{name} construct mentioned AccelEP: {ex}") from ex
                raise
            if lib.AccelEpBytesToDevice() != 0 or lib.AccelEpMatmulOps() != 0 or lib.AccelEpGemmOps() != 0:
                raise SystemExit(f"{name} was claimed at construct: {counters(lib)}")
            try:
                got = ep.run(None, feeds)[0]
                want = cpu.run(None, feeds)[0]
            except Exception as ex:
                if "AccelEP" in str(ex) or "dynamic dimension" in str(ex):
                    raise SystemExit(f"{name} run failed: {ex}") from ex
                raise
            if got.shape != want.shape or got.dtype != want.dtype or not np.allclose(
                got, want, rtol=1e-5, atol=1e-5, equal_nan=True
            ):
                raise SystemExit(f"{name} mismatch\n got {got}\n want {want}")
            if any(value != 0 for value in counters(lib)):
                raise SystemExit(f"{name} counters {counters(lib)}")

        def load_failure(name, proto, strict_shape=False):
            lib.AccelEpResetStats()
            cpu_raised = False
            ep_raised = False
            try:
                cpu_session(proto, strict_shape)
            except Exception:
                cpu_raised = True
            try:
                ep_session(proto, strict_shape)
            except Exception:
                ep_raised = True
            if not cpu_raised or not ep_raised:
                raise SystemExit(f"{name} constructors cpu_raised={cpu_raised} ep_raised={ep_raised}")
            if any(value != 0 for value in counters(lib)):
                raise SystemExit(f"{name} counters after rejected load {counters(lib)}")

        x64 = rng.standard_normal((2, 3)).astype(np.float64)
        w64 = rng.standard_normal((3, 4)).astype(np.float64)
        runnable(
            "float64 matmul",
            model(
                [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                [
                    helper.make_tensor_value_info("X", TensorProto.DOUBLE, [2, 3]),
                    helper.make_tensor_value_info("W", TensorProto.DOUBLE, [3, 4]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [2, 4])],
                [],
            ),
            {"X": x64, "W": w64},
        )

        xk = rng.standard_normal((2, 5)).astype(np.float32)
        wk = rng.standard_normal((5, 3)).astype(np.float32)
        runnable(
            "dynamic k",
            model(
                [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, "k"]),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, ["k", 3]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [],
            ),
            {"X": xk, "W": wk},
        )

        rank_x = rng.standard_normal((4,)).astype(np.float32)
        rank_w = rng.standard_normal((4, 3)).astype(np.float32)
        runnable(
            "rank-1 matmul",
            model(
                [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
                [numpy_helper.from_array(rank_w, "W")],
            ),
            {"X": rank_x},
        )

        unranked = rng.standard_normal((4,)).astype(np.float32)
        unranked_b = rng.standard_normal((4, 3)).astype(np.float32)
        runnable(
            "unranked gemm",
            model(
                [helper.make_node("Gemm", ["A", "B"], ["Y"])],
                [helper.make_tensor_value_info("A", TensorProto.FLOAT, None)],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
                [numpy_helper.from_array(unranked_b, "B")],
                infer=False,
            ),
            {"A": unranked},
        )

        def float_pair(a_shape, b_shape):
            return (
                rng.standard_normal(a_shape).astype(np.float32),
                rng.standard_normal(b_shape).astype(np.float32),
            )

        a, b = float_pair((4, 2), (4, 3))
        runnable(
            "gemm transA 2",
            model(
                [helper.make_node("Gemm", ["A", "B"], ["Y"], transA=2)],
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 2]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [],
            ),
            {"A": a, "B": b},
        )
        a, b = float_pair((2, 4), (3, 4))
        runnable(
            "gemm transB 2",
            model(
                [helper.make_node("Gemm", ["A", "B"], ["Y"], transB=2)],
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [],
            ),
            {"A": a, "B": b},
        )
        a, b = float_pair((4, 2), (4, 3))
        runnable(
            "fused transA 2",
            model(
                [helper.make_node("FusedMatMul", ["A", "B"], ["Y"], domain="com.microsoft", transA=2)],
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 2]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [],
                extra_opsets=(MICROSOFT,),
            ),
            {"A": a, "B": b},
        )
        a, b = float_pair((2, 4), (3, 4))
        runnable(
            "fused transB -1",
            model(
                [helper.make_node("FusedMatMul", ["A", "B"], ["Y"], domain="com.microsoft", transB=-1)],
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [],
                extra_opsets=(MICROSOFT,),
            ),
            {"A": a, "B": b},
        )

        batch_a = rng.standard_normal((2, 2, 3)).astype(np.float32)
        batch_b = rng.standard_normal((2, 3, 4)).astype(np.float32)
        runnable(
            "transBatchA static-k",
            model(
                [helper.make_node("FusedMatMul", ["A", "B"], ["Y"], domain="com.microsoft", transBatchA=1)],
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2, 3]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3, 4]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 4])],
                [numpy_helper.from_array(batch_b, "B")],
                extra_opsets=(MICROSOFT,),
            ),
            {"A": batch_a},
        )

        relu_x = rng.standard_normal((3, 5)).astype(np.float32)
        runnable(
            "lone relu",
            model(
                [helper.make_node("Relu", ["X"], ["Y"])],
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])],
                [],
            ),
            {"X": relu_x},
        )
        add_a = rng.standard_normal((2, 3)).astype(np.float32)
        add_b = rng.standard_normal((2, 3)).astype(np.float32)
        runnable(
            "lone add",
            model(
                [helper.make_node("Add", ["A", "B"], ["Y"])],
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [],
            ),
            {"A": add_a, "B": add_b},
        )

        # Default sessions lenient-merge this declared [2,5] with the [2,4] product
        # (ORT 1.31 graph.cc MergeShapeInfo). The strict entry is what fails the load.
        # Do not allclose that resolved [2,4] product.
        load_failure(
            "matmul output dim",
            model(
                [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 4]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])],
                [],
                infer=False,
            ),
            strict_shape=True,
        )
        bad_w = rng.standard_normal((2, 2)).astype(np.float32)
        load_failure(
            "fused beta",
            model(
                [helper.make_node("FusedMatMul", ["X", "W"], ["Y"], domain="com.microsoft", beta=0.0)],
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
                [numpy_helper.from_array(bad_w, "W")],
                extra_opsets=(MICROSOFT,),
                infer=False,
            ),
        )
    finally:
        ort.unregister_execution_provider_library(EP_NAME)
    print("reject tests passed")


if __name__ == "__main__":
    main()
