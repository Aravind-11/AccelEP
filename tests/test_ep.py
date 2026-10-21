"""AccelEP runs MatMul and Gemm. A single-consumer RMSNormalization can stay on the device."""

import ctypes
import gc
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

LIB = Path(__file__).resolve().parents[1] / "build" / "libaccel_ep.so"
EP_NAME = "AccelEP"
MICROSOFT = helper.make_opsetid("com.microsoft", 1)
COUNTERS = (
    "AccelEpMatmulOps",
    "AccelEpGemmOps",
    "AccelEpKernelMatmuls",
    "AccelEpReluEpilogues",
    "AccelEpGeluEpilogues",
    "AccelEpAddEpilogues",
    "AccelEpDeviceTransposes",
    "AccelEpBytesToDevice",
    "AccelEpBytesToHost",
    "AccelEpLiveDeviceAllocs",
    "AccelEpConstantTouchCount",
)
TOUCHES = (
    "AccelEpConstantTouchId",
    "AccelEpConstantTouchAddress",
    "AccelEpConstantTouchBytes",
)


def model(nodes, inputs, outputs, initializers, extra_opsets=(), opset=17, ir_version=None, infer=True):
    graph = helper.make_graph(nodes, "accel", inputs, outputs, initializers)
    opsets = [helper.make_opsetid("", opset), *extra_opsets]
    built = helper.make_model(graph, opset_imports=opsets)
    if ir_version is not None:
        built.ir_version = ir_version
    if not infer:
        return built
    return onnx.shape_inference.infer_shapes(built)


def bind(lib):
    lib.AccelEpResetStats.restype = None
    lib.AccelEpClearConstantTouches.restype = None
    lib.AccelEpWriteProfile.restype = ctypes.c_int
    lib.AccelEpWriteProfile.argtypes = [ctypes.c_char_p]
    lib.AccelEpIsStreamAware.restype = ctypes.c_int
    for name in COUNTERS:
        getattr(lib, name).restype = ctypes.c_int64
    for name in TOUCHES:
        fn = getattr(lib, name)
        fn.restype = ctypes.c_uint64
        fn.argtypes = [ctypes.c_int64]


def snap(lib):
    return {
        "matmul": lib.AccelEpMatmulOps(),
        "gemm": lib.AccelEpGemmOps(),
        "kernels": lib.AccelEpKernelMatmuls(),
        "relu": lib.AccelEpReluEpilogues(),
        "gelu": lib.AccelEpGeluEpilogues(),
        "add": lib.AccelEpAddEpilogues(),
        "transposes": lib.AccelEpDeviceTransposes(),
        "to_device": lib.AccelEpBytesToDevice(),
        "to_host": lib.AccelEpBytesToHost(),
    }


def require(lib, name, **want):
    got = snap(lib)
    bad = {key: (got[key], value) for key, value in want.items() if got[key] != value}
    if bad:
        raise SystemExit(f"{name} stats mismatch {bad} full {got}")


def require_zero(lib, name):
    require(
        lib,
        name,
        matmul=0,
        gemm=0,
        kernels=0,
        relu=0,
        gelu=0,
        add=0,
        transposes=0,
        to_device=0,
        to_host=0,
    )


def check(name, got, want):
    if not np.allclose(got, want, rtol=1e-5, atol=1e-5):
        raise SystemExit(f"{name} mismatch\n got {got}\n want {want}")


def cpu_options(level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL, path=None, profile=False):
    options = ort.SessionOptions()
    if level is not None:
        options.graph_optimization_level = level
    if path is not None:
        options.optimized_model_filepath = path
    if profile:
        options.enable_profiling = True
    return options


def cpu_session(proto, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL, path=None, profile=False):
    return ort.InferenceSession(proto.SerializeToString(), cpu_options(level, path, profile))


def saved_graph(proto, level, path):
    cpu_session(proto, level, path)
    return onnx.load(path)


def assert_ops(graph, expected, label):
    found = [node.op_type for node in graph.graph.node]
    if found != expected:
        raise SystemExit(f"{label} ops {found} expected {expected}")


def fused_attrs(node):
    found = {}
    allowed = {"transA", "transB", "transBatchA", "transBatchB", "alpha"}
    for attr in node.attribute:
        if attr.name not in allowed:
            raise SystemExit(f"FusedMatMul stored unexpected attribute {attr.name}")
        if attr.name == "alpha":
            found[attr.name] = attr.f
        else:
            found[attr.name] = attr.i
    if found.get("transA") != 1:
        raise SystemExit(f"FusedMatMul transA is {found.get('transA')}, expected 1")
    for name in ("transB", "transBatchA", "transBatchB"):
        if name in found and found[name] != 0:
            raise SystemExit(f"FusedMatMul {name} is {found[name]}")
    if "alpha" in found and found["alpha"] != 1:
        raise SystemExit(f"FusedMatMul alpha is {found['alpha']}")
    return found


def prepare_chain():
    rng = np.random.default_rng(1)
    chain_x = rng.standard_normal((8, 2)).astype(np.float32)
    chain_w1 = rng.standard_normal((8, 16)).astype(np.float32)
    chain_w2 = rng.standard_normal((16, 4)).astype(np.float32)
    chain_model = model(
        [
            helper.make_node("Transpose", ["X"], ["XT"], perm=[1, 0]),
            helper.make_node("MatMul", ["XT", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["R"]),
            helper.make_node("MatMul", ["R", "W2"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [8, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(chain_w1, "W1"), numpy_helper.from_array(chain_w2, "W2")],
    )
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    graphs = {name: saved_graph(chain_model, level, f"/tmp/accel-chain-{name}.onnx") for name, level in levels.items()}
    for name in ("disable", "basic"):
        assert_ops(graphs[name], ["Transpose", "MatMul", "Relu", "MatMul"], name)
    for name in ("extended", "all"):
        assert_ops(graphs[name], ["FusedMatMul", "Relu", "MatMul"], name)
        fused = graphs[name].graph.node[0]
        if fused.domain != "com.microsoft":
            raise SystemExit(f"{name} FusedMatMul domain {fused.domain}")
        fused_attrs(fused)
    return {
        "x": chain_x,
        "want": np.maximum(chain_x.T @ chain_w1, 0) @ chain_w2,
        "extended": graphs["extended"],
    }


def device_rms(value, scale, eps=1e-5):
    """Double sum, then float(x * inv) * weight. This is Device::RmsNorm, not ORT's layer norm."""
    value = np.asarray(value, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    x = value.astype(np.float64)
    dim = x.shape[-1]
    square = np.sum(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(square / dim + np.float64(eps))
    return (x * inv).astype(np.float32) * scale


def layer_norm(value, scale):
    value = np.asarray(value, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    mean = value.mean(axis=-1, keepdims=True).astype(np.float32)
    centered = (value - mean).astype(np.float32)
    var = np.mean(centered * centered, axis=-1, keepdims=True).astype(np.float32)
    return (centered / np.sqrt(var + np.float32(1e-5)) * scale).astype(np.float32)


def stable_softmax(value):
    value = np.asarray(value, dtype=np.float32)
    shifted = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float32))
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def write_profile(lib, path):
    encoded = None if path is None else path.encode()
    return lib.AccelEpWriteProfile(encoded)


def trace_events(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("traceEvents", [])
    return payload


def kernel_events(path):
    return [event for event in trace_events(path) if str(event.get("name", "")).endswith("_kernel_time")]


def run_unclaimed(lib, session, proto, feeds, name):
    cpu_exc = None
    ep_exc = None
    cpu_out = None
    ep_out = None
    try:
        cpu_out = cpu_session(proto).run(None, feeds)
    except Exception as ex:
        cpu_exc = ex
    lib.AccelEpResetStats()
    try:
        ep_out = session(proto).run(None, feeds)
    except Exception as ex:
        ep_exc = ex
    if ep_exc is not None and ("AccelEP" in str(ep_exc) or "dynamic dimension" in str(ep_exc)):
        raise SystemExit(f"{name} {ep_exc}")
    if cpu_exc is None and ep_exc is None:
        check(name, ep_out[0], cpu_out[0])
    elif cpu_exc is None and ep_exc is not None:
        raise SystemExit(f"{name} EP failed: {ep_exc}")
    elif cpu_exc is not None and ep_exc is None:
        raise SystemExit(f"{name} CPU failed but EP ran: {cpu_exc}")
    require_zero(lib, name)


def run_once(lib, session, proto, feeds):
    sess = session(proto)
    lib.AccelEpResetStats()
    return sess, sess.run(None, feeds)[0]


def bits_of(value):
    return np.ascontiguousarray(value).view(np.uint32)


def profile_chain(lib, session):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w1 = rng.standard_normal((8, 16)).astype(np.float32)
    w2 = rng.standard_normal((16, 4)).astype(np.float32)
    proto = model(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["R"]),
            helper.make_node("MatMul", ["R", "W2"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )
    _, got = run_once(lib, session, proto, {"X": x})
    check("profile chain", got, np.maximum(x @ w1, 0) @ w2)
    require(lib, "profile chain", matmul=2, kernels=2, relu=1, gemm=0, to_device=x.nbytes, to_host=got.nbytes)
    path = "/tmp/accel-ep-profile.json"
    if write_profile(lib, path) != 0:
        raise SystemExit("WriteProfile failed")
    raw = Path(path).read_bytes()
    expected = (
        b'{"records":['
        b'{"op":"MatMul","activation":"relu","inputs":['
        b'{"name":"X","shape":[2,8]},{"name":"W1","shape":[8,16]}'
        b'],"output":{"name":"R","shape":[2,16]},"bytes_to_device":64,"bytes_to_host":0},'
        b'{"op":"MatMul","activation":"none","inputs":['
        b'{"name":"R","shape":[2,16]},{"name":"W2","shape":[16,4]}'
        b'],"output":{"name":"Y","shape":[2,4]},"bytes_to_device":0,"bytes_to_host":32}'
        b"]}"
    )
    if raw != expected:
        raise SystemExit(f"profile json {raw!r}")
    if b'"name":"H"' in raw:
        raise SystemExit("profile named the pre-Relu activation")
    if write_profile(lib, path) != 0 or Path(path).read_bytes() != raw:
        raise SystemExit("second WriteProfile changed the log")
    if write_profile(lib, None) == 0 or Path(path).read_bytes() != raw:
        raise SystemExit("null WriteProfile cleared or rewrote the log")
    if write_profile(lib, "/tmp") == 0 or Path(path).read_bytes() != raw:
        raise SystemExit("directory WriteProfile cleared or rewrote the log")
    lib.AccelEpResetStats()
    empty = "/tmp/accel-ep-profile-empty.json"
    if write_profile(lib, empty) != 0 or Path(empty).read_bytes() != b'{"records":[]}':
        raise SystemExit(f"empty profile is {Path(empty).read_bytes()!r}")


def test_float16(lib, session):
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16)
    w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float16)
    proto = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3])],
        [numpy_helper.from_array(w, "W")],
    )
    lib.AccelEpResetStats()
    message = ""
    out = None
    try:
        out = session(proto).run(None, {"X": x})[0]
    except Exception as ex:
        message = str(ex)
    if "AccelEP" in message:
        raise SystemExit(f"float16 raised AccelEP: {message}")
    require_zero(lib, "float16")
    if out is not None and (out.shape != (2, 3) or out.dtype != np.float16):
        raise SystemExit(f"float16 returned {out.shape} {out.dtype}")


def test_dequantize(lib, session):
    x = np.array([[1, 2], [0, -1]], dtype=np.float32)
    w = np.array([[-128, 127], [1, -2]], dtype=np.int8)
    scale = np.array([0.5, 2.0], dtype=np.float32)
    zp = np.array([1, -3], dtype=np.int8)
    wf = (w.astype(np.float32) - zp.astype(np.float32).reshape(2, 1)) * scale.reshape(2, 1)
    want = x @ wf
    if want[0, 0] != np.float32(-48.5):
        raise SystemExit(f"dq oracle {want}")

    def dq_model(nodes, initializers, output_shape, opset=13):
        return model(
            nodes,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
            initializers,
            opset=opset,
        )

    base_inits = [
        numpy_helper.from_array(w, "W8"),
        numpy_helper.from_array(scale, "S"),
        numpy_helper.from_array(zp, "Z"),
    ]
    proto = dq_model(
        [
            helper.make_node("DequantizeLinear", ["W8", "S", "Z"], ["W"], axis=0),
            helper.make_node("MatMul", ["X", "W"], ["Y"]),
        ],
        base_inits,
        [2, 2],
    )
    _, got = run_once(lib, session, proto, {"X": x})
    check("dq matmul", got, want)
    require(lib, "dq matmul", matmul=1, gemm=0, kernels=1, relu=0, gelu=0, add=0, transposes=0, to_device=16, to_host=16)

    relu_proto = dq_model(
        [
            helper.make_node("DequantizeLinear", ["W8", "S", "Z"], ["W"], axis=0),
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Relu", ["H"], ["Y"]),
        ],
        base_inits,
        [2, 2],
    )
    _, got = run_once(lib, session, relu_proto, {"X": x})
    check("dq relu", got, np.maximum(want, 0))
    require(lib, "dq relu", matmul=1, kernels=1, relu=1, gelu=0, to_device=16, to_host=16)

    float_proto = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        [numpy_helper.from_array(wf, "W")],
    )
    _, got = run_once(lib, session, float_proto, {"X": x})
    check("float wf", got, want)
    require(lib, "float wf", matmul=1, kernels=1, to_device=16, to_host=16)

    scalar = numpy_helper.from_array(np.float32(0.5), "S")
    no_zp = dq_model(
        [
            helper.make_node("DequantizeLinear", ["W8", "S"], ["W"]),
            helper.make_node("MatMul", ["X", "W"], ["Y"]),
        ],
        [numpy_helper.from_array(w, "W8"), scalar],
        [2, 2],
    )
    _, got = run_once(lib, session, no_zp, {"X": x})
    check("dq no zp", got, x @ (w.astype(np.float32) * np.float32(0.5)))
    require(lib, "dq no zp", matmul=1, kernels=1, relu=0, to_device=16, to_host=16)

    blocked = model(
        [
            helper.make_node("DequantizeLinear", ["W8", "S"], ["W"], axis=0, block_size=2),
            helper.make_node("MatMul", ["X", "W"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        [numpy_helper.from_array(w, "W8"), numpy_helper.from_array(np.array([[0.5, 2.0]], dtype=np.float32), "S")],
        opset=21,
    )
    cpu = cpu_session(blocked).run(None, {"X": x})[0]
    _, got = run_once(lib, session, blocked, {"X": x})
    check("dq block", got, cpu)
    require(lib, "dq block", matmul=1, gemm=0, kernels=1, relu=0, to_device=32, to_host=16)

    u8 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    uint_proto = dq_model(
        [
            helper.make_node("DequantizeLinear", ["W8", "S"], ["W"]),
            helper.make_node("MatMul", ["X", "W"], ["Y"]),
        ],
        [numpy_helper.from_array(u8, "W8"), scalar],
        [2, 2],
    )
    cpu = cpu_session(uint_proto).run(None, {"X": x})[0]
    _, got = run_once(lib, session, uint_proto, {"X": x})
    check("dq uint8", got, cpu)
    require(lib, "dq uint8", matmul=1, kernels=1, to_device=32, to_host=16)

    no_axis = dq_model(
        [
            helper.make_node("DequantizeLinear", ["W8", "S"], ["W"]),
            helper.make_node("MatMul", ["X", "W"], ["Y"]),
        ],
        [numpy_helper.from_array(w, "W8"), numpy_helper.from_array(scale, "S")],
        [2, 2],
    )
    cpu = cpu_session(no_axis).run(None, {"X": x})[0]
    _, got = run_once(lib, session, no_axis, {"X": x})
    check("dq rank1 no axis", got, cpu)
    require(lib, "dq rank1 no axis", matmul=1, kernels=1, to_device=32, to_host=16)

    both = model(
        [
            helper.make_node("DequantizeLinear", ["A8", "SA"], ["A"]),
            helper.make_node("DequantizeLinear", ["B8", "SB"], ["B"]),
            helper.make_node("MatMul", ["A", "B"], ["Y"]),
        ],
        [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        [
            numpy_helper.from_array(w, "A8"),
            numpy_helper.from_array(np.float32(0.5), "SA"),
            numpy_helper.from_array(w, "B8"),
            numpy_helper.from_array(np.float32(0.25), "SB"),
        ],
        opset=13,
    )
    cpu = cpu_session(both).run(None, {})[0]
    _, got = run_once(lib, session, both, {})
    check("dq both", got, cpu)
    require(lib, "dq both", matmul=1, kernels=1, relu=0, to_device=32, to_host=16)

    a = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)
    b = np.array([[1, 0], [0, 1], [1, -1]], dtype=np.int8)
    integer = model(
        [helper.make_node("MatMulInteger", ["A", "B"], ["Y"])],
        [
            helper.make_tensor_value_info("A", TensorProto.INT8, [2, 3]),
            helper.make_tensor_value_info("B", TensorProto.INT8, [3, 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [2, 2])],
        [],
    )
    run_unclaimed(lib, session, integer, {"A": a, "B": b}, "matmul integer")


def test_leading(lib, session):
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    proto = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 4])],
        [numpy_helper.from_array(w, "W")],
    )
    sess = session(proto)
    for batch, device_bytes, host_bytes in ((1, 32, 16), (4, 128, 64)):
        x = rng.standard_normal((batch, 8)).astype(np.float32)
        lib.AccelEpResetStats()
        got = sess.run(None, {"X": x})[0]
        check(f"leading batch {batch}", got, x @ w)
        require(
            lib,
            f"leading batch {batch}",
            matmul=1,
            gemm=0,
            kernels=1,
            relu=0,
            gelu=0,
            add=0,
            transposes=0,
            to_device=device_bytes,
            to_host=host_bytes,
        )
    del sess

    named = rng.standard_normal((4, 3)).astype(np.float32)
    named_x = rng.standard_normal((2, 4)).astype(np.float32)
    named_proto = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 3])],
        [numpy_helper.from_array(named, "W")],
    )
    _, got = run_once(lib, session, named_proto, {"X": named_x})
    check("leading name N", got, named_x @ named)
    require(lib, "leading name N", matmul=1, kernels=1, gemm=0, relu=0, to_device=32, to_host=24, transposes=0)

    relu_w = rng.standard_normal((8, 4)).astype(np.float32)
    relu_x = rng.standard_normal((1, 8)).astype(np.float32)
    relu_proto = model(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Relu", ["H"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 4])],
        [numpy_helper.from_array(relu_w, "W")],
    )
    _, got = run_once(lib, session, relu_proto, {"X": relu_x})
    check("leading relu stays cpu", got, np.maximum(relu_x @ relu_w, 0))
    require(lib, "leading relu stays cpu", matmul=1, kernels=1, relu=0, gelu=0, to_device=32, to_host=16)

    xk = rng.standard_normal((2, 8)).astype(np.float32)
    wk = rng.standard_normal((8, 4)).astype(np.float32)
    dynamic_k = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, "k"]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, ["k", 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [],
    )
    run_unclaimed(lib, session, dynamic_k, {"X": xk, "W": wk}, "dynamic k")

    a = rng.standard_normal((1, 8, 4)).astype(np.float32)
    b = rng.standard_normal((4, 4, 5)).astype(np.float32)
    broadcast = model(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, ["batch", 8, 4]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 4, 5]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 8, 5])],
        [],
    )
    run_unclaimed(lib, session, broadcast, {"A": a, "B": b}, "dynamic batch broadcast")


def test_residual(lib, session):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 2, 3)).astype(np.float32)
    w = rng.standard_normal((3, 4)).astype(np.float32)
    r = rng.standard_normal((2, 2, 4)).astype(np.float32)
    product = x @ w
    want = product + r

    def residual_model(add_inputs):
        return model(
            [
                helper.make_node("MatMul", ["X", "W"], ["H"]),
                helper.make_node("Add", add_inputs, ["Y"]),
            ],
            [
                helper.make_tensor_value_info("R", TensorProto.FLOAT, [2, 2, 4]),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 3]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 4])],
            [numpy_helper.from_array(w, "W")],
        )

    for label, inputs in (("add H R", ["H", "R"]), ("add R H", ["R", "H"])):
        _, got = run_once(lib, session, residual_model(inputs), {"R": r, "X": x})
        check(label, got, want)
        require(
            lib,
            label,
            matmul=1,
            gemm=0,
            kernels=2,
            relu=0,
            gelu=0,
            add=1,
            transposes=0,
            to_device=112,
            to_host=64,
        )

    v = rng.standard_normal((4,)).astype(np.float32)
    broadcast = model(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Add", ["H", "V"], ["Y"]),
        ],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 3]),
            helper.make_tensor_value_info("V", TensorProto.FLOAT, [4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 4])],
        [numpy_helper.from_array(w, "W")],
    )
    _, got = run_once(lib, session, broadcast, {"X": x, "V": v})
    check("residual broadcast", got, product + v)
    require(lib, "residual broadcast", matmul=1, kernels=2, add=0, relu=0, to_device=48, to_host=64)

    relu_proto = model(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Add", ["H", "R"], ["S"]),
            helper.make_node("Relu", ["S"], ["Y"]),
        ],
        [
            helper.make_tensor_value_info("R", TensorProto.FLOAT, [2, 2, 4]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 4])],
        [numpy_helper.from_array(w, "W")],
    )
    _, got = run_once(lib, session, relu_proto, {"R": r, "X": x})
    check("residual then relu", got, np.maximum(want, 0))
    require(lib, "residual then relu", matmul=1, kernels=2, relu=0, add=1, to_device=112, to_host=64)


def gelu_proto(x, weight, approximate=None, extra_nodes=(), outputs=None, opset=20, domain=""):
    nodes = [helper.make_node("MatMul", ["X", "W"], ["H"])]
    gelu_kwargs = {}
    if approximate is not None:
        gelu_kwargs["approximate"] = approximate
    if domain:
        gelu_kwargs["domain"] = domain
    nodes.append(helper.make_node("Gelu", ["H"], ["Y"], **gelu_kwargs))
    nodes.extend(extra_nodes)
    if outputs is None:
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])]
    return model(
        nodes,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        outputs,
        [numpy_helper.from_array(weight, "W")],
        opset=opset,
        extra_opsets=(MICROSOFT,) if domain == "com.microsoft" else (),
    )


def test_gelu(lib, session):
    # A NaN in X poisons every MatMul output: each dot product multiplies that lane by 0.
    # Put the NaN in one column of W so Gelu(2.7) and Gelu(-1) stay finite and Gelu(NaN) stays NaN.
    x = np.array([[2.7, -1.0, 1.0]], dtype=np.float32)
    weight = np.eye(3, dtype=np.float32)
    nan_weight = weight.copy()
    nan_weight[2, 2] = np.float32(np.nan)
    proto = gelu_proto(x, nan_weight)
    cpu = cpu_session(proto).run(None, {"X": x})[0]
    _, got = run_once(lib, session, proto, {"X": x})
    if not np.allclose(got[0, :2], cpu[0, :2], rtol=1e-5, atol=1e-5) or not np.isnan(got[0, 2]):
        raise SystemExit(f"gelu mismatch\n got {got}\n cpu {cpu}")
    require(lib, "gelu", matmul=1, gemm=0, kernels=1, relu=0, gelu=1, add=0, transposes=0, to_device=12, to_host=12)

    none = gelu_proto(x, nan_weight, approximate="none")
    cpu = cpu_session(none).run(None, {"X": x})[0]
    _, got = run_once(lib, session, none, {"X": x})
    if not np.allclose(got[0, :2], cpu[0, :2], rtol=1e-5, atol=1e-5) or not np.isnan(got[0, 2]):
        raise SystemExit(f"gelu none mismatch\n got {got}\n cpu {cpu}")
    require(lib, "gelu none", matmul=1, kernels=1, gelu=1, relu=0, to_device=12, to_host=12)

    tanh = gelu_proto(x, nan_weight, approximate="tanh")
    cpu = cpu_session(tanh).run(None, {"X": x})[0]
    _, got = run_once(lib, session, tanh, {"X": x})
    if not np.allclose(got[0, :2], cpu[0, :2], rtol=1e-5, atol=1e-5, equal_nan=True):
        raise SystemExit(f"gelu tanh mismatch\n got {got}\n cpu {cpu}")
    require(lib, "gelu tanh", matmul=1, kernels=1, gelu=0, relu=0, to_device=12, to_host=12)

    for label, approx in (("gelu None", "None"), ("gelu empty", "")):
        try:
            variant = gelu_proto(x, nan_weight, approximate=approx)
        except Exception as ex:
            if "AccelEP" in str(ex):
                raise SystemExit(f"{label} builder raised AccelEP: {ex}") from ex
            lib.AccelEpResetStats()
            require_zero(lib, label)
            continue
        cpu_exc = None
        try:
            cpu = cpu_session(variant).run(None, {"X": x})[0]
        except Exception as ex:
            cpu_exc = ex
        lib.AccelEpResetStats()
        ep_exc = None
        got = None
        try:
            got = session(variant).run(None, {"X": x})[0]
        except Exception as ex:
            ep_exc = ex
        if ep_exc is not None and "AccelEP" in str(ep_exc):
            raise SystemExit(f"{label} {ep_exc}")
        if cpu_exc is not None:
            require_zero(lib, label)
            continue
        if ep_exc is not None:
            raise SystemExit(f"{label} EP failed: {ep_exc}")
        # Weight upload happened before the per-run reset inside a successful session.
        # Reconstruct so the byte check excludes it.
        _, got = run_once(lib, session, variant, {"X": x})
        if not np.allclose(got[0, :2], cpu[0, :2], rtol=1e-5, atol=1e-5, equal_nan=True):
            raise SystemExit(f"{label} mismatch")
        require(lib, label, gelu=0, matmul=1, to_device=12, to_host=12)

    fanout = gelu_proto(
        x,
        nan_weight,
        extra_nodes=[helper.make_node("Add", ["H", "H"], ["Z"])],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3]),
        ],
    )
    _, got = run_once(lib, session, fanout, {"X": x})
    require(lib, "gelu fanout", matmul=1, kernels=1, gelu=0, relu=0, to_device=12, to_host=12)
    del got

    graph_out = model(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Gelu", ["H"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [
            helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3]),
        ],
        [numpy_helper.from_array(weight, "W")],
        opset=20,
    )
    sess = session(graph_out)
    lib.AccelEpResetStats()
    h, y = sess.run(None, {"X": x})
    del y
    check("gelu graph output edge", h, x)
    require(lib, "gelu graph output", matmul=1, kernels=1, gelu=0, to_device=12, to_host=12)

    for op_type in ("Gelu", "FastGelu"):
        nodes = [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node(op_type, ["H"], ["Y"], domain="com.microsoft"),
        ]
        try:
            contrib = model(
                nodes,
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
                [numpy_helper.from_array(weight, "W")],
                opset=20,
                extra_opsets=(MICROSOFT,),
            )
        except Exception as ex:
            if "AccelEP" in str(ex):
                raise SystemExit(f"{op_type} builder raised AccelEP: {ex}") from ex
            lib.AccelEpResetStats()
            require_zero(lib, op_type)
            continue
        cpu_exc = None
        cpu = None
        try:
            cpu = cpu_session(contrib).run(None, {"X": x})[0]
        except Exception as ex:
            cpu_exc = ex
        lib.AccelEpResetStats()
        ep_exc = None
        try:
            session(contrib).run(None, {"X": x})
        except Exception as ex:
            ep_exc = ex
        if ep_exc is not None and "AccelEP" in str(ep_exc):
            raise SystemExit(f"{op_type} {ep_exc}")
        if cpu_exc is not None:
            require_zero(lib, op_type)
            continue
        if ep_exc is not None:
            raise SystemExit(f"{op_type} EP failed: {ep_exc}")
        _, got = run_once(lib, session, contrib, {"X": x})
        check(op_type, got, cpu)
        require(lib, op_type, gelu=0)


def test_two_sessions(lib, session):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w1 = rng.standard_normal((8, 16)).astype(np.float32)
    w2 = rng.standard_normal((16, 4)).astype(np.float32)
    chain = model(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["R"]),
            helper.make_node("MatMul", ["R", "W2"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )
    gx = rng.standard_normal((2, 4)).astype(np.float32)
    gw = rng.standard_normal((3, 4)).astype(np.float32)
    gbias = rng.standard_normal((3,)).astype(np.float32)
    gv = rng.standard_normal((2, 3)).astype(np.float32)
    gemm = model(
        [
            helper.make_node("Gemm", ["X", "W", "B"], ["G"], alpha=1.5, beta=0.5, transA=0, transB=1),
            helper.make_node("Add", ["G", "V"], ["Y"]),
        ],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("V", TensorProto.FLOAT, [2, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(gw, "W"), numpy_helper.from_array(gbias, "B")],
    )
    session_a = session(chain)
    session_b = session(gemm)
    lib.AccelEpResetStats()
    check("session A", session_a.run(None, {"X": x})[0], np.maximum(x @ w1, 0) @ w2)
    check("session B", session_b.run(None, {"X": gx, "V": gv})[0], np.float32(1.5) * (gx @ gw.T) + np.float32(0.5) * gbias + gv)
    require(lib, "two sessions", matmul=2, gemm=1, kernels=3, relu=1)
    live = lib.AccelEpLiveDeviceAllocs()
    touches = lib.AccelEpConstantTouchCount()
    lib.AccelEpResetStats()
    if lib.AccelEpLiveDeviceAllocs() != live or lib.AccelEpConstantTouchCount() != touches:
        raise SystemExit("reset cleared live allocs or constant touches")
    empty = "/tmp/accel-ep-profile-sessions.json"
    if write_profile(lib, empty) != 0 or Path(empty).read_bytes() != b'{"records":[]}':
        raise SystemExit("reset left profile records")
    lib.AccelEpResetStats()
    check("session B only", session_b.run(None, {"X": gx, "V": gv})[0], np.float32(1.5) * (gx @ gw.T) + np.float32(0.5) * gbias + gv)
    require(
        lib,
        "session B only",
        matmul=0,
        gemm=1,
        kernels=1,
        relu=0,
        gelu=0,
        add=0,
        transposes=0,
        to_device=32,
        to_host=24,
    )
    del session_a, session_b
    gc.collect()


def test_prenorm(lib, session):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 4, 8)).astype(np.float32)
    scale = np.ones((8,), dtype=np.float32)
    w1 = rng.standard_normal((8, 8)).astype(np.float32)
    w2 = rng.standard_normal((8, 8)).astype(np.float32)
    proto = model(
        [
            helper.make_node("RMSNormalization", ["X", "Scale"], ["N"], epsilon=1e-5),
            helper.make_node("Transpose", ["N"], ["KT"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["N", "KT"], ["S"]),
            helper.make_node("Softmax", ["S"], ["P"], axis=-1),
            helper.make_node("MatMul", ["P", "N"], ["C"]),
            helper.make_node("Add", ["X", "C"], ["R"]),
            helper.make_node("RMSNormalization", ["R", "Scale"], ["N2"], epsilon=1e-5),
            helper.make_node("MatMul", ["N2", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["A"]),
            helper.make_node("MatMul", ["A", "W2"], ["Y0"]),
            helper.make_node("Add", ["R", "Y0"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])],
        [
            numpy_helper.from_array(scale, "Scale"),
            numpy_helper.from_array(w1, "W1"),
            numpy_helper.from_array(w2, "W2"),
        ],
        opset=23,
    )
    n = device_rms(x, scale)
    scores = n @ n.transpose(0, 2, 1)
    context = stable_softmax(scores) @ n
    residual = x + context
    n2 = device_rms(residual, scale)
    want = residual + np.maximum(n2 @ w1, 0) @ w2
    cpu = cpu_session(proto).run(None, {"X": x})[0]
    _, got = run_once(lib, session, proto, {"X": x})
    check("prenorm numpy", got, want)
    check("prenorm cpu", got, cpu)
    # The closing residual is on the device, so the add count is 2.
    # The attention norm still has two consumers and stays on the CPU.
    require(
        lib,
        "prenorm",
        matmul=4,
        gemm=0,
        kernels=4,
        relu=1,
        gelu=0,
        add=2,
        transposes=0,
        to_device=576,
        to_host=192,
    )


def test_ir(lib, session):
    if onnx.IR_VERSION < 4 or onnx.IR_VERSION == 3:
        raise SystemExit(f"onnx IR_VERSION {onnx.IR_VERSION}")
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    w_init = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.float32)
    constant = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(w_init, "W")],
        ir_version=onnx.IR_VERSION,
    )
    if constant.ir_version < 4 or constant.ir_version == 3:
        raise SystemExit(f"model ir_version {constant.ir_version}")
    lib.AccelEpResetStats()
    held = session(constant)
    require(lib, "ir constant compile", matmul=0, gemm=0, to_device=48)
    lib.AccelEpResetStats()
    got = held.run(None, {"X": x})[0]
    check("ir constant", got, x @ w_init)
    require(lib, "ir constant run", matmul=1, kernels=1, gemm=0, relu=0, to_device=32, to_host=24)
    del held

    fed = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(w_init, "W")],
        ir_version=onnx.IR_VERSION,
    )
    w_feed = np.array(w_init, copy=True)
    if np.shares_memory(w_feed, w_init):
        raise SystemExit("w_feed aliases the initializer")
    lib.AccelEpResetStats()
    held = session(fed)
    require(lib, "ir feed compile", matmul=0, to_device=0, transposes=0)
    lib.AccelEpResetStats()
    got = held.run(None, {"X": x, "W": w_feed})[0]
    want = np.array([[7, 6, 5], [15, 14, 13]], dtype=np.float32)
    check("ir feed", got, want)
    require(lib, "ir feed", matmul=1, kernels=1, to_device=80, to_host=24)
    w_feed *= -1
    lib.AccelEpResetStats()
    got = held.run(None, {"X": x, "W": w_feed})[0]
    check("ir feed negated", got, x @ w_feed)
    if np.allclose(got, x @ w_init):
        raise SystemExit("ir feed run used the initializer")
    del held

    b_init = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    fused = model(
        [helper.make_node("FusedMatMul", ["X", "B"], ["Y"], domain="com.microsoft", transB=1)],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(b_init, "B")],
        extra_opsets=(MICROSOFT,),
        ir_version=onnx.IR_VERSION,
    )
    b_feed = np.array(b_init, copy=True)
    lib.AccelEpResetStats()
    held = session(fused)
    require(lib, "ir transB compile", to_device=0, transposes=0, matmul=0)
    lib.AccelEpResetStats()
    got = held.run(None, {"X": x, "B": b_feed})[0]
    check("ir transB", got, x @ b_feed.T)
    require(lib, "ir transB run", matmul=1, kernels=1, transposes=1)
    b_feed *= -1
    got = held.run(None, {"X": x, "B": b_feed})[0]
    check("ir transB negated", got, x @ b_feed.T)
    require(lib, "ir transB second", transposes=2)
    del held
    gc.collect()


def test_live(lib, session):
    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((2, 8)).astype(np.float32)
    x1 = rng.standard_normal((2, 8)).astype(np.float32)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    proto = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(w, "W")],
    )
    baseline = lib.AccelEpLiveDeviceAllocs()
    held = session(proto)
    if lib.AccelEpLiveDeviceAllocs() != baseline + 1:
        raise SystemExit(f"live after construct {lib.AccelEpLiveDeviceAllocs()} baseline {baseline}")
    touches = lib.AccelEpConstantTouchCount()
    live = lib.AccelEpLiveDeviceAllocs()
    lib.AccelEpResetStats()
    if lib.AccelEpConstantTouchCount() != touches or lib.AccelEpLiveDeviceAllocs() != live:
        raise SystemExit("reset cleared touches or live allocs")
    lib.AccelEpResetStats()
    got = held.run(None, {"X": x0})[0]
    check("live x0", got, x0 @ w)
    require(lib, "live x0", matmul=1, kernels=1, to_device=64, transposes=0)
    if lib.AccelEpConstantTouchCount() != touches + 1 or lib.AccelEpLiveDeviceAllocs() != baseline + 1:
        raise SystemExit("touch or live changed on the first run")
    touch_id = lib.AccelEpConstantTouchId(touches)
    touch_address = lib.AccelEpConstantTouchAddress(touches)
    touch_bytes = lib.AccelEpConstantTouchBytes(touches)
    if touch_id == 0 or touch_address == 0 or touch_bytes != 128:
        raise SystemExit(f"touch id {touch_id} address {touch_address} bytes {touch_bytes}")
    lib.AccelEpResetStats()
    if lib.AccelEpConstantTouchCount() != touches + 1 or lib.AccelEpLiveDeviceAllocs() != baseline + 1:
        raise SystemExit("reset cleared the touch log")
    got = held.run(None, {"X": x1})[0]
    check("live x1", got, x1 @ w)
    require(lib, "live x1", to_device=64, kernels=1, transposes=0)
    if lib.AccelEpConstantTouchCount() != touches + 2:
        raise SystemExit("second run did not record a touch")
    if lib.AccelEpConstantTouchId(touches + 1) != touch_id or lib.AccelEpConstantTouchAddress(touches + 1) != touch_address:
        raise SystemExit("second run touched a different weight")
    if lib.AccelEpConstantTouchBytes(touches + 1) != 128 or lib.AccelEpLiveDeviceAllocs() != baseline + 1:
        raise SystemExit("second touch bytes or live alloc changed")
    lib.AccelEpClearConstantTouches()
    if lib.AccelEpConstantTouchCount() != 0 or lib.AccelEpLiveDeviceAllocs() != baseline + 1:
        raise SystemExit("clearing touches changed live allocs")
    if lib.AccelEpConstantTouchId(0) != 0 or lib.AccelEpConstantTouchAddress(-1) != 0 or lib.AccelEpConstantTouchBytes(5) != 0:
        raise SystemExit("out-of-range touch did not return 0")
    del held
    gc.collect()
    if lib.AccelEpLiveDeviceAllocs() != baseline:
        raise SystemExit(f"live after free {lib.AccelEpLiveDeviceAllocs()} baseline {baseline}")
    later = session(proto)
    lib.AccelEpResetStats()
    later.run(None, {"X": x0})
    if lib.AccelEpConstantTouchId(0) == 0 or lib.AccelEpConstantTouchId(0) == touch_id:
        raise SystemExit("later session reused a device allocation id")
    del later
    gc.collect()


def test_softmax(lib, session):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w1 = rng.standard_normal((8, 16)).astype(np.float32)
    w2 = rng.standard_normal((16, 4)).astype(np.float32)
    proto = model(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"]),
            helper.make_node("Softmax", ["H"], ["P"], axis=-1),
            helper.make_node("MatMul", ["P", "W2"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(w2, "W2")],
    )
    want = stable_softmax(x @ w1) @ w2
    cpu = cpu_session(proto).run(None, {"X": x})[0]
    options = cpu_options(profile=True)
    # Profile the EP session, not the CPU reference.
    del options
    sess = session(proto, profile=True)
    lib.AccelEpResetStats()
    got = sess.run(None, {"X": x})[0]
    trace = sess.end_profiling()
    check("softmax chain numpy", got, want)
    check("softmax chain cpu", got, cpu)
    require(
        lib,
        "softmax chain",
        matmul=2,
        gemm=0,
        kernels=2,
        relu=0,
        gelu=0,
        add=0,
        transposes=0,
        to_device=192,
        to_host=160,
    )
    events = kernel_events(trace)
    softmax = [
        event
        for event in events
        if (event.get("args") or {}).get("op_name") == "Softmax"
        and (event.get("args") or {}).get("provider") == "CPUExecutionProvider"
    ]
    accel = [
        event
        for event in events
        if (event.get("args") or {}).get("provider") == EP_NAME
        or str((event.get("args") or {}).get("op_name", "")).startswith(EP_NAME)
    ]
    stolen = [
        event
        for event in events
        if (event.get("args") or {}).get("op_name") == "Softmax" and EP_NAME in str((event.get("args") or {}).get("provider", ""))
    ]
    if len(softmax) != 1 or len(accel) != 2 or stolen:
        summary = [
            ((event.get("args") or {}).get("op_name"), (event.get("args") or {}).get("provider"), event.get("name"))
            for event in events
        ]
        raise SystemExit(f"softmax trace softmax={len(softmax)} accel={len(accel)} stolen={len(stolen)} {summary}")
    del sess

    lone_x = rng.standard_normal((2, 16)).astype(np.float32)
    lone = model(
        [helper.make_node("Softmax", ["X"], ["Y"], axis=-1)],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 16])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 16])],
        [],
    )
    sess = session(lone, profile=True)
    lib.AccelEpResetStats()
    got = sess.run(None, {"X": lone_x})[0]
    trace = sess.end_profiling()
    check("lone softmax", got, stable_softmax(lone_x))
    require_zero(lib, "lone softmax")
    events = kernel_events(trace)
    softmax = [
        event
        for event in events
        if (event.get("args") or {}).get("op_name") == "Softmax"
        and (event.get("args") or {}).get("provider") == "CPUExecutionProvider"
    ]
    accel = [
        event
        for event in events
        if (event.get("args") or {}).get("provider") == EP_NAME
        or str((event.get("args") or {}).get("op_name", "")).startswith(EP_NAME)
    ]
    if len(softmax) != 1 or accel:
        raise SystemExit(f"lone softmax trace softmax={len(softmax)} accel={len(accel)}")


def test_nonfinite(lib, session):
    x = np.array([[1]], dtype=np.float32)
    w = np.array([[1]], dtype=np.float32)
    for alpha in (float("nan"), float("inf"), float("-inf")):
        gemm = model(
            [helper.make_node("Gemm", ["X", "W"], ["Y"], alpha=alpha)],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
            [numpy_helper.from_array(w, "W")],
        )
        fused = model(
            [helper.make_node("FusedMatMul", ["X", "W"], ["Y"], domain="com.microsoft", alpha=alpha)],
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
            [numpy_helper.from_array(w, "W")],
            extra_opsets=(MICROSOFT,),
        )
        for label, proto in ((f"gemm alpha {alpha}", gemm), (f"fused alpha {alpha}", fused)):
            lib.AccelEpResetStats()
            cpu = cpu_session(proto).run(None, {"X": x})[0]
            got = session(proto).run(None, {"X": x})[0]
            if not np.array_equal(bits_of(got), bits_of(cpu)):
                raise SystemExit(f"{label} bits {bits_of(got)} cpu {bits_of(cpu)}")
            require_zero(lib, label)

    control = model(
        [helper.make_node("Gemm", ["X", "W"], ["Y"], alpha=2.0)],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(w, "W")],
    )
    _, got = run_once(lib, session, control, {"X": x})
    if int(bits_of(got).reshape(-1)[0]) != 0x40000000:
        raise SystemExit(f"gemm alpha 2 bits {bits_of(got)}")
    require(lib, "gemm alpha 2", gemm=1, matmul=0, kernels=1, relu=0)

    fused_control = model(
        [helper.make_node("FusedMatMul", ["X", "W"], ["Y"], domain="com.microsoft", alpha=2.0)],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(w, "W")],
        extra_opsets=(MICROSOFT,),
    )
    _, got = run_once(lib, session, fused_control, {"X": x})
    if int(bits_of(got).reshape(-1)[0]) != 0x40000000:
        raise SystemExit(f"fused alpha 2 bits {bits_of(got)}")
    require(lib, "fused alpha 2", matmul=1, gemm=0, kernels=1)

    inf = np.array([[np.inf]], dtype=np.float32)
    zero = np.array([[0]], dtype=np.float32)
    one = np.array([[1]], dtype=np.float32)
    nan_relu = model(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Relu", ["H"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(zero, "W")],
    )
    _, got = run_once(lib, session, nan_relu, {"X": inf})
    nan_bits = int(bits_of(got).reshape(-1)[0])
    if (nan_bits >> 23) & 0xFF != 0xFF or (nan_bits & 0x7FFFFF) == 0:
        raise SystemExit(f"inf*0 relu bits {nan_bits:#x}")
    require(lib, "inf zero relu", matmul=1, kernels=1, relu=1, gelu=0)

    pos = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(one, "W")],
    )
    _, got = run_once(lib, session, pos, {"X": inf})
    if int(bits_of(got).reshape(-1)[0]) != 0x7F800000:
        raise SystemExit(f"inf bits {bits_of(got)}")

    neg = model(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"]),
            helper.make_node("Relu", ["H"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(one, "W")],
    )
    _, got = run_once(lib, session, neg, {"X": np.array([[-np.inf]], dtype=np.float32)})
    if int(bits_of(got).reshape(-1)[0]) != 0x00000000:
        raise SystemExit(f"-inf relu bits {bits_of(got)}")
    require(lib, "-inf relu", relu=1, matmul=1)

    signed = model(
        [
            helper.make_node("Gemm", ["A", "B"], ["H"], alpha=-1.0),
            helper.make_node("Relu", ["H"], ["Y"]),
        ],
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(one, "B")],
    )
    _, got = run_once(lib, session, signed, {"A": np.array([[0]], dtype=np.float32)})
    if int(bits_of(got).reshape(-1)[0]) != 0x80000000:
        raise SystemExit(f"signed zero bits {int(bits_of(got).reshape(-1)[0]):#x}")
    require(lib, "signed zero", gemm=1, matmul=0, relu=1, kernels=1)


def decoder_nodes(head_name, flat_name):
    return [
        helper.make_node("LayerNormalization", ["X", "Scale"], ["N1"], epsilon=1e-5),
        helper.make_node("MatMul", ["N1", "Wq"], ["Q"]),
        helper.make_node("MatMul", ["N1", "Wk"], ["K"]),
        helper.make_node("MatMul", ["N1", "Wv"], ["V"]),
        helper.make_node("Reshape", ["Q", head_name], ["Qr"]),
        helper.make_node("Reshape", ["K", head_name], ["Kr"]),
        helper.make_node("Reshape", ["V", head_name], ["Vr"]),
        helper.make_node("Transpose", ["Qr"], ["Qh"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["Kr"], ["Kh"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["Vr"], ["Vh"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["Kh"], ["Kt"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["Qh", "Kt"], ["Scores"]),
        helper.make_node("Softmax", ["Scores"], ["P"], axis=-1),
        helper.make_node("MatMul", ["P", "Vh"], ["Ctx"]),
        helper.make_node("Transpose", ["Ctx"], ["CtxT"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["CtxT", flat_name], ["Merged"]),
        helper.make_node("MatMul", ["Merged", "Wo"], ["Proj"]),
        helper.make_node("Add", ["X", "Proj"], ["R"]),
        helper.make_node("LayerNormalization", ["R", "Scale"], ["N2"], epsilon=1e-5),
        helper.make_node("MatMul", ["N2", "W1"], ["H"]),
        helper.make_node("Relu", ["H"], ["A"]),
        helper.make_node("MatMul", ["A", "W2"], ["Y0"]),
        helper.make_node("Add", ["R", "Y0"], ["Y"]),
    ]


def decoder_numpy(x, scale, wq, wk, wv, wo, w1, w2):
    n1 = layer_norm(x, scale)

    def heads(value):
        return value.reshape(1, 4, 2, 4).transpose(0, 2, 1, 3)

    qh = heads(n1 @ wq)
    kh = heads(n1 @ wk)
    vh = heads(n1 @ wv)
    scores = qh @ kh.transpose(0, 1, 3, 2)
    ctx = stable_softmax(scores) @ vh
    merged = ctx.transpose(0, 2, 1, 3).reshape(1, 4, 8)
    residual = x + merged @ wo
    n2 = layer_norm(residual, scale)
    return residual + np.maximum(n2 @ w1, 0) @ w2


def test_decoder(lib, session):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 4, 8)).astype(np.float32)
    scale = np.ones((8,), dtype=np.float32)
    weights = [rng.standard_normal(shape).astype(np.float32) for shape in ((8, 8), (8, 8), (8, 8), (8, 8), (8, 16), (16, 8))]
    wq, wk, wv, wo, w1, w2 = weights
    initializers = [
        numpy_helper.from_array(scale, "Scale"),
        numpy_helper.from_array(wq, "Wq"),
        numpy_helper.from_array(wk, "Wk"),
        numpy_helper.from_array(wv, "Wv"),
        numpy_helper.from_array(wo, "Wo"),
        numpy_helper.from_array(w1, "W1"),
        numpy_helper.from_array(w2, "W2"),
        numpy_helper.from_array(np.array([1, 4, 2, 4], dtype=np.int64), "HeadShape"),
        numpy_helper.from_array(np.array([1, 4, 8], dtype=np.int64), "FlatShape"),
    ]
    proto = model(
        decoder_nodes("HeadShape", "FlatShape"),
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])],
        initializers,
    )
    want = decoder_numpy(x, scale, wq, wk, wv, wo, w1, w2)
    cpu = cpu_session(proto).run(None, {"X": x})[0]
    _, got = run_once(lib, session, proto, {"X": x})
    check("decoder numpy", got, want)
    check("decoder cpu", got, cpu)
    require(
        lib,
        "decoder",
        matmul=8,
        gemm=0,
        kernels=10,
        relu=1,
        gelu=0,
        add=0,
        transposes=0,
        to_device=1152,
        to_host=896,
    )

    symbolic_inits = [
        numpy_helper.from_array(scale, "Scale"),
        numpy_helper.from_array(wq, "Wq"),
        numpy_helper.from_array(wk, "Wk"),
        numpy_helper.from_array(wv, "Wv"),
        numpy_helper.from_array(wo, "Wo"),
        numpy_helper.from_array(w1, "W1"),
        numpy_helper.from_array(w2, "W2"),
        numpy_helper.from_array(np.array([1, -1, 2, 4], dtype=np.int64), "HeadShape"),
        numpy_helper.from_array(np.array([1, -1, 8], dtype=np.int64), "FlatShape"),
    ]
    symbolic = model(
        decoder_nodes("HeadShape", "FlatShape"),
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "seq", 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "seq", 8])],
        symbolic_inits,
    )
    lib.AccelEpResetStats()
    try:
        cpu = cpu_session(symbolic).run(None, {"X": x})[0]
        got = session(symbolic).run(None, {"X": x})[0]
    except Exception as ex:
        if "AccelEP" in str(ex) or "dynamic dimension" in str(ex):
            raise SystemExit(f"symbolic decoder {ex}") from ex
        raise
    check("symbolic decoder", got, cpu)
    require_zero(lib, "symbolic decoder")


def test_pretranspose(lib, session):
    rng = np.random.default_rng(8)
    fx = rng.standard_normal((2, 4)).astype(np.float32)
    fw = rng.standard_normal((3, 4)).astype(np.float32)
    proto = model(
        [helper.make_node("FusedMatMul", ["X", "W"], ["Y"], domain="com.microsoft", transB=1, alpha=1.5)],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(fw, "W")],
        extra_opsets=(MICROSOFT,),
    )
    before = lib.AccelEpLiveDeviceAllocs()
    held = session(proto)
    if lib.AccelEpLiveDeviceAllocs() != before + 2:
        raise SystemExit(f"pretranspose live {lib.AccelEpLiveDeviceAllocs()} before {before}")
    lib.AccelEpResetStats()
    want = np.float32(1.5) * (fx @ fw.T)
    first = held.run(None, {"X": fx})[0]
    second = held.run(None, {"X": fx})[0]
    check("pretranspose first", first, want)
    check("pretranspose second", second, want)
    require(lib, "pretranspose two runs", matmul=2, kernels=2, gemm=0, transposes=0, to_device=int(2 * fx.nbytes))
    del held
    gc.collect()
    if lib.AccelEpLiveDeviceAllocs() != before:
        raise SystemExit(f"pretranspose live after free {lib.AccelEpLiveDeviceAllocs()} before {before}")

    plain_w = rng.standard_normal((4, 3)).astype(np.float32)
    plain = model(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(plain_w, "W")],
    )
    before = lib.AccelEpLiveDeviceAllocs()
    held = session(plain)
    if lib.AccelEpLiveDeviceAllocs() != before + 1:
        raise SystemExit("plain matmul allocated a transpose buffer")
    lib.AccelEpResetStats()
    held.run(None, {"X": fx})
    require(lib, "plain matmul transpose", matmul=1, kernels=1, transposes=0)
    del held
    gc.collect()

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    falsifier = model(
        [helper.make_node("Gemm", ["X", "B"], ["Y"], transB=1)],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        [],
    )
    before = lib.AccelEpLiveDeviceAllocs()
    held = session(falsifier)
    if lib.AccelEpLiveDeviceAllocs() != before:
        raise SystemExit("graph-input Gemm allocated at compile")
    lib.AccelEpResetStats()
    got = held.run(None, {"X": a, "B": b})[0]
    check("runtime transB", got, a @ b.T)
    require(lib, "runtime transB", gemm=1, matmul=0, kernels=1, transposes=1)
    got = held.run(None, {"X": a, "B": b})[0]
    check("runtime transB again", got, a @ b.T)
    require(lib, "runtime transB again", transposes=2, gemm=2)
    del held
    gc.collect()


def test_runtime_weight(lib, session):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w1 = rng.standard_normal((8, 16)).astype(np.float32)
    w2 = rng.standard_normal((16, 4)).astype(np.float32)
    proto = model(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"]),
            helper.make_node("Relu", ["H"], ["R"]),
            helper.make_node("MatMul", ["R", "W2"], ["Y"]),
        ],
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8]),
            helper.make_tensor_value_info("W2", TensorProto.FLOAT, [16, 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(w1, "W1")],
    )
    _, got = run_once(lib, session, proto, {"X": x, "W2": w2})
    check("runtime w2", got, np.maximum(x @ w1, 0) @ w2)
    require(lib, "runtime w2", matmul=2, kernels=2, relu=1, gemm=0, gelu=0, add=0, to_device=448, to_host=160)


def test_saved_chain(lib, session, prepared):
    _, got = run_once(lib, session, prepared["extended"], {"X": prepared["x"]})
    check("saved chain", got, prepared["want"])
    require(
        lib,
        "saved chain",
        matmul=2,
        gemm=0,
        kernels=2,
        relu=1,
        gelu=0,
        add=0,
        transposes=1,
        to_device=prepared["x"].nbytes,
        to_host=prepared["want"].nbytes,
    )


def run_frozen(lib, session, prepared):
    if lib.AccelEpIsStreamAware() != 0:
        raise SystemExit(f"stream flag returned {lib.AccelEpIsStreamAware()}")
    test_saved_chain(lib, session, prepared)
    test_runtime_weight(lib, session)
    profile_chain(lib, session)
    test_float16(lib, session)
    test_dequantize(lib, session)
    test_leading(lib, session)
    test_residual(lib, session)
    test_gelu(lib, session)
    test_two_sessions(lib, session)
    test_prenorm(lib, session)
    test_ir(lib, session)
    test_pretranspose(lib, session)
    test_live(lib, session)
    test_softmax(lib, session)
    test_nonfinite(lib, session)
    test_decoder(lib, session)


def main():
    lib = ctypes.CDLL(str(LIB))
    bind(lib)

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
    prepared = prepare_chain()

    ort.register_execution_provider_library(EP_NAME, str(LIB))
    try:
        devices = [device for device in ort.get_ep_devices() if device.ep_name == EP_NAME]
        if len(devices) != 1:
            raise SystemExit(f"expected one AccelEP device, found {[d.ep_name for d in ort.get_ep_devices()]}")

        def session(proto, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL, profile=False):
            options = ort.SessionOptions()
            if level is not None:
                options.graph_optimization_level = level
            if profile:
                options.enable_profiling = True
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
        # The chain keeps the Relu activation on the device. Only X is copied in and Y is copied out.
        expect_copy = x.nbytes
        expect_back = matmul_want.nbytes
        if (
            matmul_ops != 2
            or gemm_ops != 0
            or kernel_matmuls != 2
            or relu_epilogues != 1
            or lib.AccelEpGeluEpilogues() != 0
            or lib.AccelEpAddEpilogues() != 0
            or copied != expect_copy
            or copied_back != expect_back
        ):
            raise SystemExit(
                "matmul offload stats "
                f"ops={matmul_ops} gemm={gemm_ops} kernels={kernel_matmuls} "
                f"relu_epilogues={relu_epilogues} gelu={lib.AccelEpGeluEpilogues()} "
                f"add={lib.AccelEpAddEpilogues()} copied={copied} expected_copy={expect_copy} "
                f"copied_back={copied_back} expected_back={expect_back}"
            )
        lib.AccelEpResetStats()
        check("accel matmul+relu again", matmul_session.run(None, {"X": x})[0], matmul_want)
        if lib.AccelEpBytesToDevice() != x.nbytes or lib.AccelEpBytesToHost() != matmul_want.nbytes:
            raise SystemExit(
                f"second chain run copied {lib.AccelEpBytesToDevice()}/{lib.AccelEpBytesToHost()}, "
                f"expected {x.nbytes}/{matmul_want.nbytes}"
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
        if lib.AccelEpReluEpilogues() != 0 or lib.AccelEpAddEpilogues() != 0 or lib.AccelEpDeviceTransposes() != 0:
            raise SystemExit(
                "Add was folded into the Gemm epilogue "
                f"relu={lib.AccelEpReluEpilogues()} add={lib.AccelEpAddEpilogues()} "
                f"transposes={lib.AccelEpDeviceTransposes()}"
            )

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
        if (
            lib.AccelEpMatmulOps() != 1
            or lib.AccelEpKernelMatmuls() != 1
            or lib.AccelEpBytesToDevice() != fx.nbytes
            or lib.AccelEpDeviceTransposes() != 0
        ):
            raise SystemExit(
                "explicit fused matmul stats "
                f"ops={lib.AccelEpMatmulOps()} kernels={lib.AccelEpKernelMatmuls()} "
                f"copied={lib.AccelEpBytesToDevice()} expected={fx.nbytes} "
                f"transposes={lib.AccelEpDeviceTransposes()}"
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

        run_frozen(lib, session, prepared)
    finally:
        ort.unregister_execution_provider_library(EP_NAME)

    print("ep tests passed")
    print("matmul ops on accelerator: 2, relu epilogue kept the hidden activation on device")
    print("gemm op on accelerator: 1, following add left on CPU")
    print("fused matmul on accelerator: 1, copied only the activation")


if __name__ == "__main__":
    main()
