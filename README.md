# AccelEP

AccelEP is an [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.31 execution provider for matrix ops. A CPU simulator stands in for the accelerator, so the plugin runs without a GPU. A hardware backend replaces three kernels on the same device interface.

The provider claims one supported node at a time. Relu, Add, and every other op stay on the CPU execution provider.

## Ops that run on the accelerator

All three ops require float32 and static shapes. A negative dimension stays on the CPU.

| Op | Domain | What is accepted |
| --- | --- | --- |
| MatMul | `ai.onnx` | Rank 2 or higher. Batch dimensions use NumPy-style broadcasting. |
| Gemm | `ai.onnx` | Strictly 2D. `transA` and `transB` are 0 or 1. `alpha` and `beta` are honored. An optional bias must broadcast onto the `[M, N]` result. |
| FusedMatMul | `com.microsoft` | Rank 2 or higher. `transA` and `transB` swap the last two dimensions and must be 0 or 1. `alpha` scales the product. `transBatchA` and `transBatchB` must stay 0. |

`com.microsoft.FusedMatMul` is MatMul plus those transpose and scale attributes. A nonzero `transBatch` moves axis 0 into the matrix-row slot, and that layout stays on the CPU.

## CPU simulator

`SimDevice` implements `accel::Device` in `include/accel/device.h`:

- `Allocate` and `Free` use a heap that belongs to the device.
- `CopyHostToDevice` and `CopyDeviceToHost` are the only way bytes cross that boundary.
- `MatMul` computes row-major `C = A @ B`.
- `Transpose` turns a `[rows, cols]` matrix into `[cols, rows]`.
- `GemmEpilogue` applies `c = alpha * c + beta * broadcast(bias)`.

Constant weights are uploaded once, when the node is compiled. Each run copies the activation inputs to the device and the result back to the host. The plugin does not take over ONNX Runtime's own host copies.

A real accelerator keeps this runtime and replaces `MatMul`, `Transpose`, and `GemmEpilogue`.

## Build and test

The shared library links with `-Wl,-z,defs`, so build it with GNU ld. Ubuntu 24.04 with `g++`, `make`, and `python3-venv` is enough.

```bash
make venv
make test
```

`make venv` installs `onnxruntime==1.31.0` and `onnx==1.21.0` into `/tmp/accel-venv`. `make test` builds `build/libaccel_ep.so`, runs the C++ kernel tests, then runs the Python session tests. A good run ends with:

```text
runtime tests passed
ep tests passed
```

`make clean` removes `build/`.

The C++ tests check hand-computed matmul, broadcast, the Gemm epilogue, FusedMatMul transpose and alpha, and that a weight uploaded at compile time is not copied again. The Python tests load the plugin with `register_execution_provider_library`, run `X @ W1 → Relu → @ W2`, a Gemm plus a later Add, an authored FusedMatMul, and a FusedMatMul followed by Relu. Results are compared with NumPy. Those graphs are loaded with `ORT_DISABLE_ALL` so the default optimizer does not rewrite a MatMul before this provider sees it. With the default optimizer left on, a Transpose followed by a MatMul still counts as one accelerator op, and an authored FusedMatMul stays on this provider.

## Registering the provider

```python
import onnxruntime as ort

ort.register_execution_provider_library("AccelEP", "build/libaccel_ep.so")
devices = [device for device in ort.get_ep_devices() if device.ep_name == "AccelEP"]

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.add_provider_for_devices(devices, {})
session = ort.InferenceSession(model_bytes, options)
```

## Layout

```text
include/accel/     Device contract and the runtime used by every backend
src/sim_device.cc  CPU simulator
src/runtime.cc     Shape checks and MatMul, Gemm, and FusedMatMul execution
src/ep.cc          ONNX Runtime plugin
tests/             C++ kernel tests and Python session tests
third_party/       ONNX Runtime 1.31.0 public C/C++ headers
```

The headers under `third_party/onnxruntime` are the public C/C++ API from ONNX Runtime release v1.31.0. Microsoft owns that code and ships it under the MIT License. See `third_party/onnxruntime/NOTICE`.
