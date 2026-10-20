# AccelEP

AccelEP is an ONNX Runtime 1.31 plugin for matrix multiplies. It does not need a GPU. `SimDevice` is the stand-in for the accelerator: a separate heap, and kernels you call by name. A hardware backend can replace those kernels and keep the same interface.

The same simulator can serve a small Llama. The checked model is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). A full-size Llama does not fit in 8 GB of RAM.

## Serve SmolLM2

Put these files in `models/`. They are not part of the git tree.

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

```bash
make simlib
python3 -m venv .venv
.venv/bin/pip install numpy tokenizers
PYTHONPATH=. .venv/bin/python -m serve --model models --port 8000
```

The server listens on `127.0.0.1:8000`.

| Route | What it does |
| --- | --- |
| `GET /health` | Reports that the server is up. |
| `POST /v1/completions` | Completes a prompt. |
| `POST /v1/chat/completions` | Completes a chat. |

The host process writes the chat string and answers HTTP. A generate call then runs on the simulator:

1. The prompt token ids are copied onto the device.
2. Each forward step runs embedding, RMSNorm, RoPE, attention, SiLU, the residual adds, and the KV-cache update.
3. Every matrix multiply runs there too, including the vocabulary projection.
4. The simulator picks the next token and writes that id itself.

The logit row stays on the device. The host receives the completion text and the chosen token ids.

Text encode and decode run in this library. They are not device kernels. `SimDevice` itself executes on the machine's CPU.

## Simulator tests

`make sim-test` builds the simulator and runs 37 tests. It needs the `models/` files above and a `.venv` with `numpy` and `tokenizers`. A good run prints `kernel tests passed`, `runtime tests passed`, and `serve tests passed`.

`tests/test_sim_kernels.cc` has 8 tests. Each one calls a kernel directly.

- Embed gathers two rows and rejects an id past the table.
- RMSNorm matches the double-precision reference, including a row offset.
- RoPE leaves position 0 unchanged and rotates a later position.
- The KV-cache append writes only the new rows.
- Attention, SiLU, and the residual add match hand values.
- Sampling at temperature 0 is an argmax. A tie keeps the earlier id.

`tests/test_runtime.cc` has 24 tests.

- Known MatMul, an identity batch, broadcast hand values, and Gemm with a transposed bias.
- A weight that stays on the device, a zero K dimension, rejected shapes, and freeing an unknown buffer.
- Fused MatMul with `transA` and alpha, broadcast with `transB`, and a Relu that stays on the device.
- A Gemm Relu epilogue, a rejected fused shape, and the packed GEMM counters.
- Negative shape rejection, split product bits, and non-finite epilogue bits.
- Residual orientation, a pre-transpose of the last two axes, and the KV table.
- Scoped stats, live allocation ids, the recording device, and a private host copy.

`tests/test_serve.py` has 5 tests. The forward and HTTP tests load SmolLM2.

- Encode and decode match the `tokenizers` package on 24 strings, including bytes missing from the SmolLM2 vocabulary. The ids `[504, 3575, 282, 4649, 314, 7042, 30, 2]` decode to the same text.
- One simulator matrix multiply matches NumPy.
- The Python RMSNorm and RoPE helpers match their formulas.
- Greedy chat for "The capital of France is" returns text containing "The capital of France is Paris." The host copy is the UTF-8 reply plus 4 bytes per completion id. The device copy is 4 bytes per prompt id. A one-token sample at temperature 0.8 uses the same byte rule.
- `GET /health`, `POST /v1/chat/completions`, and `POST /v1/completions` return 200. The chat reply contains Paris.

## ONNX plugin

The plugin claims float32 MatMul, 2D Gemm, and `com.microsoft.FusedMatMul` when the shapes are static. One leading MatMul batch dimension may still be unknown when the graph is compiled. Softmax, layer norm, and every other op stay on the CPU execution provider.

A following Relu stays on the device. So does an opset-20 Gelu when `approximate` is absent or `"none"`. The next matrix in that group reads the activated values without a host copy. A lone MatMul can add one residual of the same shape. A constant-weight projection leaves that residual on the CPU when the Add feeds more than one node.

A constant int8 `DequantizeLinear` that feeds a float MatMul is uploaded as float32. Other quantizer shapes stay on the CPU.

| Op | Domain | What is accepted |
| --- | --- | --- |
| MatMul | `ai.onnx` | Rank 2 or higher. Batch dimensions broadcast like NumPy. |
| Gemm | `ai.onnx` | Strictly 2D. `transA` and `transB` are 0 or 1. An optional bias must broadcast onto `[M, N]`. |
| FusedMatMul | `com.microsoft` | Rank 2 or higher. `transA` and `transB` are 0 or 1. `transBatchA` and `transBatchB` stay 0. |

`MatMul` computes row-major `C = A @ B`. Each product is a multiply followed by an add. It is float32 only.

Constant weights are uploaded once, when the node is compiled. Each run copies the activations that are not already on the device, and copies that group's result back.

## Build and test the plugin

The plugin shared library uses `-Wl,-z,defs`, so build that library with GNU ld. Ubuntu 24.04 with `g++`, `make`, and `python3-venv` is enough.

```bash
make venv
make test
```

`make venv` installs `onnxruntime==1.31.0` and `onnx==1.21.0` into `/tmp/accel-venv`. A good run prints `runtime tests passed` and `ep tests passed`. `make bench` is separate. `make clean` removes `build/`.

On macOS, `make simlib` builds `build/libaccel_sim.dylib` for the Llama server. Do not reuse those Mach-O objects for a Linux plugin build. Run `make clean` first.

## Register the plugin

```python
import onnxruntime as ort

ort.register_execution_provider_library("AccelEP", "build/libaccel_ep.so")
devices = [device for device in ort.get_ep_devices() if device.ep_name == "AccelEP"]

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.add_provider_for_devices(devices, {})
session = ort.InferenceSession(model_bytes, options)
```

## Limits

The ONNX decoder-block test has no paged KV cache, continuous batching, CUDA, tokens/s figure, causal mask, or RoPE. Its attention scores stay on the CPU. AccelEP does not match or beat vLLM. The CPU simulator can be slower than ONNX Runtime's own CPU kernels.

## Layout

```text
include/accel/      Device interface and kernel headers
src/sim_device.cc   CPU simulator
src/sim_*.cc        Embedding, RMSNorm, RoPE, cache, attention, SiLU, add, sample
src/llama_model.cc  Llama forward. It only calls device kernels.
src/bpe.cc          Tokenizer encode and decode
src/ep.cc           ONNX Runtime plugin
serve/              HTTP server
tests/              C++ and Python tests
third_party/        ONNX Runtime 1.31.0 public headers
```

The headers under `third_party/onnxruntime` are the public C/C++ API from ONNX Runtime release v1.31.0. Microsoft owns that code and ships it under the MIT License. See `third_party/onnxruntime/NOTICE`.
