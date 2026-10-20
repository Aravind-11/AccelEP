"""Llama generation. Tokenization, the forward pass, and the next-token choice run on the simulator."""

import ctypes
import json
from pathlib import Path

import numpy as np

from serve.sim import _load
from serve.weights import load_safetensors

ACCEL_W_EMBED = 1
ACCEL_W_FINAL_NORM = 2
ACCEL_W_LM_HEAD = 3
ACCEL_W_ATTN_NORM = 4
ACCEL_W_POST_NORM = 5
ACCEL_W_Q = 6
ACCEL_W_K = 7
ACCEL_W_V = 8
ACCEL_W_O = 9
ACCEL_W_GATE = 10
ACCEL_W_UP = 11
ACCEL_W_DOWN = 12


class _Config(ctypes.Structure):
    _fields_ = [
        ("vocab", ctypes.c_int32),
        ("hidden", ctypes.c_int32),
        ("layers", ctypes.c_int32),
        ("heads", ctypes.c_int32),
        ("kv_heads", ctypes.c_int32),
        ("intermediate", ctypes.c_int32),
        ("context", ctypes.c_int32),
        ("eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
    ]


def rmsnorm(x, weight, eps):
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return (x * np.reciprocal(np.sqrt(variance + eps))) * weight


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def rope_tables(head_dim, n_positions, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(n_positions, dtype=np.float64)
    freqs = np.outer(positions, inv)
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(states, cos, sin):
    """states is [seq, heads, dim]. cos and sin are [seq, dim]."""
    return states * cos[:, None, :] + rotate_half(states) * sin[:, None, :]


def _bind(lib):
    lib.accel_llama_create.restype = ctypes.c_void_p
    lib.accel_llama_create.argtypes = [ctypes.POINTER(_Config)]
    lib.accel_llama_destroy.argtypes = [ctypes.c_void_p]
    lib.accel_llama_weight.restype = ctypes.c_int
    lib.accel_llama_weight.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.accel_llama_tokenizer.restype = ctypes.c_int
    lib.accel_llama_tokenizer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.accel_llama_encode.restype = ctypes.c_int
    lib.accel_llama_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.accel_llama_decode.restype = ctypes.c_int
    lib.accel_llama_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.accel_llama_generate.restype = ctypes.c_int
    lib.accel_llama_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.accel_llama_complete.restype = ctypes.c_int
    lib.accel_llama_complete.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.accel_llama_reset.argtypes = [ctypes.c_void_p]
    lib.accel_llama_last_error.restype = ctypes.c_char_p
    lib.accel_sim_reset_stats.restype = None
    lib.accel_sim_matmuls.restype = ctypes.c_int64
    lib.accel_sim_bytes_to_device.restype = ctypes.c_int64
    lib.accel_sim_bytes_to_host.restype = ctypes.c_int64


class Llama:
    def __init__(self, model_dir, context=512, load_weights=True):
        model_dir = Path(model_dir)
        with open(model_dir / "config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("architectures") != ["LlamaForCausalLM"]:
            raise ValueError(f"expected LlamaForCausalLM, got {config.get('architectures')}")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("only silu MLP is implemented")
        if config.get("rope_scaling"):
            raise ValueError("rope scaling is not implemented")
        self.config = config
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.intermediate = int(config["intermediate_size"])
        self.head_dim = self.hidden // self.heads
        self.vocab = int(config["vocab_size"])
        self.eos_id = int(config["eos_token_id"])
        self.context = context
        native = _Config(
            self.vocab,
            self.hidden,
            self.layers,
            self.heads,
            self.kv_heads,
            self.intermediate,
            context,
            float(config["rms_norm_eps"]),
            float(config["rope_theta"]),
        )
        self._lib = _load()
        _bind(self._lib)
        self._model = self._lib.accel_llama_create(ctypes.byref(native))
        if not self._model:
            raise RuntimeError(self.error() or "create failed")
        tokenizer = model_dir / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError(f"missing tokenizer: {tokenizer}")
        self.load_tokenizer(tokenizer)
        if not load_weights:
            return
        weights = load_safetensors(model_dir / "model.safetensors")
        self._weight(ACCEL_W_EMBED, -1, weights["model.embed_tokens.weight"])
        self._weight(ACCEL_W_FINAL_NORM, -1, weights["model.norm.weight"])
        self._weight(ACCEL_W_LM_HEAD, -1, weights.get("lm_head.weight", weights["model.embed_tokens.weight"]))
        for index in range(self.layers):
            prefix = f"model.layers.{index}"
            self._weight(ACCEL_W_ATTN_NORM, index, weights[f"{prefix}.input_layernorm.weight"])
            self._weight(ACCEL_W_POST_NORM, index, weights[f"{prefix}.post_attention_layernorm.weight"])
            self._weight(ACCEL_W_Q, index, weights[f"{prefix}.self_attn.q_proj.weight"])
            self._weight(ACCEL_W_K, index, weights[f"{prefix}.self_attn.k_proj.weight"])
            self._weight(ACCEL_W_V, index, weights[f"{prefix}.self_attn.v_proj.weight"])
            self._weight(ACCEL_W_O, index, weights[f"{prefix}.self_attn.o_proj.weight"])
            self._weight(ACCEL_W_GATE, index, weights[f"{prefix}.mlp.gate_proj.weight"])
            self._weight(ACCEL_W_UP, index, weights[f"{prefix}.mlp.up_proj.weight"])
            self._weight(ACCEL_W_DOWN, index, weights[f"{prefix}.mlp.down_proj.weight"])

    def error(self):
        raw = self._lib.accel_llama_last_error()
        return raw.decode() if raw else ""

    def _weight(self, kind, layer, values):
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.ndim == 1:
            rows, cols = array.shape[0], 1
        else:
            rows, cols = array.shape
        status = self._lib.accel_llama_weight(
            self._model,
            kind,
            layer,
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            cols,
        )
        if status != 0:
            raise RuntimeError(self.error() or "weight upload failed")

    def reset(self):
        self._lib.accel_llama_reset(self._model)

    def reset_stats(self):
        self._lib.accel_sim_reset_stats()

    def matmuls(self):
        return int(self._lib.accel_sim_matmuls())

    def bytes_to_device(self):
        return int(self._lib.accel_sim_bytes_to_device())

    def bytes_to_host(self):
        return int(self._lib.accel_sim_bytes_to_host())

    def load_tokenizer(self, path):
        status = self._lib.accel_llama_tokenizer(self._model, str(path).encode())
        if status != 0:
            raise RuntimeError(self.error() or "tokenizer load failed")

    def encode(self, text):
        raw = text.encode("utf-8")
        cap = max(len(raw), 1)
        out = np.empty(cap, dtype=np.int32)
        count = ctypes.c_int32(0)
        # create_string_buffer keeps embedded NUL bytes. c_char_p alone would stop there.
        buf = ctypes.create_string_buffer(raw)
        status = self._lib.accel_llama_encode(
            self._model,
            buf,
            len(raw),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            cap,
            ctypes.byref(count),
        )
        if status != 0:
            raise RuntimeError(self.error() or "encode failed")
        return [int(token) for token in out[: count.value]]

    def decode(self, token_ids, skip_special=True):
        ids = np.ascontiguousarray(list(token_ids), dtype=np.int32)
        count = int(ids.size)
        if count == 0:
            ids = np.zeros(1, dtype=np.int32)
        cap = max(count, 1) * 1024
        buf = ctypes.create_string_buffer(cap)
        length = ctypes.c_int32(0)
        status = self._lib.accel_llama_decode(
            self._model,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            count,
            1 if skip_special else 0,
            buf,
            cap,
            ctypes.byref(length),
        )
        if status != 0:
            raise RuntimeError(self.error() or "decode failed")
        return buf.raw[: length.value].decode("utf-8")

    def complete(self, prompt, max_new, eos_id, temperature=0.0, seed=0):
        limit = int(max_new)
        if limit < 0:
            raise ValueError("max_new must be non-negative")
        text_cap = max(limit, 1) * 1024
        text_buf = ctypes.create_string_buffer(text_cap)
        text_len = ctypes.c_int32(0)
        ids = np.empty(max(limit, 1), dtype=np.int32)
        count = ctypes.c_int32(0)
        prompt_tokens = ctypes.c_int32(0)
        raw = prompt.encode("utf-8")
        prompt_buf = ctypes.create_string_buffer(raw)
        status = self._lib.accel_llama_complete(
            self._model,
            prompt_buf,
            len(raw),
            limit,
            int(eos_id),
            float(temperature),
            ctypes.c_uint32(int(seed) & 0xFFFFFFFF),
            text_buf,
            text_cap,
            ctypes.byref(text_len),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(ids.size),
            ctypes.byref(count),
            ctypes.byref(prompt_tokens),
        )
        if status != 0:
            raise RuntimeError(self.error() or "complete failed")
        text = text_buf.raw[: text_len.value].decode("utf-8")
        return text, [int(token) for token in ids[: count.value]], int(prompt_tokens.value)

    def generate(self, token_ids, max_new, eos_id, temperature=0.0, seed=0):
        ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        limit = int(max_new)
        if ids.size == 0:
            raise ValueError("prompt encoded to no tokens")
        if limit < 0:
            raise ValueError("max_new must be non-negative")
        out = np.empty(max(limit, 1), dtype=np.int32)
        count = ctypes.c_int32(0)
        status = self._lib.accel_llama_generate(
            self._model,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(ids.size),
            limit,
            int(eos_id),
            float(temperature),
            ctypes.c_uint32(int(seed) & 0xFFFFFFFF),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(count),
        )
        if status != 0:
            raise RuntimeError(self.error() or "generate failed")
        return [int(token) for token in out[: count.value]]

    def close(self):
        if self._model:
            self._lib.accel_llama_destroy(self._model)
            self._model = None
