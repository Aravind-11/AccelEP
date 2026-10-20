"""ctypes binding to the CPU simulator. Weights stay on the device."""

import ctypes
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = (
    _ROOT / "build" / "libaccel_sim.dylib",
    _ROOT / "build" / "libaccel_sim.so",
)


def _load():
    for path in _CANDIDATES:
        if path.is_file():
            return ctypes.CDLL(str(path))
    raise FileNotFoundError("build libaccel_sim first: make simlib")


class Sim:
    def __init__(self):
        lib = _load()
        lib.accel_sim_create.restype = ctypes.c_void_p
        lib.accel_sim_destroy.argtypes = [ctypes.c_void_p]
        lib.accel_sim_upload.restype = ctypes.c_void_p
        lib.accel_sim_upload.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        lib.accel_sim_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.accel_sim_matmul.restype = ctypes.c_int
        lib.accel_sim_matmul.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.accel_sim_reset_stats.restype = None
        lib.accel_sim_matmuls.restype = ctypes.c_int64
        lib.accel_sim_bytes_to_device.restype = ctypes.c_int64
        lib.accel_sim_bytes_to_host.restype = ctypes.c_int64
        lib.accel_sim_last_error.restype = ctypes.c_char_p
        self._lib = lib
        self._sim = lib.accel_sim_create()
        if not self._sim:
            raise RuntimeError(self.error())

    def error(self):
        raw = self._lib.accel_sim_last_error()
        return raw.decode() if raw else ""

    def upload(self, values):
        array = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        ptr = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        buf = self._lib.accel_sim_upload(self._sim, ptr, array.size)
        if not buf:
            raise RuntimeError(self.error() or "upload failed")
        return buf

    def release(self, buf):
        self._lib.accel_sim_release(self._sim, buf)

    def matmul(self, left, weight, n):
        """left is float32 [m, k]. weight is an uploaded [k, n] matrix."""
        a = np.ascontiguousarray(left, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        m, k = a.shape
        out = np.empty((m, n), dtype=np.float32)
        status = self._lib.accel_sim_matmul(
            self._sim,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            k,
            weight,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status != 0:
            raise RuntimeError(self.error() or "matmul failed")
        return out

    def reset_stats(self):
        self._lib.accel_sim_reset_stats()

    def matmuls(self):
        return int(self._lib.accel_sim_matmuls())

    def bytes_to_device(self):
        return int(self._lib.accel_sim_bytes_to_device())

    def bytes_to_host(self):
        return int(self._lib.accel_sim_bytes_to_host())

    def close(self):
        if self._sim:
            self._lib.accel_sim_destroy(self._sim)
            self._sim = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
