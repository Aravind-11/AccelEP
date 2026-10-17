"""Python mirror of the fragment layouts in cuda/probe_mma.cu.

Used only to check, on CPU, that the layouts used by the kernel are internally
consistent: every A, B and C element is owned by exactly one (lane, register,
slot), and A and B agree on which k each slot holds. Whether the layouts match
the hardware is what `probe_mma <config> --selftest` checks on a GPU.
"""
from typing import Dict, Tuple


def a_layout_e4m3_k32() -> Dict[Tuple[int, int, int], Tuple[int, int]]:
    """(lane, reg, byte) -> (row, col) of A (16 x 32)."""
    m = {}
    for lane in range(32):
        gid, tig = lane >> 2, lane & 3
        for r in range(4):
            for j in range(4):
                m[(lane, r, j)] = (gid + (8 if r & 1 else 0), tig * 4 + j + (16 if r >= 2 else 0))
    return m


def b_layout_e4m3_k32():
    """(lane, reg, byte) -> (k, col) of B (32 x 8)."""
    m = {}
    for lane in range(32):
        gid, tig = lane >> 2, lane & 3
        for r in range(2):
            for j in range(4):
                m[(lane, r, j)] = (tig * 4 + j + (16 if r == 1 else 0), gid)
    return m


def a_layout_f16_k8():
    """(lane, reg, half) -> (row, col) of A (16 x 8)."""
    m = {}
    for lane in range(32):
        gid, tig = lane >> 2, lane & 3
        for r in range(2):
            for h in range(2):
                m[(lane, r, h)] = (gid + (8 if r == 1 else 0), tig * 2 + h)
    return m


def b_layout_f16_k8():
    m = {}
    for lane in range(32):
        gid, tig = lane >> 2, lane & 3
        for h in range(2):
            m[(lane, 0, h)] = (tig * 2 + h, gid)
    return m


def c_layout():
    """(lane, reg) -> (row, col) of C/D (16 x 8). Shared by both configs."""
    m = {}
    for lane in range(32):
        gid, tig = lane >> 2, lane & 3
        for r in range(4):
            m[(lane, r)] = (gid + (8 if r >= 2 else 0), tig * 2 + (r & 1))
    return m
