"""Host-only grouped matmul. One launch, original K, diagonal tiles returned."""

import numpy as np


def stack_matmul(pairs, matmul=np.matmul):
    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError("stack_matmul expects a non-empty list of pairs")
    checked = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("each pair must be a tuple of two arrays")
        left, right = pair
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ValueError("each input must already be an ndarray")
        if left.ndim != 2 or right.ndim != 2 or left.dtype != np.float32 or right.dtype != np.float32:
            raise ValueError("each input must be a float32 matrix")
        checked.append((left, right))
    rows, contract = checked[0][0].shape
    contract_b, cols = checked[0][1].shape
    if contract != contract_b:
        raise ValueError("K does not match")
    for left, right in checked:
        if left.shape != (rows, contract) or right.shape != (contract, cols):
            raise ValueError("every pair must share M, K, and N")
    count = len(checked)
    a_cat = np.concatenate([left for left, _right in checked], axis=0)
    b_cat = np.concatenate([right for _left, right in checked], axis=1)
    full = matmul(a_cat, b_cat)
    tiles = []
    for index in range(count):
        tile = full[index * rows : (index + 1) * rows, index * cols : (index + 1) * cols]
        tiles.append(tile.copy())
    return tiles
