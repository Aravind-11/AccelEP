"""stack_matmul launches one unpadded product and returns the diagonal tiles."""

import numpy as np

from stack_matmul import stack_matmul


def main():
    calls = []

    def spy(left, right):
        calls.append((left.shape, right.shape, left.dtype, right.dtype))
        return np.matmul(left, right)

    def rejects(pairs):
        before = len(calls)
        try:
            stack_matmul(pairs, matmul=spy)
        except ValueError:
            if len(calls) != before:
                raise SystemExit("matmul ran before validation failed")
            return
        raise SystemExit(f"expected ValueError for {pairs!r}")

    rejects([])
    rejects([("no", "nd")])
    rejects([(np.zeros((2, 2)), np.zeros((2, 2), np.float32))])
    rejects([(np.zeros((2, 2), np.float64), np.zeros((2, 2), np.float64))])
    rejects([(np.zeros((1, 2, 2), np.float32), np.zeros((1, 2, 2), np.float32))])
    rejects(
        [
            (np.zeros((1, 4), np.float32), np.zeros((4, 3), np.float32)),
            (np.zeros((2, 4), np.float32), np.zeros((4, 3), np.float32)),
        ]
    )
    rejects(
        [
            (np.zeros((2, 4), np.float32), np.zeros((4, 3), np.float32)),
            (np.zeros((2, 5), np.float32), np.zeros((5, 3), np.float32)),
        ]
    )
    rejects(
        [
            (np.zeros((2, 3), np.float32), np.zeros((4, 3), np.float32)),
        ]
    )

    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(3):
        pairs.append(
            (
                rng.integers(-3, 4, size=(2, 4)).astype(np.float32),
                rng.integers(-3, 4, size=(4, 3)).astype(np.float32),
            )
        )
    tiles = stack_matmul(pairs, matmul=spy)
    if calls != [((6, 4), (4, 9), np.float32, np.float32)]:
        raise SystemExit(f"launch shapes {calls}")
    a_cat = np.concatenate([left for left, _right in pairs], axis=0)
    b_cat = np.concatenate([right for _left, right in pairs], axis=1)
    full = np.matmul(a_cat, b_cat)
    for index, (left, right) in enumerate(pairs):
        if not np.array_equal(tiles[index], left @ right):
            raise SystemExit(f"tile {index} is not the unbatched product")
        off = full[index * 2 : (index + 1) * 2, ((index + 1) % 3) * 3 : ((index + 1) % 3) * 3 + 3]
        if not np.array_equal(off, left @ right) and np.array_equal(tiles[index], off):
            raise SystemExit(f"tile {index} returned an off-diagonal block")
    print("stack_matmul tests passed")


if __name__ == "__main__":
    main()
