import numpy as np


def multiply_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ai, aj = a.shape
    bj, bk = b.shape
    assert aj == bj

    c = np.zeros((ai, bk))

    for i in range(ai):
        for j in range(aj):
            for k in range(bk):
                c[i][k] += a[i][j] * b[j][k]

    return c


def multiply_strassen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)

    m1 = (a[0, 0] + a[1, 1]) * (b[0, 0] + b[1, 1])
    m2 = (a[1, 0] + a[1, 1]) * b[0, 0]
    m3 = a[0, 0] * (b[0, 1] - b[1, 1])
    m4 = a[1, 1] * (b[1, 0] - b[0, 0])
    m5 = (a[0, 0] + a[0, 1]) * b[1, 1]
    m6 = (a[1, 0] - a[0, 0]) * (b[0, 0] + b[0, 1])
    m7 = (a[0, 1] - a[1, 1]) * (b[1, 0] + b[1, 1])

    c = np.zeros((2, 2))
    c[0, 0] = m1 + m4 - m5 + m7
    c[0, 1] = m3 + m5
    c[1, 0] = m2 + m4
    c[1, 1] = m1 - m2 + m3 + m6

    return c


def multiply_strassen_recursive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape

    if a.shape == (1, 1):
        return a[0, 0] * b[0, 0]

    full, _ = a.shape
    half = int(full / 2)

    m1 = multiply_strassen_recursive(
        (a[0:half, 0:half] + a[half:full, half:full]),
        (b[0:half, 0:half] + b[half:full, half:full]),
    )
    m2 = multiply_strassen_recursive(
        (a[half:full, 0:half] + a[half:full, half:full]), b[0:half, 0:half]
    )
    m3 = multiply_strassen_recursive(
        a[0:half, 0:half], (b[0:half, half:full] - b[half:full, half:full])
    )
    m4 = multiply_strassen_recursive(
        a[half:full, half:full], (b[half:full, 0:half] - b[0:half, 0:half])
    )
    m5 = multiply_strassen_recursive(
        (a[0:half, 0:half] + a[0:half, half:full]), b[half:full, half:full]
    )
    m6 = multiply_strassen_recursive(
        (a[half:full, 0:half] - a[0:half, 0:half]),
        (b[0:half, 0:half] + b[0:half, half:full]),
    )
    m7 = multiply_strassen_recursive(
        (a[0:half, half:full] - a[half:full, half:full]),
        (b[half:full, 0:half] + b[half:full, half:full]),
    )

    c = np.zeros((full, full), dtype=a.dtype)
    c[0:half, 0:half] = m1 + m4 - m5 + m7
    c[0:half, half:full] = m3 + m5
    c[half:full, 0:half] = m2 + m4
    c[half:full, half:full] = m1 - m2 + m3 + m6

    return c
