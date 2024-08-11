import numpy as np
import scipy.special as sp

PI = np.longdouble(3.1415926535897932384626433832795028841971)


def bartlett_longdouble(M):
    values = np.array([0.0, M], dtype=np.longdouble)
    M = values[1]

    if M < 1:
        return np.array([], dtype=values.dtype)
    if M == 1:
        return np.ones(1, dtype=values.dtype)
    n = np.arange(1 - M, M, 2, dtype=values.dtype)
    return np.where(np.less_equal(n, 0), 1 + n / (M - 1), 1 - n / (M - 1))


def hanning_longdouble(M):
    values = np.array([0.0, M], dtype=np.longdouble)
    M = values[1]

    if M < 1:
        return np.array([], dtype=values.dtype)
    if M == 1:
        return np.ones(1, dtype=values.dtype)
    n = np.arange(1 - M, M, 2, dtype=values.dtype)
    return 0.5 + 0.5 * np.cos(PI * n / (M - 1), dtype=values.dtype)


# kaiserの中で使われているi0の計算が、longdoubleでは実行できないらしい


def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError("Window length M must be a non-negative integer")
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w


def gaussian_longdouble(M, std, sym=True):
    if _len_guards(M):
        return np.ones(M, dtype=np.longdouble)
    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M, dtype=np.longdouble) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-(n**2) / sig2, dtype=np.longdouble)

    return _truncate(w, needs_trunc)
