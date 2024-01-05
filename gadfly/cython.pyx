import numpy as np
cimport numpy as np
cimport cython

__all__ = ["p_mode_frequencies"]

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# @cython.boundscheck(False)
# @cython.wraparound(False)
def p_mode_frequencies(double [:, :, :] nu_peak_init,
                       long [:] n, long [:] ell, long [:] m,
                       float eps, float delta_nu, #float delta_nu_star,
                       float delta_nu_02):
    """
    """
    cdef Py_ssize_t len_n = len(n)
    cdef Py_ssize_t len_ell = len(ell)
    cdef Py_ssize_t len_m = len(m)
    cdef Py_ssize_t i, j, k

    for k in range(len_n):
        for i in range(len_m):
            for j in range(len_ell):
                if k > 0 and j == 0:
                    nu_peak_init[i, j, k] = nu_peak_init[i, j+2, k-1] + delta_nu_02

                if j == 1 and k > 0:
                    # García (2019) eqn 17
                    delt = 0.5 * (
                        nu_peak_init[i, j, k-1] +
                        nu_peak_init[i, j, k]
                    ) - nu_peak_init[i, j-1, k]
                    nu_peak_init[i, j, k] -= delt

                if j == 1 and 0 < k < len_n - 1 and j + 2 < len_ell:
                    # García (2019) eqn 17
                    delt = 0.5 * (
                        nu_peak_init[i, j, k-1] +
                        nu_peak_init[i, j, k]
                    ) - nu_peak_init[i, j-1, k]

                    nu_peak_init[i, j+2, k+1] -= delta_nu_02 - delt
    return nu_peak_init
