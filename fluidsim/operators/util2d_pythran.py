"""
Pythran compatible functions: 2d operators (:mod:`fluidsim.operators.util2d_pythran`)
=====================================================================================

.. autofunction:: dealiasing_setofvar

.. autofunction:: laplacian_fft

.. autofunction:: invlaplacian_fft

.. autofunction:: compute_increments_dim1

"""

# import numpy as np

# pythran export dealiasing_setofvar(complex128[][][], uint8[][], int, int)


def dealiasing_setofvar(setofvar_fft, where, n0, n1):
    """Dealiasing of a setofvar arrays."""
    nk = setofvar_fft.shape[0]

    for i0 in range(n0):
        for i1 in range(n1):
            if where[i0, i1]:
                for ik in range(nk):
                    setofvar_fft[ik, i0, i1] = 0.


# pythran export laplacian_fft(complex128[][], float64[][])


def laplacian_fft(a_fft, Kn):
    """Compute the n-th order Laplacian."""
    return a_fft * Kn


# pythran export invlaplacian_fft(complex128[][], float64[][], int)


def invlaplacian_fft(a_fft, Kn_not0, rank):
    """Compute the n-th order inverse Laplacian."""
    invlap_afft = a_fft / Kn_not0
    if rank == 0:
        invlap_afft[0, 0] = 0.
    return invlap_afft


# pythran export compute_increments_dim1(float64[][], int)


def compute_increments_dim1(var, irx):
    """Compute the increments of var over the dim 1."""
    n1 = var.shape[1]
    n1new = n1 - irx
    inc_var = var[:, irx:n1] - var[:, 0:n1new]
    return inc_var
