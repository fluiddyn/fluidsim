"""
Pythran compatible functions: 2d operators (:mod:`fluidsim.operators.util2d_pythran`)
=====================================================================================

.. autofunction:: dealiasing_setofvar

.. autofunction:: laplacian2_fft

.. autofunction:: invlaplacian2_fft

.. autofunction:: compute_increments_dim1

"""

# import numpy as np

# pythran export dealiasing_setofvar(complex128[][][], uint8[][], int, int)

def dealiasing_setofvar(setofvar_fft, where, n0, n1):
    """Dealiasing of a setofvar arrays"""
    nk = setofvar_fft.shape[0]

    for i0 in range(n0):
        for i1 in range(n1):
            if where[i0, i1]:
                for ik in range(nk):
                    setofvar_fft[ik, i0, i1] = 0.


# pythran export laplacian2_fft(complex128[][], float64[][])

def laplacian2_fft(a_fft, K4):
    """Compute the Laplacian square."""
    return a_fft * K4


# pythran export invlaplacian2_fft(complex128[][], float64[][], int)

def invlaplacian2_fft(a_fft, K4_not0, rank):
    """Compute the inverse Laplace square."""
    invlap2_afft = a_fft / K4_not0
    if rank == 0:
        invlap2_afft[0, 0] = 0.
    return invlap2_afft


# pythran export compute_increments_dim1(float64[][], int)

def compute_increments_dim1(var, irx):
    """Compute the increments of var over the dim 1."""
    n0 = var.shape[0]
    n1 = var.shape[1]
    n1new = n1 - irx
    inc_var = var[:, irx:n1] - var[:, 0:n1new]
    return inc_var
