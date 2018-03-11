"""
Pythran compatible functions: 3d operators (:mod:`fluidsim.operators.util3d_pythran`)
=====================================================================================

.. autofunction:: dealiasing_setofvar

.. autofunction:: dealiasing_variable

"""

# pythran export dealiasing_setofvar(complex128[][][][], uint8[][][])

def dealiasing_setofvar(sov, where_dealiased):
    """dealiasing 3d setofvar object"""
    nk, n0, n1, n2 = sov.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                if where_dealiased[i0, i1, i2]:
                    for ik in range(nk):
                        sov[ik, i0, i1, i2] = 0.


# pythran export dealiasing_variable(complex128[][][], uint8[][][])

def dealiasing_variable(ff_fft, where_dealiased):
    """dealiasing 3d array"""
    n0, n1, n2 = ff_fft.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                if where_dealiased[i0, i1, i2]:
                    ff_fft[i0, i1, i2] = 0.
