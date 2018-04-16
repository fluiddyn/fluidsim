"""
Pythran compatible functions: 3d operators (:mod:`fluidsim.operators.util3d_pythran`)
=====================================================================================

.. autofunction:: dealiasing_setofvar

.. autofunction:: dealiasing_variable

.. autofunction:: urudfft_from_vxvyfft

"""

# pythran export dealiasing_setofvar(complex128[][][][], uint8[][][])


def dealiasing_setofvar(sov, where_dealiased):
    """Dealiasing 3d setofvar object.

    Parameters
    ----------

    sov : 4d ndarray
        A set of variables array.

    where_dealiased : 3d ndarray
        A 3d array of "booleans" (actually uint8).

    """
    nk, n0, n1, n2 = sov.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                if where_dealiased[i0, i1, i2]:
                    for ik in range(nk):
                        sov[ik, i0, i1, i2] = 0.


# pythran export dealiasing_variable(complex128[][][], uint8[][][])


def dealiasing_variable(ff_fft, where_dealiased):
    """Dealiasing 3d array"""
    n0, n1, n2 = ff_fft.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                if where_dealiased[i0, i1, i2]:
                    ff_fft[i0, i1, i2] = 0.


# pythran export urudfft_from_vxvyfft(
#     complex128[][][], complex128[][][], float64[][][], float64[][][], int)


def urudfft_from_vxvyfft(vx_fft, vy_fft, kx, ky, rank):
    k2 = kx ** 2 + ky ** 2
    k2[k2 == 0.] = 1e-10

    divh_fft = 1j * (kx * vx_fft + ky * vy_fft)
    urx_fft = vx_fft - divh_fft * kx / k2
    ury_fft = vy_fft - divh_fft * ky / k2

    udx_fft = vx_fft - urx_fft
    udy_fft = vy_fft - ury_fft

    return urx_fft, ury_fft, udx_fft, udy_fft
