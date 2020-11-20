"""Mini operator to modify the resolution (Fourier)
===================================================

Internal API
------------

.. autofunction:: fill_field_fft_3d

.. autoclass:: MiniOperModifResol
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex, FFTW3DReal2Complex

from fluidsim.base.init_fields import fill_field_fft_2d

from transonic import boost, Array

A = Array[np.complex128, "3d", "C"]


@boost
def fill_field_fft_3d(field_fft_in: A, field_fft_out: A):
    """Fill the values from field_fft_in in field_fft_out

    This function is specialized for FFTW3DReal2Complex (no MPI).
    """

    [nk0_out, nk1_out, nk2_out] = field_fft_out.shape
    [nk0_in, nk1_in, nk2_in] = field_fft_in.shape

    nk0_min = min(nk0_out, nk0_in)
    nk1_min = min(nk1_out, nk1_in)
    nk2_min = min(nk2_out, nk2_in)

    for ik0 in range(nk0_min // 2 + 1):
        for ik1 in range(nk1_min // 2 + 1):
            for ik2 in range(nk2_min):
                # positive wavenumbers
                field_fft_out[ik0, ik1, ik2] = field_fft_in[ik0, ik1, ik2]
                # negative wavenumbers
                if ik0 > 0 and ik0 < nk0_min // 2:
                    field_fft_out[-ik0, ik1, ik2] = field_fft_in[-ik0, ik1, ik2]
                    if ik1 > 0 and ik1 < nk1_min // 2:
                        field_fft_out[-ik0, -ik1, ik2] = field_fft_in[
                            -ik0, -ik1, ik2
                        ]
                if ik1 > 0 and ik1 < nk1_min // 2:
                    field_fft_out[ik0, -ik1, ik2] = field_fft_in[ik0, -ik1, ik2]


class MiniOperModifResol:
    def __init__(self, shape):

        self.shape = shape

        dimension = self.dimension = len(shape)

        if dimension not in [2, 3]:
            raise NotImplementedError

        if dimension == 2:
            ny, nx = shape
            self.oper_fft = FFTW2DReal2Complex(nx, ny)
            self.axes = tuple("xy")
        else:
            nz, ny, nx = shape
            self.oper_fft = FFTW3DReal2Complex(nx, ny, nz)
            self.axes = tuple("xyz")

        self.fft_as_arg = self.oper_fft.fft_as_arg
        self.ifft_as_arg = self.oper_fft.ifft_as_arg

        self.create_arrayX = self.oper_fft.create_arrayX
        self.create_arrayK = self.oper_fft.create_arrayK

    def fill_field_fft(self, field_spect, field2_spect, oper):

        if self.dimension == 2:
            return fill_field_fft_2d(field_spect, field2_spect)

        fill_field_fft_3d(field_spect, field2_spect)
