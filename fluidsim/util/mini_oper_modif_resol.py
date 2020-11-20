from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex, FFTW3DReal2Complex

from fluidsim.base.init_fields import fill_field_fft_2d


def fill_field_fft_3d(field_fft_in, field_fft_out):
    """

    This function is specialized for FFTW3DReal2Complex (no MPI).

    It's better if we don't pass oper_in and oper_out to this function, but we
    can add other arguments if needed.

    To run the test testing this function::

      pytest fluidsim/util/test_util.py::TestModifResol3d

    """
    raise NotImplementedError

    [nk0, nk1, nk2] = field_fft_out.shape
    [nk0_in, nk1_in, nk2_in] = field_fft_in.shape

    nk0_min = min(nk0, nk0_in)
    nk1_min = min(nk1, nk1_in)
    nk2_min = min(nk2, nk2_in)

    for ik0 in range(nk0_min):
        for ik1 in range(nk1_min):
            for ik2 in range(nk2_min):
                kx_adim, ky_adim, kz_adim = oper_in.kadim_from_ik012rank(
                    ik0, ik1, ik2
                )
                oper_out.set_value_spect(
                    field_fft_out,
                    field_fft_in[ik0, ik1, ik2],
                    kx_adim,
                    ky_adim,
                    kz_adim,
                )


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
