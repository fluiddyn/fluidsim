
import unittest
import numpy as np
import sys

from fluiddyn.io import stdout_redirected
import fluiddyn.util.mpi as mpi

from fluidsim.base.solvers.pseudo_spect import Simul
from fluidsim.operators.operators import OperatorsPseudoSpectral2D

try:
    from fluidsim.operators.fft import fftw2dmpicy
    FFTWMPI = True
    type_fft = 'FFTWCY'
except ImportError:
    FFTWMPI = False
    if mpi.nb_proc == 1:
        type_fft = 'FFTWPY'
    else:
        type_fft = 'FFTWCCY'


def create_oper(type_fft='FFTWCY'):

    params = Simul.create_default_params()

    nh = 8
    params.oper.nx = nh
    params.oper.ny = nh
    Lh = 6.
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    params.oper.type_fft = type_fft

    params.oper.coef_dealiasing = 2./3

    with stdout_redirected():
        oper = OperatorsPseudoSpectral2D(params=params)

    return oper


@unittest.skipIf(sys.platform.startswith("win"), "Will fail on Windows")
class TestOperators(unittest.TestCase):
    @unittest.skipIf(not FFTWMPI, 'fftw2dmpicy fails to be imported.')
    @unittest.skipIf(mpi.nb_proc > 1, 'Will fail if mpi.nb_proc > 1')
    def test_create(self):
        """Should be able to ..."""
        oper = create_oper('FFTWCY')

        rot = oper.random_arrayX()
        rot_fft = oper.fft2(rot)
        rot_fft[0, 0] = 0.

        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        rot2_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)

        self.assertTrue(np.allclose(rot2_fft, rot_fft))

        oper_py = create_oper('FFTWPY')

        ux_fft, uy_fft = oper_py.vecfft_from_rotfft(rot_fft)
        rot2_fft = oper_py.rotfft_from_vecfft(ux_fft, uy_fft)

        self.assertTrue(np.allclose(rot2_fft, rot_fft))

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot2_fft, py_rot2_fft = oper_py.gradfft_from_fft(rot_fft)

        self.assertTrue(np.allclose(px_rot_fft, px_rot2_fft))

    @unittest.skipIf(mpi.nb_proc > 1, 'Will fail if mpi.nb_proc > 1')
    @unittest.skipIf(not FFTWMPI, 'fftw2dmpicy fails to be imported.')
    def test_tendency(self):

        oper = create_oper('FFTWCY')
        rot = oper.random_arrayX()
        rot_fft = oper.fft2(rot)
        rot_fft[0, 0] = 0.
        oper.dealiasing(rot_fft)

        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        ux = oper.ifft2(ux_fft)
        uy = oper.ifft2(uy_fft)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = oper.ifft2(px_rot_fft)
        py_rot = oper.ifft2(py_rot_fft)

        Frot = -ux*px_rot - uy*py_rot
        Frot_fft = oper.fft2(Frot)
        oper.dealiasing(Frot_fft)

        T_rot = np.real(Frot_fft.conj()*rot_fft)

        ratio = (oper.sum_wavenumbers(T_rot) /
                 oper.sum_wavenumbers(abs(T_rot)))

        self.assertGreater(1e-15, ratio)

        # print ('sum(T_rot) = {0:9.4e} ; '
        #        'sum(abs(T_rot)) = {1:9.4e}').format(
        #            oper.sum_wavenumbers(T_rot),
        #            oper.sum_wavenumbers(abs(T_rot)))

        oper2 = create_oper('FFTWPY')

        ux_fftpy, uy_fftpy = oper2.vecfft_from_rotfft(rot_fft)

        self.assertTrue(np.allclose(ux_fft, ux_fftpy))
        self.assertTrue(np.allclose(uy_fft, uy_fftpy))

        uxpy = oper2.ifft2(ux_fftpy)
        uypy = oper2.ifft2(uy_fftpy)

        self.assertTrue(np.allclose(ux_fft, ux_fftpy))
        self.assertTrue(np.allclose(uy_fft, uy_fftpy))

        self.assertTrue(np.allclose(ux, uxpy))
        self.assertTrue(np.allclose(uy, uypy))

        px_rot_fftpy, py_rot_fftpy = oper2.gradfft_from_fft(rot_fft)
        px_rotpy = oper2.ifft2(px_rot_fftpy)
        py_rotpy = oper2.ifft2(py_rot_fftpy)

        Frotpy = -uxpy*px_rotpy - uypy*py_rotpy
        Frot_fftpy = oper2.fft2(Frotpy)
        oper2.dealiasing(Frot_fftpy)

        T_rotpy = np.real(Frot_fftpy.conj()*rot_fft)

        ratio = (oper2.sum_wavenumbers(T_rotpy) /
                 oper2.sum_wavenumbers(abs(T_rotpy)))

        # print ('sum(T_rot) = {0:9.4e} ; '
        #        'sum(abs(T_rot)) = {1:9.4e}').format(
        #            oper2.sum_wavenumbers(T_rotpy),
        #            oper2.sum_wavenumbers(abs(T_rotpy)))

        self.assertGreater(1e-15, ratio)

    def test_laplacian2(self):
        oper = create_oper(type_fft)
        ff = oper.random_arrayX()
        ff_fft = oper.fft2(ff)
        ff_fft[0, 0] = 0.

        lap_fft = oper.laplacian2_fft(ff_fft)
        ff_fft_back = oper.invlaplacian2_fft(lap_fft)

        self.assertTrue(np.allclose(ff_fft, ff_fft_back))

    def test_monge_ampere(self):
        oper = create_oper(type_fft)
        a = oper.random_arrayX()
        a_fft = oper.fft2(a)
        a_fft[0, 0] = 0.

        b = oper.random_arrayX()
        b_fft = oper.fft2(b)
        b_fft[0, 0] = 0.

        ma_py = oper.monge_ampere_from_fft_python(a_fft, b_fft)
        ma_cy = oper.monge_ampere_from_fft(a_fft, b_fft)

        self.assertTrue(np.allclose(ma_py, ma_cy))


if __name__ == '__main__':
    unittest.main()
