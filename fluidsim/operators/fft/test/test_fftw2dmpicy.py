
import unittest
import numpy as np
import sys

try:
    from fluidsim.operators.fft import fftw2dmpicy
    FFTWMPI = True
except ImportError:
    FFTWMPI = False


@unittest.skipIf(not FFTWMPI, 'fftw2dmpicy fails to be imported.')
@unittest.skipIf(sys.platform.startswith("win"), "Will fail on Windows")
class TestFFT2Dmpi(unittest.TestCase):

    def test_fft(self):
        """Should be able to..."""
        nx = 4
        ny = 2
        n0, n1 = ny, nx
        op = fftw2dmpicy.FFT2Dmpi(n0, n1, TRANSPOSED=False)

        func_fft = np.zeros(op.shapeK_loc, dtype=np.complex128)
        func_fft[0, 1] = 1

        self.compute_and_check(func_fft, op)

    def compute_and_check(self, func_fft, op):

        energyK = op.compute_energy_from_Fourier(func_fft)

        func = op.ifft2d(func_fft)
        energyX = op.compute_energy_from_spatial(func)

        back_fft = op.fft2d(func)
        energyKback = op.compute_energy_from_Fourier(back_fft)
        back = op.ifft2d(back_fft)

        # mean_fft = op.get_mean_fft(func_fft)

        self.assertTrue(np.allclose(func_fft, back_fft))
        self.assertTrue(np.allclose(func, back))

        self.assertAlmostEqual(energyX, energyK)
        self.assertAlmostEqual(energyK, energyKback)

    def test_fft_random(self):
        """Should be able to..."""
        nx = 32
        ny = 64
        op = fftw2dmpicy.FFT2Dmpi(nx, ny, TRANSPOSED=False)

        func_fft = (np.random.random(op.shapeK_loc) +
                    1.j*np.random.random(op.shapeK_loc))

        func = op.ifft2d(func_fft)
        func_fft = op.fft2d(func)

        self.compute_and_check(func_fft, op)


if __name__ == '__main__':
    unittest.main()
