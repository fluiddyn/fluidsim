
import unittest

import numpy as np


from fluidsim.operators.fft import easypyfft

# from fluiddyn.io import stdout_redirected


class TestFFTW1DReal2Complex(unittest.TestCase):

    def test_fft(self):
        """Should be able to..."""
        nx = 128
        op = easypyfft.FFTW1DReal2Complex(nx)

        func_fft = np.zeros(op.shapeK, dtype=np.complex128)
        func_fft[0] = 1

        self.compute_and_check(func_fft, op)

    def compute_and_check(self, func_fft, op):

        func = op.ifft(func_fft)
        back_fft = op.fft(func)
        back = op.ifft(back_fft)

        self.assertTrue(np.allclose(func_fft, back_fft))
        self.assertTrue(np.allclose(func, back))

        energyX = op.compute_energy_from_spatial(func)
        energyK = op.compute_energy_from_Fourier(func_fft)
        energyKback = op.compute_energy_from_Fourier(back_fft)

        self.assertAlmostEqual(energyX, energyK)
        self.assertAlmostEqual(energyK, energyKback)

    def test_fft_random(self):
        """Should be able to..."""
        nx = 128
        op = easypyfft.FFTW1DReal2Complex(nx)

        func_fft = (np.random.random(op.shapeK)
                    + 1.j*np.random.random(op.shapeK))
        func = op.ifft(func_fft)
        func_fft = op.fft(func)

        self.compute_and_check(func_fft, op)


class TestFFTW2DReal2Complex(unittest.TestCase):

    def test_fft(self):
        """Should be able to..."""
        nx = 4
        ny = 2
        op = easypyfft.FFTW2DReal2Complex(nx, ny)

        func_fft = np.zeros(op.shapeK, dtype=np.complex128)
        func_fft[0, 1] = 1

        self.compute_and_check(func_fft, op)

    def compute_and_check(self, func_fft, op):

        energyK = op.compute_energy_from_Fourier(func_fft)

        func = op.ifft2d(func_fft)
        energyX = op.compute_energy_from_spatial(func)

        back_fft = op.fft2d(func)
        energyKback = op.compute_energy_from_Fourier(back_fft)
        back = op.ifft2d(back_fft)

        self.assertTrue(np.allclose(func_fft, back_fft))
        self.assertTrue(np.allclose(func, back))

        self.assertAlmostEqual(energyX, energyK)
        self.assertAlmostEqual(energyK, energyKback)

    def test_fft_random(self):
        """Should be able to..."""
        nx = 64
        ny = 128
        op = easypyfft.FFTW2DReal2Complex(nx, ny)

        func_fft = (np.random.random(op.shapeK)
                    + 1.j*np.random.random(op.shapeK))
        func = op.ifft2d(func_fft)
        func_fft = op.fft2d(func)

        self.compute_and_check(func_fft, op)


class TestFFTW3DReal2Complex(unittest.TestCase):

    def test_fft(self):
        """Should be able to..."""
        nx = 4
        ny = 2
        nz = 8
        op = easypyfft.FFTW3DReal2Complex(nx, ny, nz)

        func_fft = np.zeros(op.shapeK, dtype=np.complex128)
        func_fft[0, 0, 1] = 1

        self.compute_and_check(func_fft, op)

    def compute_and_check(self, func_fft, op):

        energyK = op.compute_energy_from_Fourier(func_fft)

        func = op.ifft(func_fft)
        energyX = op.compute_energy_from_spatial(func)

        back_fft = op.fft(func)
        energyKback = op.compute_energy_from_Fourier(back_fft)
        back = op.ifft(back_fft)

        self.assertTrue(np.allclose(func_fft, back_fft))
        self.assertTrue(np.allclose(func, back))

        self.assertAlmostEqual(energyX, energyK)
        self.assertAlmostEqual(energyK, energyKback)

    def test_fft_random(self):
        """Should be able to..."""
        nx = 8
        ny = 8
        nz = 32
        op = easypyfft.FFTW3DReal2Complex(nx, ny, nz)

        func_fft = (np.random.random(op.shapeK)
                    + 1.j*np.random.random(op.shapeK))
        func = op.ifft(func_fft)
        func_fft_back = op.fft(func)

        self.compute_and_check(func_fft_back, op)


if __name__ == '__main__':
    unittest.main()
