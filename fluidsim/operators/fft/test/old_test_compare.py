
from __future__ import division, print_function

import unittest

import sys

import numpy as np

import fluiddyn.util.mpi as mpi
from fluidsim.operators.fft import easypyfft
try:
    from fluidsim.operators.fft import fftw2dmpicy
    FFTWMPI = True
except ImportError:
    FFTWMPI = False

@unittest.skipIf(mpi.nb_proc > 1, 'Will fail if mpi.nb_proc > 1')
@unittest.skipIf(not FFTWMPI, 'fftw2dmpicy fails to be imported.')
@unittest.skipIf(sys.platform.startswith("win"), "Will fail on Windows")
class TestFFT2Dmpi(unittest.TestCase):

    def test_fft(self):
        """Should be able to..."""
        nx = 8
        ny = 8
        n0, n1 = ny, nx
        op = fftw2dmpicy.FFT2Dmpi(n0, n1, TRANSPOSED=False)
        op2 = easypyfft.FFTW2DReal2Complex(nx, ny)

        func_fft = (np.random.random(op.shapeK_loc) +
                    1.j*np.random.random(op.shapeK_loc))

        func = op.ifft2d(func_fft)
        func2 = op2.ifft2d(func_fft)

        self.assertTrue(np.allclose(func, func2))


if __name__ == '__main__':
    unittest.main()
