"""Fast Fourier transforms (:mod:`fluidsim.operators.fft.easypyfft`)
==========================================================================

.. currentmodule:: fluidsim.operators.fft.easypyfft

Provides classes for performing fft in 1, 2, and 3 dimensions:

.. autoclass:: fftp2D
   :members:

.. autoclass:: FFTW2DReal2Complex
   :members:

.. autoclass:: FFTW3DReal2Complex
   :members:

.. autoclass:: FFTW1D
   :members:

.. autoclass:: FFTW1DReal2Complex
   :members:

"""

from __future__ import division, print_function

import os
import numpy as np
from copy import copy
import scipy.fftpack as fftp

if 'OMP_NUM_THREADS' in os.environ:
    nthreads = int(os.environ['OMP_NUM_THREADS'])
else:
    nthreads = 1


class fftp2D:
    """ A class to use fftp """
    def __init__(self, nx, ny):
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError('nx and ny should be even')
        self.nx = nx
        self.ny = ny
        self.shapeX = [ny, nx]
        self.nkx = int(float(nx)/2+1)
        self.shapeK = [ny, self.nkx]
        self.coef_norm = nx*ny

        self.fft2D = self.fftp2D
        self.ifft2D = self.ifftp2D

    def fftp2D(self, ff):
        if not (isinstance(ff[0, 0], float)):
            print('Warning: not array of floats')
        big_ff_fft = fftp.fft2(ff)/self.coef_norm
        small_ff_fft = big_ff_fft[:, 0:self.nkx]
        return small_ff_fft

    def ifftp2D(self, small_ff_fft, ARG_IS_COMPLEX=False):
        if not (isinstance(small_ff_fft[0, 0], complex)):
            print('Warning: not array of complexes')
        print('small_ff_fft\n', small_ff_fft)
        big_ff_fft = np.empty(self.shapeX, dtype=np.complex128)
        big_ff_fft[:, 0:self.nkx] = small_ff_fft
        for iky in range(self.ny):
            big_ff_fft[iky, self.nkx:] = \
                small_ff_fft[-iky, self.nkx-2:0:-1].conj()

        print('big_ff_fft final\n', big_ff_fft)
        result_ifft = fftp.ifft2(big_ff_fft*self.coef_norm)
        if np.max(np.imag(result_ifft)) > 10**(-8):
            print ('ifft2: imaginary part of ifft not equal to zero,',
                   np.max(np.imag(result_ifft)))
        return np.real(result_ifft)


class FFTW2DReal2Complex:
    """ A class to use fftw """
    def __init__(self, nx, ny):
        try:
            import pyfftw
        except ImportError as err:
            raise ImportError(
                "ImportError {0}. Instead fftpack can be used (?)", err)
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError('nx and ny should be even')
        shapeX = [ny, nx]
        shapeK = [ny, nx//2 + 1]

        self.shapeX = shapeX
        self.shapeK = shapeK

        self.arrayX = pyfftw.n_byte_align_empty(shapeX, 16, 'float64')
        self.arrayK = pyfftw.n_byte_align_empty(shapeK, 16, 'complex128')

        self.fftplan = pyfftw.FFTW(input_array=self.arrayX,
                                   output_array=self.arrayK,
                                   axes=(0, 1),
                                   direction='FFTW_FORWARD',
                                   threads=nthreads)
        self.ifftplan = pyfftw.FFTW(input_array=self.arrayK,
                                    output_array=self.arrayX,
                                    axes=(0, 1),
                                    direction='FFTW_BACKWARD',
                                    threads=nthreads)

        self.coef_norm = nx*ny

    def fft2d(self, ff):
        self.arrayX[:] = ff
        self.fftplan(normalise_idft=False)
        return self.arrayK/self.coef_norm

    def ifft2d(self, ff_fft):
        self.arrayK[:] = ff_fft
        self.ifftplan(normalise_idft=False)
        return self.arrayX.copy()

    def compute_energy_from_Fourier(self, ff_fft):
        return (np.sum(abs(ff_fft[:, 0])**2 + abs(ff_fft[:, -1])**2) +
                2*np.sum(abs(ff_fft[:, 1:-1])**2))/2

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff)**2)/2

    def project_fft_on_realX(self, ff_fft):
        return self.fft2d(self.ifft2d(ff_fft))


class FFTW3DReal2Complex:
    """ A class to use fftw """
    def __init__(self, nx, ny, nz):
        try:
            import pyfftw
        except ImportError as err:
            raise ImportError(
                "ImportError {0}. Instead fftpack can be used (?)", err)
        if nx % 2 != 0 or ny % 2 != 0 or nz % 2 != 0:
            raise ValueError('nx, ny and nz should be even')
        shapeX = [nz, ny, nx]
        shapeK = [nz, ny, nx//2 + 1]

        self.shapeX = shapeX
        self.shapeK = shapeK

        self.arrayX = pyfftw.n_byte_align_empty(shapeX, 16, 'float64')
        self.arrayK = pyfftw.n_byte_align_empty(shapeK, 16, 'complex128')

        self.fftplan = pyfftw.FFTW(input_array=self.arrayX,
                                   output_array=self.arrayK,
                                   axes=(0, 1, 2),
                                   direction='FFTW_FORWARD',
                                   threads=nthreads)
        self.ifftplan = pyfftw.FFTW(input_array=self.arrayK,
                                    output_array=self.arrayX,
                                    axes=(0, 1, 2),
                                    direction='FFTW_BACKWARD',
                                    threads=nthreads)

        self.coef_norm = nx*ny*nz

    def fft(self, ff):
        self.arrayX[:] = ff
        self.fftplan(normalise_idft=False)
        return self.arrayK/self.coef_norm

    def ifft(self, ff_fft):
        self.arrayK[:] = ff_fft
        self.ifftplan(normalise_idft=False)
        return self.arrayX.copy()

    def sum_wavenumbers(self, ff_fft):
        return (np.sum(ff_fft[:, :, 0] + ff_fft[:, :, -1]) +
                2*np.sum(ff_fft[:, :, 1:-1]))/2

    def compute_energy_from_Fourier(self, ff_fft):
        return self.sum_wavenumbers(abs(ff_fft)**2)

    def get_shapeX_loc(self):
        return self.shapeX

    def get_shapeX_seq(self):
        return self.shapeX

    def get_shapeK_loc(self):
        return self.shapeK

    def get_shapeK_seq(self):
        return self.shapeK

    def get_k_adim(self):
        nK0, nK1, nK2 = self.shapeK
        kz_adim_max = nK0//2
        ky_adim_max = nK1//2
        return (np.r_[0:kz_adim_max+1, -kz_adim_max+1:0],
                np.r_[0:ky_adim_max+1, -ky_adim_max+1:0],
                np.arange(nK2))

    def get_k_adim_loc(self):
        return self.get_k_adim()

    def get_orderK_dimX(self):
        return 0, 1, 2

    def get_seq_index_firstK(self):
        return 0, 0

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff)**2)/2

    def project_fft_on_realX(self, ff_fft):
        return self.fft2d(self.ifft2d(ff_fft))


class FFTW1D:
    """ A class to use fftw 1D """
    def __init__(self, n):
        try:
            import pyfftw
        except ImportError as err:
            raise ImportError("ImportError. Instead fftpack?", err)

        if n % 2 != 0:
            raise ValueError('n should be even')
        shapeX = [n]
        shapeK = [n]
        self.shapeX = shapeX
        self.shapeK = shapeK
        self.arrayX = pyfftw.n_byte_align_empty(shapeX, 16, 'complex128')
        self.arrayK = pyfftw.n_byte_align_empty(shapeK, 16, 'complex128')
        self.fftplan = pyfftw.FFTW(input_array=self.arrayX,
                                   output_array=self.arrayK,
                                   axes=(-1,),
                                   direction='FFTW_FORWARD', threads=nthreads)
        self.ifftplan = pyfftw.FFTW(input_array=self.arrayK,
                                    output_array=self.arrayX,
                                    axes=(-1,),
                                    direction='FFTW_BACKWARD',
                                    threads=nthreads)

        self.coef_norm = n

    def fft(self, ff):
        self.arrayX[:] = ff
        self.fftplan()
        return self.arrayK/self.coef_norm

    def ifft(self, ff_fft):
        self.arrayK[:] = ff_fft
        self.ifftplan()
        return self.arrayX.copy()


class FFTW1DReal2Complex:
    """ A class to use fftw 1D """
    def __init__(self, arg, axis=-1):
        try:
            import pyfftw
        except ImportError as err:
            raise ImportError("ImportError. Instead fftpack?", err)

        if isinstance(arg, int):
            n = arg
            shapeX = [n]
            shapeK = [n//2+1]
        else:
            n = arg[axis]
            shapeX = arg
            shapeK = list(copy(arg))
            shapeK[axis] = n//2+1

        if n % 2 != 0:
            raise ValueError('n should be even')

        self.shapeX = shapeX
        self.shapeK = shapeK
        self.arrayX = pyfftw.n_byte_align_empty(shapeX, 16, 'float64')
        self.arrayK = pyfftw.n_byte_align_empty(shapeK, 16, 'complex128')
        self.fftplan = pyfftw.FFTW(input_array=self.arrayX,
                                   output_array=self.arrayK,
                                   axes=(axis,),
                                   direction='FFTW_FORWARD', threads=nthreads)
        self.ifftplan = pyfftw.FFTW(input_array=self.arrayK,
                                    output_array=self.arrayX,
                                    axes=(axis,),
                                    direction='FFTW_BACKWARD',
                                    threads=nthreads)

        self.coef_norm = n

    def fft(self, ff):
        self.arrayX[:] = ff
        self.fftplan(normalise_idft=False)
        return self.arrayK/self.coef_norm

    def ifft(self, ff_fft):
        self.arrayK[:] = ff_fft
        self.ifftplan(normalise_idft=False)
        return self.arrayX.copy()

    def compute_energy_from_Fourier(self, ff_fft):
        return (abs(ff_fft[0])**2 +
                2*np.sum(abs(ff_fft[1:-1])**2) +
                abs(ff_fft[-1])**2)/2

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff)**2)/2
