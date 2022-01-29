"""Operators 1d (:mod:`fluidsim.operators.operators1d`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectral1D
   :members:
   :private-members:

"""

import numpy as np
from fluiddyn.calcul.easypyfft import FFTW1DReal2Complex
from .base import OperatorsBase1D
from ..base.setofvariables import SetOfVariables


class OperatorsPseudoSpectral1D(OperatorsBase1D):
    """1D operators for pseudospectral solvers."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container."""
        params = OperatorsBase1D._complete_params_with_default(params)
        params.oper._set_attribs(
            {"type_fft": "sequential", "coef_dealiasing": 2.0 / 3}
        )
        return params

    def __init__(self, params, SEQUENTIAL=None):
        super().__init__(params)

        assert params.oper.type_fft == "sequential"
        nx = params.oper.nx
        opfft = FFTW1DReal2Complex(nx)
        self.fft = opfft.fft
        self.ifft = opfft.ifft
        self.fft_as_arg = opfft.fft_as_arg
        self.ifft_as_arg = opfft.ifft_as_arg
        self.sum_wavenumbers = opfft.sum_wavenumbers

        self.shapeK = self.shapeK_loc = opfft.shapeK
        self.deltakx = 2 * np.pi / self.Lx
        self.nkx = self.shapeK[0]
        self.kx = self.deltakx * np.arange(self.nkx)
        assert self.nkx == self.nx // 2 + 1

        self.coef_dealiasing = params.oper.coef_dealiasing
        kx_max = self.deltakx * (self.nkx - 1)
        assert kx_max == self.kx.max()

        CONDKX = abs(self.kx) > self.coef_dealiasing * kx_max
        self.where_dealiased = np.array(CONDKX, dtype=bool)

        # for spectra
        self.nkxE = self.nx // 2 + 1
        self.nkxE2 = (self.nx + 1) // 2

        # print('nkxE, nkxE2', self.nkxE, self.nkxE2)

        self.kxE = self.deltakx * np.arange(self.nkxE)
        self.khE = self.kxE
        self.nkhE = self.nkxE

        self.KX = self.kx
        self.K2 = self.KX**2
        self.K4 = self.K2**2
        self.K8 = self.K4**2
        self.K = self.KX

    def produce_long_str_describing_oper(self):
        return super().produce_long_str_describing_oper("Pseudospectral")

    def dealiasing(self, *args):
        for thing in args:
            if isinstance(thing, SetOfVariables):
                self.dealiasing_setofvar(thing)
            elif isinstance(thing, np.ndarray):
                self.dealiasing_variable(thing)

    def dealiasing_variable(self, f_fft):
        f_fft[self.where_dealiased] = 0.0

    def dealiasing_setofvar(self, sov):
        for ik in range(sov.nvar):
            self.dealiasing_variable(sov[ik])

    def pxffft_from_fft(self, f_fft):
        """Return the first derivative of f_fft in spectral space."""
        kx = self.kx
        px_f_fft = 1j * kx * f_fft
        return px_f_fft

    def get_phases_random(self):
        alpha = np.random.uniform(-0.5, 0.5)
        beta = alpha + 0.5 if alpha < 0 else alpha - 0.5

        phase_alpha = alpha * self.deltax * self.kx
        phase_beta = beta * self.deltax * self.kx
        return phase_alpha, phase_beta
