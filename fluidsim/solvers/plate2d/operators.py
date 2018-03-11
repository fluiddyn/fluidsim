"""Operators plate2d model (:mod:`fluidsim.operators.operators_plate2d`)
========================================================================

Provides

.. autoclass:: OperatorsPseudoSpectralPlate2D
   :members:
   :private-members:

"""


import numpy as np

from ...operators.operators2d import OperatorsPseudoSpectral2D

from .util_oper_pythran import (
    monge_ampere_step0, monge_ampere_step1)


class OperatorsPseudoSpectralPlate2D(OperatorsPseudoSpectral2D):
    """Operators for the plate2d model.

    """
    def __init__(self, params, SEQUENTIAL=None, goal_to_print=None):
        super(OperatorsPseudoSpectralPlate2D, self).__init__(
            params, SEQUENTIAL=SEQUENTIAL, goal_to_print=goal_to_print)

        self.KX2 = self.KX**2
        self.KY2 = self.KY**2
        self.KXKY = self.KX*self.KY

        self.tmp_pxx_a = np.empty_like(self.XX)
        self.tmp_pyy_a = np.empty_like(self.XX)
        self.tmp_pxy_a = np.empty_like(self.XX)

        self.tmp_pxx_b = np.empty_like(self.XX)
        self.tmp_pyy_b = np.empty_like(self.XX)
        self.tmp_pxy_b = np.empty_like(self.XX)

    # def monge_ampere_from_fft2(self, a_fft, b_fft):
    #     return monge_ampere_from_fft(
    #         a_fft, b_fft, self.KX, self.KY, self.ifft2)

    def monge_ampere_from_fft(self, a_fft, b_fft):
        """Compute the Monge-Ampere operator"""

        pxx_a_fft, pyy_a_fft, pxy_a_fft, pxx_b_fft, pyy_b_fft, pxy_b_fft = \
            monge_ampere_step0(a_fft, b_fft, self.KX2, self.KY2, self.KXKY)

        self.ifft_as_arg(pxx_a_fft, self.tmp_pxx_a)
        self.ifft_as_arg(pyy_a_fft, self.tmp_pyy_a)
        self.ifft_as_arg(pxy_a_fft, self.tmp_pxy_a)
        self.ifft_as_arg(pxx_b_fft, self.tmp_pxx_b)
        self.ifft_as_arg(pyy_b_fft, self.tmp_pyy_b)
        self.ifft_as_arg(pxy_b_fft, self.tmp_pxy_b)

        return monge_ampere_step1(
            self.tmp_pxx_a, self.tmp_pyy_a, self.tmp_pxy_a,
            self.tmp_pxx_b, self.tmp_pyy_b, self.tmp_pxy_b)


def monge_ampere_from_fft2(a_fft, b_fft, KX, KY, ifft2):

    pxx_a = - ifft2(a_fft * KX**2)
    pyy_a = - ifft2(a_fft * KY**2)
    pxy_a = - ifft2(a_fft * KX * KY)

    pxx_b = - ifft2(b_fft * KX**2)
    pyy_b = - ifft2(b_fft * KY**2)
    pxy_b = - ifft2(b_fft * KX * KY)

    return pxx_a*pyy_b + pyy_a*pxx_b - 2*pxy_a*pxy_b
